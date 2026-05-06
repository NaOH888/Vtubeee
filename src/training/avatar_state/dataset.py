from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from vtuber.avatar.avatar_state import AvatarState


DEFAULT_AVATAR_STATE_FIELDS: tuple[str, ...] = (
    "face.yaw",
    "face.pitch",
    "face.roll",
    "eyes.left_open",
    "eyes.right_open",
    "eyes.gaze_x",
    "eyes.gaze_y",
    "mouth.open",
    "mouth.smile",
    "mouth.form",
    "expression.brow_left_y",
    "expression.brow_right_y",
)


@dataclass(frozen=True)
class AvatarStateSampleIndex:
    """单条样本的轻量索引信息。"""

    dataset_idx: int
    sample_index: int
    avatar_id: str
    profile_id: str
    image_key: bytes
    bbox_xyxy: tuple[int, int, int, int]
    state_vector: tuple[float, ...]


def flatten_avatar_state(
    state: AvatarState,
    fields: tuple[str, ...] = DEFAULT_AVATAR_STATE_FIELDS,
) -> tuple[float, ...]:
    """把 AvatarState 按固定字段顺序展开为数值向量。"""
    return tuple(float(_read_avatar_field(state, field_path)) for field_path in fields)


class AvatarStateDataset(Dataset[dict[str, Any]]):
    """从 LMDB 按需解码、裁切并返回 AvatarState 训练样本。

    当前实现默认使用 `obs` 图像和 `obs_bbox_xyxy`。如果要先做更干净的 baseline，
    可以切到 `vtube` 图像和 `vtube_bbox_xyxy`。
    """

    def __init__(
        self,
        lmdb_paths: str | Path | list[str | Path],
        *,
        image_source: str = "obs",
        bbox_source: str | None = None,
        image_size: int | tuple[int, int] = 256,
        padding_ratio: float = 0.0,
        state_fields: tuple[str, ...] = DEFAULT_AVATAR_STATE_FIELDS,
    ) -> None:
        super().__init__()
        if image_source not in {"obs", "vtube"}:
            raise ValueError("image_source must be 'obs' or 'vtube'.")
        if bbox_source is None:
            bbox_source = image_source
        if bbox_source not in {"obs", "vtube"}:
            raise ValueError("bbox_source must be 'obs' or 'vtube'.")
        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError("image_size must be positive.")
            parsed_image_size = (image_size, image_size)
        else:
            if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
                raise ValueError("image_size must be a positive int or (height, width).")
            parsed_image_size = (int(image_size[0]), int(image_size[1]))
        if padding_ratio < 0.0:
            raise ValueError("padding_ratio must be >= 0.")
        if not state_fields:
            raise ValueError("state_fields must not be empty.")

        raw_paths = lmdb_paths if isinstance(lmdb_paths, list) else [lmdb_paths]
        self.lmdb_paths = [Path(path).resolve() for path in raw_paths]
        self.image_source = image_source
        self.bbox_source = bbox_source
        self.image_size = parsed_image_size
        self.padding_ratio = float(padding_ratio)
        self.state_fields = tuple(state_fields)
        self._envs: list[lmdb.Environment | None] = [None] * len(self.lmdb_paths)
        self.samples: list[AvatarStateSampleIndex] = []

        for dataset_idx, lmdb_path in enumerate(self.lmdb_paths):
            self._collect_dataset_index(dataset_idx, lmdb_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        env = self._ensure_env(sample.dataset_idx)
        with env.begin(write=False) as txn:
            image_bytes = _read_bytes(txn, sample.image_key)

        image_bgr = _decode_jpeg(image_bytes)
        crop_bgr = _crop_with_padding(image_bgr, sample.bbox_xyxy, self.padding_ratio)
        resized_bgr = cv2.resize(
            crop_bgr,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        state_tensor = torch.tensor(sample.state_vector, dtype=torch.float32)

        return {
            "image": image_tensor,
            "target": state_tensor,
            "index": sample.sample_index,
            "avatar_id": sample.avatar_id,
            "profile_id": sample.profile_id,
        }

    def _collect_dataset_index(self, dataset_idx: int, lmdb_path: Path) -> None:
        """读取单个 LMDB 的轻量索引，只缓存必要元信息。"""
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")
        env = lmdb.open(
            str(lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with env.begin(write=False) as txn:
                raw_length = _read_bytes(txn, b"__length__")
                total_length = int(raw_length.decode("utf-8"))
                for sample_index in range(total_length):
                    prefix = f"sample/{sample_index:06d}".encode("utf-8")
                    meta = _read_json(txn, prefix + b"/meta")
                    if "state" not in meta:
                        continue
                    state = AvatarState.from_dict(meta["state"])
                    bbox = meta[f"{self.bbox_source}_bbox_xyxy"]
                    self.samples.append(
                        AvatarStateSampleIndex(
                            dataset_idx=dataset_idx,
                            sample_index=sample_index,
                            avatar_id=str(meta.get("avatar_id", "")),
                            profile_id=str(meta.get("profile_id", "")),
                            image_key=prefix + f"/{self.image_source}".encode("utf-8"),
                            bbox_xyxy=_parse_bbox_xyxy(bbox),
                            state_vector=flatten_avatar_state(state, self.state_fields),
                        )
                    )
        finally:
            env.close()

    def _ensure_env(self, dataset_idx: int) -> lmdb.Environment:
        """在当前 worker 进程内延迟打开只读 LMDB。"""
        env = self._envs[dataset_idx]
        if env is None:
            env = lmdb.open(
                str(self.lmdb_paths[dataset_idx]),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._envs[dataset_idx] = env
        return env


def _read_avatar_field(state: AvatarState, field_path: str) -> float:
    """按 `face.yaw` 这样的路径读取 AvatarState 浮点字段。"""
    current: Any = state
    for part in field_path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"AvatarState has no field path: {field_path}")
        current = getattr(current, part)
    return float(current)


def _read_json(txn: lmdb.Transaction, key: bytes) -> dict[str, Any]:
    """从 LMDB 读取 JSON 对象。"""
    return json.loads(_read_bytes(txn, key).decode("utf-8"))


def _read_bytes(txn: lmdb.Transaction, key: bytes) -> bytes:
    """从 LMDB 读取原始字节。"""
    value = txn.get(key)
    if value is None:
        raise KeyError(f"Missing key: {key.decode('utf-8', errors='ignore')}")
    return value


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    """把 JPEG 字节解码为 BGR 图像。"""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image from LMDB.")
    return image


def _parse_bbox_xyxy(raw_bbox: Any) -> tuple[int, int, int, int]:
    """把 JSON 中的 bbox 规范化为整数四元组。"""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"Invalid bbox format: {raw_bbox!r}")
    return tuple(int(value) for value in raw_bbox)


def _crop_with_padding(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    """按 bbox 裁切图像，并在 bbox 尺寸基础上附加 padding。"""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox values: {bbox_xyxy!r}")

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    pad_x = round(bbox_width * padding_ratio)
    pad_y = round(bbox_height * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Collapsed crop after padding: {bbox_xyxy!r}")
    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
