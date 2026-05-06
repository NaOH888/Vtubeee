from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def read_json(txn: lmdb.Transaction, key: bytes) -> dict:
    """读取 JSON 键值。"""
    raw = txn.get(key)
    if raw is None:
        raise KeyError(f"Missing key: {key.decode('utf-8', errors='ignore')}")
    return json.loads(raw.decode("utf-8"))


def read_bytes(txn: lmdb.Transaction, key: bytes) -> bytes:
    """读取字节键值。"""
    raw = txn.get(key)
    if raw is None:
        raise KeyError(f"Missing key: {key.decode('utf-8', errors='ignore')}")
    return raw


def decode_jpeg(image_bytes: bytes) -> np.ndarray:
    """从 JPEG 字节解码 BGR 图像。"""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image from LMDB.")
    return image


def draw_bbox(frame: np.ndarray, bbox_xyxy: list[int], label: str) -> np.ndarray:
    """在图像上绘制 bbox。"""
    output = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        output,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """生成左右对照图。"""
    target_height = min(left.shape[0], right.shape[0], 720)
    left_width = round(left.shape[1] * target_height / left.shape[0])
    right_width = round(right.shape[1] * target_height / right.shape[0])
    left_small = cv2.resize(left, (left_width, target_height))
    right_small = cv2.resize(right, (right_width, target_height))
    return np.concatenate([left_small, right_small], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LMDB bbox dataset contents.")
    parser.add_argument("lmdb_path", help="Path to the LMDB dataset directory.")
    parser.add_argument("--show-fig", action="store_true", help="Show one sample in a window.")
    parser.add_argument("--fig-index", type=int, default=0, help="Sample index to visualize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.is_absolute():
        lmdb_path = REPO_ROOT / lmdb_path
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
            raw_length = read_bytes(txn, b"__length__")
            length = int(raw_length.decode("utf-8"))
            config = read_json(txn, b"__config__")
            print(f"Inspecting: {lmdb_path}", flush=True)
            print("Format: bbox_dataset_lmdb_v1", flush=True)
            print(f"length: {length}", flush=True)
            print(f"config: {config}", flush=True)

            if length <= 0:
                return

            first_meta = read_json(txn, b"sample/000000/meta")
            print(f"sample[0].meta: {first_meta}", flush=True)
            if "state" not in first_meta:
                print("sample[0] has no `state` field. This is old recorded data.", flush=True)

            if args.fig_index < 0 or args.fig_index >= length:
                raise ValueError(
                    f"fig-index out of range: {args.fig_index}, sample_count={length}"
                )

            if not args.show_fig:
                return

            sample_prefix = f"sample/{args.fig_index:06d}".encode("utf-8")
            meta = read_json(txn, sample_prefix + b"/meta")
            if "state" not in meta:
                print(
                    f"sample[{args.fig_index}] has no `state` field. "
                    "This sample should be skipped in state training.",
                    flush=True,
                )
            obs_image = decode_jpeg(read_bytes(txn, sample_prefix + b"/obs"))
            vtube_image = decode_jpeg(read_bytes(txn, sample_prefix + b"/vtube"))

        obs_debug = draw_bbox(obs_image, meta["obs_bbox_xyxy"], f"OBS #{args.fig_index}")
        vtube_debug = draw_bbox(vtube_image, meta["vtube_bbox_xyxy"], f"VTube #{args.fig_index}")
        canvas = make_side_by_side(vtube_debug, obs_debug)
        window_name = f"bbox_lmdb_inspect_{args.fig_index}"
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Inspect error: {exc}", flush=True)
        sys.exit(1)
