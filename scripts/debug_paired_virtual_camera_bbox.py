from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset.utils import (  # noqa: E402
    BBox,
    collect_bbox_transfer_samples,
    open_camera_grabber,
)
from vtuber.avatar.capture import (  # noqa: E402
    CaptureError,
    list_directshow_cameras,
    resolve_camera_source,
)


def draw_bbox(frame: np.ndarray, bbox: BBox, label: str) -> np.ndarray:
    """在图像上绘制 bbox 调试框。"""
    output = frame.copy()
    x1, y1, x2, y2 = bbox
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
    """生成左右对照图，便于快速肉眼检查两路 bbox 是否一致。"""
    target_height = min(left.shape[0], right.shape[0], 720)
    left_width = round(left.shape[1] * target_height / left.shape[0])
    right_width = round(right.shape[1] * target_height / right.shape[0])
    left_small = cv2.resize(left, (left_width, target_height))
    right_small = cv2.resize(right, (right_width, target_height))
    return np.concatenate([left_small, right_small], axis=1)


def save_debug_pair(output_dir: Path, sample: Any) -> dict[str, Any]:
    """保存单次采样的两路画框图片和 mask。"""
    vtube_path = output_dir / f"vtube_{sample.index:04d}.png"
    obs_path = output_dir / f"obs_{sample.index:04d}.png"
    mask_path = output_dir / f"mask_{sample.index:04d}.png"
    side_by_side_path = output_dir / f"pair_{sample.index:04d}.png"

    vtube_debug = draw_bbox(sample.vtube_frame, sample.vtube_bbox_xyxy, "VTube bbox")
    obs_debug = draw_bbox(sample.obs_frame, sample.obs_bbox_xyxy, "OBS mapped bbox")

    cv2.imwrite(str(vtube_path), vtube_debug)
    cv2.imwrite(str(obs_path), obs_debug)
    cv2.imwrite(str(mask_path), sample.mask)
    cv2.imwrite(str(side_by_side_path), make_side_by_side(vtube_debug, obs_debug))

    row = sample.to_metadata()
    row.update(
        {
            "vtube_image": str(vtube_path),
            "obs_image": str(obs_path),
            "mask_image": str(mask_path),
            "pair_image": str(side_by_side_path),
        }
    )
    return row


def write_metadata(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """写入 jsonl metadata，记录每次采样的 bbox 和文件路径。"""
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    """运行双虚拟摄像头 bbox 对齐测试。"""
    if args.list_cameras:
        print("DirectShow cameras:", flush=True)
        for index, name in enumerate(list_directshow_cameras()):
            print(f"  {index}: {name}", flush=True)
        return

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vtube_index = resolve_camera_source(args.vtube_camera)
    obs_index = resolve_camera_source(args.obs_camera)
    print(f"VTube camera {args.vtube_camera!r} -> index {vtube_index}", flush=True)
    print(f"OBS camera {args.obs_camera!r} -> index {obs_index}", flush=True)

    vtube_grabber = open_camera_grabber(vtube_index, args.capture_timeout, args.vtube_backend)
    obs_grabber = open_camera_grabber(obs_index, args.capture_timeout, args.obs_backend)
    rows: list[dict[str, Any]] = []

    try:
        samples = collect_bbox_transfer_samples(
            vtube_grabber=vtube_grabber,
            obs_grabber=obs_grabber,
            count=args.count,
            interval=args.interval,
            warmup=args.warmup,
            threshold=args.threshold,
            padding_ratio=args.padding_ratio,
            border_ratio=args.border_ratio,
            min_area_ratio=args.min_area_ratio,
            bbox_lag_frames=args.bbox_lag_frames,
        )
    finally:
        vtube_grabber.close()
        obs_grabber.close()

    for sample in samples:
        row = save_debug_pair(output_dir, sample)
        rows.append(row)
        print(
            f"Saved sample {sample.index + 1}/{args.count}: "
            f"source_step={sample.vtube_bbox_source_step} -> obs_step={sample.obs_capture_step}, "
            f"vtube_bbox={sample.vtube_bbox_xyxy}, obs_bbox={sample.obs_bbox_xyxy}",
            flush=True,
        )

    write_metadata(output_dir, rows)
    print(f"Saved debug output: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug bbox transfer from VTubeStudioCam to OBS Virtual Camera."
    )
    parser.add_argument("--vtube-camera", default="VTubeStudioCam")
    parser.add_argument("--obs-camera", default="OBS Virtual Camera")
    parser.add_argument(
        "--vtube-backend",
        choices=["auto", "directshow", "opencv"],
        default="directshow",
    )
    parser.add_argument(
        "--obs-backend",
        choices=["auto", "directshow", "opencv"],
        default="auto",
    )
    parser.add_argument("--output", default="outputs/paired_virtual_camera_bbox_debug")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--capture-timeout", type=float, default=3.0)
    parser.add_argument("--bbox-lag-frames", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--border-ratio", type=float, default=0.03)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--list-cameras", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except (CaptureError, RuntimeError, ValueError) as exc:
        print(f"Capture error: {exc}", flush=True)


if __name__ == "__main__":
    main()
