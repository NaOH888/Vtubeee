from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import cv2
import lmdb


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset.utils import (  # noqa: E402
    bbox_from_mask,
    bbox_touches_border,
    estimate_background_color,
    foreground_mask_from_background,
    map_bbox_between_frames,
    open_camera_grabber,
    yolo_bbox_from_xyxy,
)
from vtuber.avatar import Avatar, AvatarProfile, AvatarState, AvatarTransform  # noqa: E402
from vtuber.avatar.capture import list_directshow_cameras, resolve_camera_source  # noqa: E402


def encode_jpeg(frame, quality: int) -> bytes:
    """把 BGR 图像编码为 JPEG 字节。"""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG.")
    return encoded.tobytes()


def build_config(
    args: argparse.Namespace,
    profile: AvatarProfile,
    avatar_id: str,
) -> dict[str, object]:
    """构造当前采样配置。"""
    return {
        "format": "bbox_dataset_lmdb_v2",
        "avatar_id": avatar_id,
        "profile_id": profile.profile_id,
        "profile_path": str(Path(args.profile).resolve()),
        "host": args.host,
        "port": args.port,
        "token_path": args.token_path,
        "avatar_fps": args.avatar_fps,
        "vtube_camera": args.vtube_camera,
        "obs_camera": args.obs_camera,
        "vtube_backend": args.vtube_backend,
        "obs_backend": args.obs_backend,
        "obs_width": args.obs_width,
        "obs_height": args.obs_height,
        "interval": args.interval,
        "settle_seconds": args.settle_seconds,
        "obs_lag_frames": args.obs_lag_frames,
        "transform_time_seconds": args.transform_time_seconds,
        "transform_x_min": args.transform_x_min,
        "transform_x_max": args.transform_x_max,
        "transform_y_min": args.transform_y_min,
        "transform_y_max": args.transform_y_max,
        "transform_rotation_min": args.transform_rotation_min,
        "transform_rotation_max": args.transform_rotation_max,
        "transform_size_min": args.transform_size_min,
        "transform_size_max": args.transform_size_max,
        "warmup": args.warmup,
        "capture_timeout": args.capture_timeout,
        "threshold": args.threshold,
        "padding_ratio": args.padding_ratio,
        "border_ratio": args.border_ratio,
        "min_area_ratio": args.min_area_ratio,
        "jpeg_quality": args.jpeg_quality,
        "commit_every": args.commit_every,
        "state_seed": args.state_seed,
    }


def open_env(output_path: Path, map_size_mb: int) -> lmdb.Environment:
    """打开 LMDB 数据库目录。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(output_path),
        map_size=map_size_mb * 1024 * 1024,
        subdir=True,
        create=True,
        lock=True,
        readonly=False,
        readahead=False,
        meminit=False,
    )


def read_length(env: lmdb.Environment) -> int:
    """读取当前样本数。"""
    with env.begin(write=False) as txn:
        raw = txn.get(b"__length__")
    if raw is None:
        return 0
    return int(raw.decode("utf-8"))


def write_batch(
    env: lmdb.Environment,
    pending: list[tuple[int, bytes, bytes, bytes]],
    total_length: int,
    config_json: bytes,
) -> None:
    """把缓存中的样本批量写入 LMDB。"""
    if not pending:
        return

    while True:
        try:
            with env.begin(write=True) as txn:
                if txn.get(b"__config__") is None:
                    txn.put(b"__config__", config_json)
                for sample_index, obs_bytes, vtube_bytes, meta_bytes in pending:
                    prefix = f"sample/{sample_index:06d}".encode("utf-8")
                    txn.put(prefix + b"/obs", obs_bytes)
                    txn.put(prefix + b"/vtube", vtube_bytes)
                    txn.put(prefix + b"/meta", meta_bytes)
                txn.put(b"__length__", str(total_length).encode("utf-8"))
            return
        except lmdb.MapFullError:
            current_size = env.info()["map_size"]
            new_size = current_size * 2
            env.set_mapsize(new_size)
            print(
                f"LMDB map_size full, grow from {current_size / (1024**2):.0f}MB "
                f"to {new_size / (1024**2):.0f}MB.",
                flush=True,
            )


async def run(args: argparse.Namespace) -> None:
    """运行主动驱动的双路 bbox 数据采样，并写入 LMDB。"""
    if args.list_cameras:
        print("DirectShow cameras:", flush=True)
        for index, name in enumerate(list_directshow_cameras()):
            print(f"  {index}: {name}", flush=True)
        return

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    profile = AvatarProfile.from_file(args.profile)
    avatar_id = args.avatar_id or profile.profile_id
    vtube_index = resolve_camera_source(args.vtube_camera)
    obs_index = resolve_camera_source(args.obs_camera)
    print(f"Avatar profile: {profile.profile_id}", flush=True)
    print(f"VTube camera {args.vtube_camera!r} -> index {vtube_index}", flush=True)
    print(f"OBS camera {args.obs_camera!r} -> index {obs_index}", flush=True)

    env = open_env(output_path, args.map_size_mb)
    start_index = read_length(env)
    next_index = start_index
    pending: list[tuple[int, bytes, bytes, bytes]] = []
    rng = random.Random(args.state_seed)
    config = build_config(args, profile, avatar_id)
    config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")

    vtube_grabber = open_camera_grabber(
        vtube_index,
        args.capture_timeout,
        args.vtube_backend,
    )
    obs_grabber = open_camera_grabber(
        obs_index,
        args.capture_timeout,
        args.obs_backend,
        width=args.obs_width,
        height=args.obs_height,
    )

    try:
        async with Avatar(
            backend="vtube_studio",
            host=args.host,
            port=args.port,
            token_path=args.token_path,
            fps=args.avatar_fps,
            profile=profile,
        ) as avatar:
            await avatar.ensure_model_loaded()
            await avatar.reset_neutral()
            await avatar.reset_transform()
            await avatar.flush_once()
            await asyncio.sleep(args.settle_seconds)

            for _ in range(args.warmup):
                vtube_grabber.capture()
                obs_grabber.capture()
                await asyncio.sleep(0.03)

            mapper = avatar.get_driver().mapper
            for _ in range(args.count):
                state = AvatarState.sample_random(rng)
                transform = AvatarTransform.sample_random(
                    rng,
                    x_min=args.transform_x_min,
                    x_max=args.transform_x_max,
                    y_min=args.transform_y_min,
                    y_max=args.transform_y_max,
                    rotation_min=args.transform_rotation_min,
                    rotation_max=args.transform_rotation_max,
                    size_min=args.transform_size_min,
                    size_max=args.transform_size_max,
                )
                await avatar.set_transform(
                    transform,
                    time_in_seconds=args.transform_time_seconds,
                )
                await avatar.set_state(state)
                await avatar.flush_once()
                await asyncio.sleep(args.settle_seconds)

                vtube_frame = vtube_grabber.capture()
                obs_frame = obs_grabber.capture()
                for _ in range(args.obs_lag_frames):
                    obs_frame = obs_grabber.capture()
                background_color = estimate_background_color(vtube_frame, args.border_ratio)
                mask = foreground_mask_from_background(vtube_frame, background_color, args.threshold)
                vtube_bbox = bbox_from_mask(mask, args.padding_ratio, args.min_area_ratio)
                obs_bbox = map_bbox_between_frames(vtube_bbox, vtube_frame.shape, obs_frame.shape)

                sample_index = next_index
                next_index += 1
                backend_parameters = {
                    key: float(value)
                    for key, value in mapper.to_parameters(state).items()
                }
                meta = {
                    "index": sample_index,
                    "avatar_id": avatar_id,
                    "profile_id": profile.profile_id,
                    "state": state.to_dict(),
                    "transform": transform.to_dict(),
                    "backend_parameters": backend_parameters,
                    "vtube_bbox_xyxy": list(vtube_bbox),
                    "obs_bbox_xyxy": list(obs_bbox),
                    "vtube_shape": list(vtube_frame.shape),
                    "obs_shape": list(obs_frame.shape),
                    "background_bgr": [int(value) for value in background_color.tolist()],
                    "obs_bbox_yolo": list(yolo_bbox_from_xyxy(obs_bbox, obs_frame.shape)),
                    "touches_border": list(bbox_touches_border(obs_bbox, obs_frame.shape)),
                    "storage": {
                        "obs": "jpeg",
                        "vtube": "jpeg",
                        "jpeg_quality": args.jpeg_quality,
                    },
                    "config": config,
                }
                pending.append(
                    (
                        sample_index,
                        encode_jpeg(obs_frame, args.jpeg_quality),
                        encode_jpeg(vtube_frame, args.jpeg_quality),
                        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                    )
                )

                if len(pending) >= args.commit_every:
                    write_batch(env, pending, next_index, config_json)
                    print(
                        f"Committed {len(pending)} samples, total={next_index}, "
                        f"last_index={sample_index}",
                        flush=True,
                    )
                    pending.clear()

                if args.interval > 0.0:
                    await asyncio.sleep(args.interval)

            await avatar.reset_neutral()
            await avatar.reset_transform()
            await avatar.flush_once()

        if pending:
            write_batch(env, pending, next_index, config_json)
            print(
                f"Committed {len(pending)} samples, total={next_index}, "
                f"last_index={next_index - 1}",
                flush=True,
            )
            pending.clear()
    finally:
        vtube_grabber.close()
        obs_grabber.close()
        env.sync()
        env.close()

    print(f"Saved LMDB dataset: {output_path}", flush=True)
    print(f"Samples before run: {start_index}", flush=True)
    print(f"Samples added: {next_index - start_index}", flush=True)
    print(f"Samples after run: {next_index}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actively drive VTube Studio, capture VTube/OBS images, and save bbox/state data to LMDB."
    )
    parser.add_argument("--profile", required=True, help="Avatar profile JSON or .vtube.json path.")
    parser.add_argument("--avatar-id", default=None, help="Optional avatar id stored in metadata.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--token-path", default=".vts_token.json")
    parser.add_argument("--avatar-fps", type=float, default=30.0)
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
    parser.add_argument("--output", default="outputs/bbox_dataset_samples_hiyori.lmdb")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--obs-width", type=int, default=1920)
    parser.add_argument("--obs-height", type=int, default=1080)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    parser.add_argument("--obs-lag-frames", type=int, default=1)
    parser.add_argument("--transform-time-seconds", type=float, default=0.0)
    parser.add_argument("--transform-x-min", type=float, default=-0.8)
    parser.add_argument("--transform-x-max", type=float, default=0.8)
    parser.add_argument("--transform-y-min", type=float, default=-0.7)
    parser.add_argument("--transform-y-max", type=float, default=0.3)
    parser.add_argument("--transform-rotation-min", type=float, default=-20.0)
    parser.add_argument("--transform-rotation-max", type=float, default=20.0)
    parser.add_argument("--transform-size-min", type=float, default=-90.0)
    parser.add_argument("--transform-size-max", type=float, default=-75.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--capture-timeout", type=float, default=3.0)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--border-ratio", type=float, default=0.03)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--commit-every", type=int, default=64)
    parser.add_argument("--map-size-mb", type=int, default=256)
    parser.add_argument("--state-seed", type=int, default=42)
    parser.add_argument("--list-cameras", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
