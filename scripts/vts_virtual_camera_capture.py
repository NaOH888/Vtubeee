from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from vtuber.avatar import (  # noqa: E402
    Avatar,
    AvatarCapture,
    AvatarError,
    AvatarState,
    CaptureError,
    CapturedFrame,
    FacePose,
    MouthState,
    list_directshow_cameras,
)


DEFAULT_SINGLE_OUTPUT = "outputs/vtube_virtual_camera_capture.png"
DEFAULT_SEQUENCE_OUTPUT = "outputs/vtube_virtual_camera_sequence.mp4"


def resolve_output_path(output: str | None, default_output: str) -> Path:
    """解析脚本输出路径；相对路径固定落在仓库根目录下。"""
    output_path = Path(output or default_output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def sample_pose() -> AvatarState:
    """返回单张截图 demo 使用的非中性姿态。"""
    return AvatarState(
        face=FacePose(yaw=0.55, pitch=0.08, roll=-0.18),
        mouth=MouthState(open=0.65, smile=0.35),
    )


def sample_sequence_frames(fps: float, seconds: float) -> list[AvatarState]:
    """返回 sequence demo 使用的逐帧姿态列表。

    sequence 的正式输入就是 `list[AvatarState]`。这里用三角函数只是为了快速生成
    一段摇头和张嘴 demo；后续数据集采样时可以直接替换为任意逐帧状态列表。
    """
    if fps <= 0:
        raise ValueError("sequence fps must be positive.")
    if seconds <= 0:
        raise ValueError("sequence seconds must be positive.")

    total_frames = max(1, round(fps * seconds))
    states: list[AvatarState] = []
    for frame_index in range(total_frames):
        t = frame_index / fps
        head_phase = 2.0 * math.pi * 0.7 * t
        mouth_phase = 2.0 * math.pi * 2.4 * t
        states.append(
            AvatarState(
                face=FacePose(
                    yaw=0.55 * math.sin(head_phase),
                    pitch=0.08 * math.sin(head_phase * 0.5),
                    roll=-0.16 * math.sin(head_phase),
                ),
                mouth=MouthState(
                    open=0.18 + 0.52 * (0.5 + 0.5 * math.sin(mouth_phase)),
                    smile=0.25,
                ),
            )
        )
    return states


def create_avatar(args: argparse.Namespace) -> Avatar:
    """按 CLI 参数创建 avatar facade。"""
    if args.profile is None:
        raise ValueError("VTube Studio capture requires --profile.")
    return Avatar(
        backend="vtube_studio",
        host=args.host,
        port=args.port,
        token_path=REPO_ROOT / ".vts_token.json",
        fps=args.fps,
        profile_path=args.profile,
    )


def create_capture(args: argparse.Namespace) -> AvatarCapture:
    """按 CLI 参数创建 avatar 捕获器。"""
    return AvatarCapture(
        camera=args.camera,
        capture_timeout=args.capture_timeout,
        settle_seconds=args.settle_seconds,
        frame_settle_seconds=args.frame_settle_seconds,
    )


async def run_single(args: argparse.Namespace) -> None:
    """设置一个 demo 姿态，并从 VTube Studio 虚拟摄像头截取单张图片。"""
    output_path = resolve_output_path(args.output, DEFAULT_SINGLE_OUTPUT)
    avatar = create_avatar(args)
    capture = create_capture(args)

    try:
        print("Connecting and authenticating...", flush=True)
        await asyncio.wait_for(avatar.start(), timeout=args.auth_timeout)
        model = await avatar.ensure_model_loaded()
        print("Model:", model["modelName"], flush=True)

        frame = await capture.capture_image_to_file(avatar, sample_pose(), output_path)
        print(f"Saved frame: {output_path}", flush=True)
        print(f"Frame shape: {frame.image.shape}", flush=True)

        await avatar.reset_neutral()
    except TimeoutError:
        print(
            "Authentication timed out. Please allow the plugin popup in VTube Studio.",
            flush=True,
        )
    except (AvatarError, CaptureError, RuntimeError, ValueError) as exc:
        print(f"Capture error: {exc}", flush=True)
    finally:
        await avatar.stop()


async def run_sequence(args: argparse.Namespace) -> None:
    """播放逐帧姿态列表，并从 VTube Studio 虚拟摄像头录制一段视频。"""
    output_path = resolve_output_path(args.output, DEFAULT_SEQUENCE_OUTPUT)
    states = sample_sequence_frames(fps=args.sequence_fps, seconds=args.sequence_seconds)
    avatar = create_avatar(args)
    capture = create_capture(args)
    progress_interval = max(1, round(args.sequence_fps))

    def print_progress(frame: CapturedFrame) -> None:
        frame_number = frame.index + 1
        should_print = (
            frame.index == 0
            or frame_number % progress_interval == 0
            or frame_number == len(states)
        )
        if should_print:
            print(f"Captured {frame_number}/{len(states)} frames", flush=True)

    try:
        print("Connecting and authenticating...", flush=True)
        await asyncio.wait_for(avatar.start(), timeout=args.auth_timeout)
        model = await avatar.ensure_model_loaded()
        print("Model:", model["modelName"], flush=True)
        print(f"Recording {len(states)} frames at {args.sequence_fps:g} FPS...", flush=True)

        result = await capture.capture_sequence_to_file(
            avatar,
            states,
            output_path,
            fps=args.sequence_fps,
            progress=print_progress,
        )
        print(f"Saved video: {result.output_path}", flush=True)
        print(f"Saved metadata: {result.metadata_path}", flush=True)
        print(f"Video: {result.width}x{result.height}, {result.frame_count} frames", flush=True)

        await avatar.reset_neutral()
    except TimeoutError:
        print(
            "Authentication timed out. Please allow the plugin popup in VTube Studio.",
            flush=True,
        )
    except (AvatarError, CaptureError, RuntimeError, ValueError) as exc:
        print(f"Capture error: {exc}", flush=True)
    finally:
        await avatar.stop()


async def run(args: argparse.Namespace) -> None:
    if args.list_cameras:
        print("DirectShow cameras:", flush=True)
        for index, name in enumerate(list_directshow_cameras()):
            print(f"  {index}: {name}", flush=True)
        return

    if args.profile is None:
        print("Capture error: --profile is required for VTube Studio capture.", flush=True)
        return

    if args.mode == "single":
        await run_single(args)
        return
    if args.mode == "sequence":
        await run_sequence(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture rendered VTube Studio output from its virtual camera."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sequence"],
        default="sequence",
        help="single captures one image; sequence records a short demo video.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument(
        "--camera",
        default="VTubeStudioCam",
        help="DirectShow camera index or device name, e.g. VTubeStudioCam.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image/video path. Defaults depend on --mode.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--profile",
        default=None,
        help="Required for capture: Avatar profile JSON or VTube Studio .vtube.json path.",
    )
    parser.add_argument("--sequence-fps", type=float, default=60.0)
    parser.add_argument("--sequence-seconds", type=float, default=3.0)
    parser.add_argument("--auth-timeout", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--frame-settle-seconds", type=float, default=0.04)
    parser.add_argument("--capture-timeout", type=float, default=3.0)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List DirectShow camera devices and exit.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
