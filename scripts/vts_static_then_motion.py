from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from vtuber.avatar import Avatar, AvatarError, AvatarState, FacePose, MouthState  # noqa: E402


def shake_and_talk_state(elapsed: float) -> AvatarState:
    return AvatarState(
        face=FacePose(
            yaw=math.sin(elapsed * math.tau * 0.55) * (25.0 / 30.0),
            pitch=math.sin(elapsed * math.tau * 0.35) * (6.0 / 30.0),
            roll=math.sin(elapsed * math.tau * 0.45) * (10.0 / 30.0),
        ),
        mouth=MouthState(
            open=(math.sin(elapsed * math.tau * 1.8) + 1.0) * 0.5,
            smile=0.25,
        ),
    )


async def run(args: argparse.Namespace) -> None:
    avatar = Avatar(
        backend="vtube_studio",
        host=args.host,
        port=args.port,
        token_path=REPO_ROOT / ".vts_token.json",
        fps=args.fps,
        profile_path=args.profile,
    )

    try:
        print("Connecting and authenticating...", flush=True)
        await asyncio.wait_for(avatar.start(), timeout=args.auth_timeout)

        state = await avatar.get_backend_state()
        print(
            "VTube Studio API:",
            f"active={state['active']}",
            f"version={state['vTubeStudioVersion']}",
            flush=True,
        )

        model = await avatar.ensure_model_loaded()
        print("Model loaded:", model["modelLoaded"], flush=True)
        print("Model:", model["modelName"], flush=True)

        print(f"Static neutral pose: {args.static_seconds:.1f}s", flush=True)
        await avatar.hold_state(AvatarState.neutral(), args.static_seconds)

        print(f"Shake head and open mouth: {args.motion_seconds:.1f}s", flush=True)
        await avatar.play(shake_and_talk_state, args.motion_seconds)
        await avatar.reset_neutral()
        print("Done.", flush=True)
    except TimeoutError:
        print(
            "Authentication timed out. Please allow the plugin popup in VTube Studio.",
            flush=True,
        )
    except (AvatarError, ValueError) as exc:
        print(f"Avatar error: {exc}", flush=True)
    finally:
        await avatar.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep VTube Studio still for 5s, then shake head/open mouth for 5s."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25565)
    parser.add_argument("--static-seconds", type=float, default=5.0)
    parser.add_argument("--motion-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--profile",
        required=True,
        default=None,
        help="Required: Avatar profile JSON or VTube Studio .vtube.json path.",
    )
    parser.add_argument("--auth-timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
