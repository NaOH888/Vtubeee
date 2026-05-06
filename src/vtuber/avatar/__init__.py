"""Avatar-side canonical state definitions."""

from vtuber.avatar.avatar import Avatar, AvatarError, AvatarPolicy
from vtuber.avatar.avatar_profile import AvatarParameterMapping, AvatarProfile
from vtuber.avatar.avatar_state import (
    AvatarState,
    BodyState,
    ExpressionState,
    EyeState,
    FacePose,
    MouthState,
    SpeechState,
)
from vtuber.avatar.avatar_transform import AvatarTransform
from vtuber.avatar.capture import (
    AvatarCapture,
    CaptureError,
    CapturedFrame,
    CaptureSequenceResult,
    list_directshow_cameras,
)

__all__ = [
    "Avatar",
    "AvatarCapture",
    "AvatarError",
    "AvatarParameterMapping",
    "AvatarPolicy",
    "AvatarProfile",
    "AvatarState",
    "AvatarTransform",
    "BodyState",
    "CaptureError",
    "CaptureSequenceResult",
    "CapturedFrame",
    "ExpressionState",
    "EyeState",
    "FacePose",
    "list_directshow_cameras",
    "MouthState",
    "SpeechState",
]
