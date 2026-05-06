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

try:
    from vtuber.avatar.capture import (
        AvatarCapture,
        CaptureError,
        CapturedFrame,
        CaptureSequenceResult,
        list_directshow_cameras,
    )
    _CAPTURE_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    _CAPTURE_IMPORT_ERROR = exc

    def _raise_capture_import_error() -> None:
        raise ModuleNotFoundError(
            "Avatar capture utilities depend on optional runtime capture "
            f"dependencies that are not installed: {_CAPTURE_IMPORT_ERROR}"
        ) from _CAPTURE_IMPORT_ERROR

    class _MissingCaptureDependency:
        def __init__(self, *args, **kwargs) -> None:
            _raise_capture_import_error()

    AvatarCapture = _MissingCaptureDependency  # type: ignore[assignment]
    CaptureError = _MissingCaptureDependency  # type: ignore[assignment]
    CapturedFrame = _MissingCaptureDependency  # type: ignore[assignment]
    CaptureSequenceResult = _MissingCaptureDependency  # type: ignore[assignment]

    def list_directshow_cameras(*args, **kwargs):
        _raise_capture_import_error()

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
