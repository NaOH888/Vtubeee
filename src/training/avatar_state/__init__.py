"""AvatarState 回归训练相关组件。"""

from training.avatar_state.dataset import (
    DEFAULT_AVATAR_STATE_FIELDS,
    AvatarStateDataset,
    flatten_avatar_state,
)
from training.avatar_state.model_builder import (
    MODEL_SPECS,
    AvatarStateModelSpec,
    ResolvedModelAssets,
    build_avatar_state_model,
    get_model_spec,
    resolve_model_assets,
)
from training.avatar_state.timm_wrapper import TimmAvatarStateWrapper

__all__ = [
    "AvatarStateModelSpec",
    "AvatarStateDataset",
    "MODEL_SPECS",
    "DEFAULT_AVATAR_STATE_FIELDS",
    "TimmAvatarStateWrapper",
    "ResolvedModelAssets",
    "build_avatar_state_model",
    "flatten_avatar_state",
    "get_model_spec",
    "resolve_model_assets",
]
