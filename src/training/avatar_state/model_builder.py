from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from training.avatar_state.timm_wrapper import TimmAvatarStateWrapper


@dataclass(frozen=True)
class AvatarStateModelSpec:
    """AvatarState 模型配置项。"""

    name: str
    description: str
    family: str
    pretrained_source: str | None = None


@dataclass(frozen=True)
class ResolvedModelAssets:
    """模型构造时实际使用的预训练来源。"""

    pretrained_source: str | None = None
    local_pretrained_path: str | None = None


MODEL_SPECS: dict[str, AvatarStateModelSpec] = {
    "timm_convnextv2_tiny": AvatarStateModelSpec(
        name="timm_convnextv2_tiny",
        description="使用 timm ConvNeXtV2-Tiny ImageNet 预训练模型作为主干。",
        family="timm",
        pretrained_source="convnextv2_tiny.fcmae_ft_in22k_in1k",
    ),
    "timm_convnextv2_base": AvatarStateModelSpec(
        name="timm_convnextv2_base",
        description="使用 timm ConvNeXtV2-Base ImageNet 预训练模型作为主干。",
        family="timm",
        pretrained_source="convnextv2_base.fcmae_ft_in22k_in1k",
    ),
    "timm_efficientnetv2_rw_s": AvatarStateModelSpec(
        name="timm_efficientnetv2_rw_s",
        description="使用 timm EfficientNetV2-RW-S ImageNet 预训练模型作为主干。",
        family="timm",
        pretrained_source="efficientnetv2_rw_s",
    ),
}


def get_model_spec(model_name: str) -> AvatarStateModelSpec:
    """按模型名读取配置项。"""
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model_name: {model_name}. Available: {available}") from exc


def resolve_model_assets(
    model_name: str,
    *,
    pretrained_path: str | Path | None = None,
) -> ResolvedModelAssets:
    """解析模型所需资产。"""
    spec = get_model_spec(model_name)
    local_path = None
    if pretrained_path is not None:
        resolved_path = Path(pretrained_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pretrained path does not exist: {resolved_path}")
        local_path = str(resolved_path)
    return ResolvedModelAssets(
        pretrained_source=spec.pretrained_source,
        local_pretrained_path=local_path,
    )


def build_avatar_state_model(
    model_name: str,
    *,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    pretrained_path: str | Path | None = None,
    use_pretrained_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, ResolvedModelAssets]:
    """根据配置构造 AvatarState 回归模型，并返回实际使用的预训练来源。"""
    spec = get_model_spec(model_name)
    assets = resolve_model_assets(model_name, pretrained_path=pretrained_path)
    if spec.family != "timm":
        raise ValueError(f"Unsupported model family: {spec.family}")
    model = _build_timm_wrapper(
        timm_model_name=assets.pretrained_source or spec.pretrained_source,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        local_pretrained_path=assets.local_pretrained_path,
        use_pretrained_backbone=use_pretrained_backbone,
        device=device,
    )
    return model, assets


def _build_timm_wrapper(
    *,
    timm_model_name: str | None,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    local_pretrained_path: str | None,
    use_pretrained_backbone: bool,
    device: str | torch.device,
) -> nn.Module:
    """从 timm 预训练模型构造 AvatarState 包装器。"""
    if not timm_model_name:
        raise ValueError("Missing timm model name.")
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import `timm`. Install timm in the current venv first."
        ) from exc

    backbone = timm.create_model(
        timm_model_name,
        pretrained=use_pretrained_backbone and local_pretrained_path is None,
        features_only=True,
    )
    if local_pretrained_path is not None:
        _load_local_timm_weights(backbone, local_pretrained_path, device)
    wrapper = TimmAvatarStateWrapper(
        backbone=backbone,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return wrapper.to(device)


def _load_local_timm_weights(
    backbone: nn.Module,
    pretrained_path: str,
    device: str | torch.device,
) -> None:
    """加载本地 timm/backbone 权重。

    支持：
    - timm 常见 `state_dict` / `model` 格式
    - 本项目完整 checkpoint 中的 `model_state_dict`，会自动截取 `backbone.` 前缀
    - `.safetensors`
    """
    path = Path(pretrained_path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import safetensors. Install `safetensors` in the current venv first."
            ) from exc
        state_dict = load_file(str(path), device=str(device))
    else:
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict):
            if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                state_dict = payload["state_dict"]
            elif "model" in payload and isinstance(payload["model"], dict):
                state_dict = payload["model"]
            elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
                model_state = payload["model_state_dict"]
                if any(key.startswith("backbone.") for key in model_state):
                    state_dict = {
                        key.removeprefix("backbone."): value
                        for key, value in model_state.items()
                        if key.startswith("backbone.")
                    }
                else:
                    state_dict = model_state
            else:
                state_dict = payload
        else:
            raise TypeError(
                "Unsupported pretrained payload type from local path: "
                f"{type(payload).__name__}"
            )

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    if any(key.startswith("backbone.") for key in state_dict):
        state_dict = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    if not state_dict:
        raise ValueError(f"No usable state_dict found in local pretrained path: {path}")
    if len(state_dict) > 0 and len(state_dict) == len(unexpected_keys):
        raise RuntimeError(
            "Loaded local pretrained file, but all keys were unexpected for the timm backbone. "
            f"path={path}"
        )
