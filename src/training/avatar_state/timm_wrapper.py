from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class TimmAvatarStateWrapper(nn.Module):
    """把 timm 主干包装为 AvatarState 回归模型。"""

    def __init__(
        self,
        backbone: nn.Module,
        output_dim: int,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head: nn.Module | None = None

    def freeze_backbone(self) -> None:
        """冻结 backbone，常用于先只训练回归头。"""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """解冻 backbone。"""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward_features(self, inputs: Tensor) -> Tensor:
        """提取并规整特征图，统一成 NCHW 四维张量。"""
        features = self.backbone(inputs)
        return self._select_feature_map(features)

    def forward(self, inputs: Tensor) -> Tensor:
        """前向推理，输出 AvatarState 向量。"""
        feature_map = self.forward_features(inputs)
        pooled = self.global_pool(feature_map).flatten(1)
        head = self._get_or_build_head(pooled.shape[1], pooled.device, pooled.dtype)
        return head(pooled)

    def _get_or_build_head(
        self,
        in_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """按第一次真实输入的通道数延迟构造回归头。"""
        if self.regression_head is None:
            self.regression_head = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            ).to(device=device, dtype=dtype)
        return self.regression_head

    @staticmethod
    def _select_feature_map(features: object) -> Tensor:
        """从 timm 不同返回格式中挑出最后一级特征图。"""
        if isinstance(features, Tensor):
            return TimmAvatarStateWrapper._ensure_4d(features)
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Empty feature sequence returned by timm backbone.")
            return TimmAvatarStateWrapper._select_feature_map(features[-1])
        raise TypeError(
            "Unsupported feature type from timm backbone: "
            f"{type(features).__name__}"
        )

    @staticmethod
    def _ensure_4d(feature_map: Tensor) -> Tensor:
        """确保特征图是 NCHW 四维张量。"""
        if feature_map.ndim != 4:
            raise ValueError(
                "Expected feature map with shape [N, C, H, W], "
                f"got ndim={feature_map.ndim}."
            )
        return feature_map


def collect_trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """收集当前可训练参数，便于训练脚本显式传给优化器。"""
    return (parameter for parameter in module.parameters() if parameter.requires_grad)
