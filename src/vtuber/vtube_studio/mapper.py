from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtuber.avatar.avatar_profile import AvatarParameterMapping, AvatarProfile
from vtuber.avatar.avatar_state import AvatarState
from vtuber.vtube_studio.runtime import ParameterFrame


@dataclass(frozen=True)
class VTubeParameterMapper:
    """把项目标准 avatar 状态映射到 VTube Studio tracking 参数。

    mapper 本身不再硬编码某个模型的参数范围，而是读取 `AvatarProfile` 中的映射规则。
    没有出现在 profile 里的 `AvatarState` 字段不会输出到 VTube Studio，这样可以避免
    发送对当前模型无效的参数。
    """

    profile: AvatarProfile | Path | str | dict[str, Any]

    def __post_init__(self) -> None:
        """归一化 profile 输入。

        VTube Studio 模型差异必须通过 `AvatarProfile` 明确描述；mapper 不再提供
        无 profile 的默认映射。
        """
        profile = AvatarProfile.coerce(self.profile)
        if profile.backend != "vtube_studio":
            raise ValueError(
                f"VTubeParameterMapper requires vtube_studio profile, got {profile.backend!r}."
            )
        profile.validate()
        object.__setattr__(self, "profile", profile)

    def to_parameters(self, state: AvatarState) -> ParameterFrame:
        """转换为 VTube Studio 可注入的参数字典。

        Args:
            state: 项目内部标准 avatar 状态。转换前会先调用 `clamp()`，避免越界值
                直接进入 mapper。

        Returns:
            参数名到数值的映射，可直接传给 `VTubeRuntime.set_parameters()`。
        """
        state = state.clamp()
        assert isinstance(self.profile, AvatarProfile)

        parameters: ParameterFrame = {}
        for rule in self.profile.enabled_mappings():
            value = _read_avatar_field(state, rule.avatar_field)
            if rule.clamp:
                value = _clamp(value, rule.canonical_min, rule.canonical_max)
            parameters[rule.backend_parameter] = _map_value(value, rule)
        return parameters

    def neutral_parameters(self) -> ParameterFrame:
        """返回中性 avatar 状态对应的 VTube Studio 参数。"""
        return self.to_parameters(AvatarState.neutral())

    def effective_fields(self) -> tuple[str, ...]:
        """返回当前 mapper 实际会读取的 `AvatarState` 字段。"""
        assert isinstance(self.profile, AvatarProfile)
        return self.profile.effective_fields()


def _read_avatar_field(state: AvatarState, field_path: str) -> float:
    """按 `face.yaw` 这样的路径读取 `AvatarState` 数值字段。"""
    current: Any = state
    for part in field_path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"AvatarState has no field path: {field_path}")
        current = getattr(current, part)

    if isinstance(current, bool) or not isinstance(current, int | float):
        raise ValueError(f"Avatar field {field_path!r} is not a numeric field.")
    return float(current)


def _map_value(value: float, rule: AvatarParameterMapping) -> float:
    """根据映射策略把 canonical 数值转换到 backend 数值。"""
    if rule.strategy == "linear":
        return _linear_scale(
            value=value,
            source_min=rule.canonical_min,
            source_max=rule.canonical_max,
            target_min=rule.backend_min,
            target_max=rule.backend_max,
        )
    raise ValueError(f"Unsupported mapping strategy: {rule.strategy}")


def _linear_scale(
    *,
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    """把数值从一个范围线性缩放到另一个范围。"""
    if source_min == source_max:
        raise ValueError("Cannot scale value from a zero-width source range.")
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """把数值裁剪到给定范围，兼容反向范围。"""
    low = min(minimum, maximum)
    high = max(minimum, maximum)
    return max(low, min(high, value))
