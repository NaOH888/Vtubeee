from __future__ import annotations

import random
from dataclasses import asdict, dataclass


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class AvatarTransform:
    """avatar 在画面中的整体变换。

    这层不属于表情/姿态参数空间，而是渲染层控制：

    - `position_x` / `position_y`：模型整体位置
    - `rotation`：整体旋转角度
    - `size`：整体缩放，VTube Studio 范围为 `-100 .. 100`
    """

    position_x: float = 0.0
    position_y: float = 0.0
    rotation: float = 0.0
    size: float = 0.0

    @classmethod
    def neutral(cls) -> AvatarTransform:
        """返回画面中心、无旋转、默认大小的中性变换。"""
        return cls()

    def clamp(self) -> AvatarTransform:
        """把整体变换裁剪到 VTube Studio 官方允许范围。"""
        return AvatarTransform(
            position_x=_clamp(self.position_x, -1000.0, 1000.0),
            position_y=_clamp(self.position_y, -1000.0, 1000.0),
            rotation=_clamp(self.rotation, -360.0, 360.0),
            size=_clamp(self.size, -100.0, 100.0),
        )

    @staticmethod
    def sample_random(
        rng: random.Random | None = None,
        *,
        x_min: float = -0.7,
        x_max: float = 0.7,
        y_min: float = -0.9,
        y_max: float = 0.1,
        rotation_min: float = -10.0,
        rotation_max: float = 10.0,
        size_min: float = -18.0,
        size_max: float = 18.0,
    ) -> AvatarTransform:
        """采样一个随机整体变换。"""
        rng = rng or random.Random()
        return AvatarTransform(
            position_x=rng.uniform(x_min, x_max),
            position_y=rng.uniform(y_min, y_max),
            rotation=rng.uniform(rotation_min, rotation_max),
            size=rng.uniform(size_min, size_max),
        ).clamp()

    def to_dict(self) -> dict[str, float]:
        """转换为适合写入 JSON 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AvatarTransform:
        """从字典恢复 `AvatarTransform`。"""
        return cls(
            position_x=float(data.get("position_x", 0.0)),
            position_y=float(data.get("position_y", 0.0)),
            rotation=float(data.get("rotation", 0.0)),
            size=float(data.get("size", 0.0)),
        ).clamp()
