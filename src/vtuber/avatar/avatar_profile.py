from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VTUBE_INPUT_TO_CANONICAL: dict[str, tuple[str, float, float]] = {
    "FaceAngleX": ("face.yaw", -1.0, 1.0),
    "FaceAngleY": ("face.pitch", -1.0, 1.0),
    "FaceAngleZ": ("face.roll", -1.0, 1.0),
    "MouthOpen": ("mouth.open", 0.0, 1.0),
    "MouthSmile": ("mouth.smile", 0.0, 1.0),
    "MouthX": ("mouth.form", -1.0, 1.0),
    "EyeOpenLeft": ("eyes.left_open", 0.0, 1.0),
    "EyeOpenRight": ("eyes.right_open", 0.0, 1.0),
    "EyeLeftX": ("eyes.gaze_x", -1.0, 1.0),
    "EyeLeftY": ("eyes.gaze_y", -1.0, 1.0),
    "EyeRightX": ("eyes.gaze_x", -1.0, 1.0),
    "EyeRightY": ("eyes.gaze_y", -1.0, 1.0),
    "BrowLeftY": ("expression.brow_left_y", 0.0, 1.0),
    "BrowRightY": ("expression.brow_right_y", 0.0, 1.0),
    "FaceAngry": ("expression.angry", 0.0, 1.0),
    "VoiceVolume": ("speech.volume", 0.0, 1.0),
    "VoiceFrequency": ("speech.prosody", 0.0, 1.0),
}


@dataclass(frozen=True)
class AvatarParameterMapping:
    """一条 canonical avatar 字段到后端参数的映射规则。

    canonical 字段指 `AvatarState` 内部字段，例如 `face.yaw`；backend 参数指具体
    后端能接收的参数，例如 VTube Studio 的 `FaceAngleX`。mapper 会读取 canonical
    字段，按这里声明的范围缩放到 backend 参数范围。
    """

    avatar_field: str
    backend_parameter: str
    canonical_min: float
    canonical_max: float
    backend_min: float
    backend_max: float
    strategy: str = "linear"
    enabled: bool = True
    clamp: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvatarParameterMapping:
        """从标准 profile JSON 字典创建映射规则。"""
        avatar_field = data.get("avatar_field")
        backend_parameter = data.get("backend_parameter")
        strategy = data.get("strategy", "linear")
        if not avatar_field:
            raise ValueError("mapping requires avatar_field.")
        if not backend_parameter:
            raise ValueError("mapping requires backend_parameter.")

        return cls(
            avatar_field=str(avatar_field),
            backend_parameter=str(backend_parameter),
            canonical_min=float(data["canonical_min"]),
            canonical_max=float(data["canonical_max"]),
            backend_min=float(data["backend_min"]),
            backend_max=float(data["backend_max"]),
            strategy=str(strategy),
            enabled=bool(data.get("enabled", True)),
            clamp=bool(data.get("clamp", True)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""
        return {
            "avatar_field": self.avatar_field,
            "backend_parameter": self.backend_parameter,
            "canonical_min": self.canonical_min,
            "canonical_max": self.canonical_max,
            "backend_min": self.backend_min,
            "backend_max": self.backend_max,
            "strategy": self.strategy,
            "enabled": self.enabled,
            "clamp": self.clamp,
            "metadata": self.metadata,
        }

    def validate(self) -> None:
        """校验映射规则是否可执行。"""
        if not self.avatar_field:
            raise ValueError("avatar_field cannot be empty.")
        if not self.backend_parameter:
            raise ValueError("backend_parameter cannot be empty.")
        if self.canonical_min == self.canonical_max:
            raise ValueError(f"{self.avatar_field} has zero canonical range.")
        if self.strategy != "linear":
            raise ValueError(f"Unsupported mapping strategy: {self.strategy}")


@dataclass(frozen=True)
class AvatarProfile:
    """单个模型或后端配置的能力描述。

    profile 不替代训练 label。训练 label 仍然是统一的 `AvatarState`；profile 只说明
    当前模型哪些 `AvatarState` 字段真正能映射到后端参数，以及范围如何换算。
    """

    profile_id: str
    backend: str
    mappings: tuple[AvatarParameterMapping, ...]
    source_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file: Path | str) -> AvatarProfile:
        """从 profile JSON 或 VTube Studio `.vtube.json` 文件读取 profile。"""
        path = Path(file)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if "mappings" in data:
            return cls.from_dict(data, source_file=path)
        if "ParameterSettings" in data:
            return cls.from_vtube_studio_dict(data, source_file=path)
        raise ValueError(f"Unsupported avatar profile file: {path}")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_file: Path | None = None,
    ) -> AvatarProfile:
        """从标准 profile JSON 字典创建 `AvatarProfile`。"""
        if "profile_id" not in data:
            raise ValueError("profile requires profile_id.")
        if "backend" not in data:
            raise ValueError("profile requires backend.")
        if "mappings" not in data:
            raise ValueError("profile requires mappings.")

        mappings = tuple(
            AvatarParameterMapping.from_dict(item)
            for item in data["mappings"]
        )
        profile = cls(
            profile_id=str(data["profile_id"]),
            backend=str(data["backend"]),
            mappings=mappings,
            source_file=source_file,
            metadata=dict(data.get("metadata", {})),
        )
        profile.validate()
        return profile

    @classmethod
    def from_vtube_studio_dict(
        cls,
        data: dict[str, Any],
        *,
        source_file: Path | None = None,
        profile_id: str | None = None,
    ) -> AvatarProfile:
        """从 VTube Studio 的 `.vtube.json` 配置推导 profile。

        `.vtube.json` 记录的是 VTube Studio 输入参数到 Live2D 参数的映射。这里抽取
        唯一的 VTube Studio 输入参数，并把已知输入名对齐到 `AvatarState` 字段。
        """
        mappings_by_backend: dict[str, AvatarParameterMapping] = {}
        live2d_outputs: dict[str, list[dict[str, Any]]] = {}

        for setting in data.get("ParameterSettings", []):
            backend_parameter = str(setting.get("Input", ""))
            if not backend_parameter:
                continue

            live2d_outputs.setdefault(backend_parameter, []).append(
                {
                    "live2d_parameter": setting.get("OutputLive2D"),
                    "output_min": setting.get("OutputRangeLower"),
                    "output_max": setting.get("OutputRangeUpper"),
                    "smoothing": setting.get("Smoothing"),
                    "name": setting.get("Name"),
                }
            )

            if backend_parameter in mappings_by_backend:
                continue
            if backend_parameter not in VTUBE_INPUT_TO_CANONICAL:
                continue

            avatar_field, canonical_min, canonical_max = VTUBE_INPUT_TO_CANONICAL[backend_parameter]
            mappings_by_backend[backend_parameter] = AvatarParameterMapping(
                avatar_field=avatar_field,
                backend_parameter=backend_parameter,
                canonical_min=canonical_min,
                canonical_max=canonical_max,
                backend_min=float(setting.get("InputRangeLower", canonical_min)),
                backend_max=float(setting.get("InputRangeUpper", canonical_max)),
                metadata={
                    "source": "vtube_studio_parameter_settings",
                    "first_live2d_parameter": setting.get("OutputLive2D"),
                },
            )

        model_id = data.get("ModelID") or data.get("Name") or "vtube_studio_profile"
        profile = cls(
            profile_id=str(profile_id or model_id),
            backend="vtube_studio",
            mappings=tuple(mappings_by_backend.values()),
            source_file=source_file,
            metadata={
                "source_format": "vtube_studio",
                "model_name": data.get("Name"),
                "model_id": data.get("ModelID"),
                "live2d_outputs": live2d_outputs,
            },
        )
        profile.validate()
        return profile

    @classmethod
    def coerce(cls, value: AvatarProfile | Path | str | dict[str, Any]) -> AvatarProfile:
        """把常见 profile 输入统一转换为 `AvatarProfile`。"""
        if isinstance(value, AvatarProfile):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        if isinstance(value, Path | str):
            return cls.from_file(value)
        raise TypeError(f"Unsupported avatar profile value: {type(value)!r}")

    def validate(self) -> None:
        """校验 profile 是否可用于 mapper。"""
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty.")
        if not self.backend:
            raise ValueError("backend cannot be empty.")
        if not self.mappings:
            raise ValueError("profile must contain at least one mapping.")

        seen_backend_parameters: set[str] = set()
        for mapping in self.mappings:
            mapping.validate()
            if mapping.enabled and mapping.backend_parameter in seen_backend_parameters:
                raise ValueError(f"Duplicate backend parameter: {mapping.backend_parameter}")
            if mapping.enabled:
                seen_backend_parameters.add(mapping.backend_parameter)

    def check(self) -> bool:
        """返回 profile 是否通过校验。"""
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def get_source_file(self) -> Path | None:
        """返回 profile 来源文件；默认 profile 或手工构造 profile 可能没有来源文件。"""
        return self.source_file

    def enabled_mappings(self) -> tuple[AvatarParameterMapping, ...]:
        """返回当前启用的映射规则。"""
        return tuple(mapping for mapping in self.mappings if mapping.enabled)

    def effective_fields(self) -> tuple[str, ...]:
        """返回当前模型实际会被映射的 canonical 字段。"""
        return tuple(dict.fromkeys(mapping.avatar_field for mapping in self.enabled_mappings()))

    def backend_parameters(self) -> tuple[str, ...]:
        """返回当前 profile 会输出的后端参数名。"""
        return tuple(mapping.backend_parameter for mapping in self.enabled_mappings())

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的标准 profile 字典。"""
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "metadata": self.metadata,
        }

    def save(self, file: Path | str) -> None:
        """把当前 profile 写入 JSON 文件。"""
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    def __getitem__(self, key: str) -> Any:
        """按字典方式读取 profile 顶层字段，方便临时调试。"""
        return self.to_dict()[key]
