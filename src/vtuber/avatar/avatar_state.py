from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def _sample_signed(
    rng: random.Random,
    *,
    center_scale: float = 1.0,
    extreme_prob: float = 0.12,
) -> float:
    """采样一个偏向中心、少量触及极值的 `-1.0 .. 1.0` 标量。"""
    if rng.random() < extreme_prob:
        return rng.uniform(-1.0, 1.0)
    limit = _clamp(center_scale, 0.05, 1.0)
    return rng.triangular(-limit, limit, 0.0)


def _sample_unit(
    rng: random.Random,
    *,
    mode: float,
    extreme_prob: float = 0.12,
) -> float:
    """采样一个偏向指定 mode、少量触及极值的 `0.0 .. 1.0` 标量。"""
    mode = _clamp(mode, 0.0, 1.0)
    if rng.random() < extreme_prob:
        return rng.uniform(0.0, 1.0)
    return rng.triangular(0.0, 1.0, mode)


@dataclass(frozen=True)
class FacePose:
    """头部姿态状态。

    取值范围均为 `-1.0 .. 1.0`，具体角度范围由运行时 mapper 决定。
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @classmethod
    def neutral(cls) -> FacePose:
        """返回头部居中的中性状态。"""
        return cls()

    def clamp(self) -> FacePose:
        """把头部姿态裁剪到标准范围。"""
        return FacePose(
            yaw=_clamp(self.yaw, -1.0, 1.0),
            pitch=_clamp(self.pitch, -1.0, 1.0),
            roll=_clamp(self.roll, -1.0, 1.0),
        )

    def lerp(self, target: FacePose, alpha: float) -> FacePose:
        """向目标头部姿态线性插值。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return FacePose(
            yaw=_lerp(self.yaw, target.yaw, alpha),
            pitch=_lerp(self.pitch, target.pitch, alpha),
            roll=_lerp(self.roll, target.roll, alpha),
        )


@dataclass(frozen=True)
class EyeState:
    """眼睛和视线状态。

    `left_open/right_open` 使用 `0.0 .. 1.0`，`gaze_x/gaze_y` 使用 `-1.0 .. 1.0`。
    """

    left_open: float = 1.0
    right_open: float = 1.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0

    @classmethod
    def neutral(cls) -> EyeState:
        """返回双眼睁开、视线居中的中性状态。"""
        return cls()

    def clamp(self) -> EyeState:
        """把眼睛和视线状态裁剪到标准范围。"""
        return EyeState(
            left_open=_clamp(self.left_open, 0.0, 1.0),
            right_open=_clamp(self.right_open, 0.0, 1.0),
            gaze_x=_clamp(self.gaze_x, -1.0, 1.0),
            gaze_y=_clamp(self.gaze_y, -1.0, 1.0),
        )

    def lerp(self, target: EyeState, alpha: float) -> EyeState:
        """向目标眼睛状态线性插值。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return EyeState(
            left_open=_lerp(self.left_open, target.left_open, alpha),
            right_open=_lerp(self.right_open, target.right_open, alpha),
            gaze_x=_lerp(self.gaze_x, target.gaze_x, alpha),
            gaze_y=_lerp(self.gaze_y, target.gaze_y, alpha),
        )


@dataclass(frozen=True)
class MouthState:
    """嘴部状态。

    `open` 使用 `0.0 .. 1.0`，`smile/form` 使用 `-1.0 .. 1.0`。
    """

    open: float = 0.0
    smile: float = 0.0
    form: float = 0.0

    @classmethod
    def neutral(cls) -> MouthState:
        """返回闭嘴、无笑容、嘴型居中的中性状态。"""
        return cls()

    def clamp(self) -> MouthState:
        """把嘴部状态裁剪到标准范围。"""
        return MouthState(
            open=_clamp(self.open, 0.0, 1.0),
            smile=_clamp(self.smile, -1.0, 1.0),
            form=_clamp(self.form, -1.0, 1.0),
        )

    def lerp(self, target: MouthState, alpha: float) -> MouthState:
        """向目标嘴部状态线性插值。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return MouthState(
            open=_lerp(self.open, target.open, alpha),
            smile=_lerp(self.smile, target.smile, alpha),
            form=_lerp(self.form, target.form, alpha),
        )


@dataclass(frozen=True)
class ExpressionState:
    """表情和眉毛状态。

    眉毛使用 `-1.0 .. 1.0`；表情强度使用 `0.0 .. 1.0`。这些字段是通用语义，
    具体能否生效取决于后端 mapper 和皮套参数映射。
    """

    brow_left_y: float = 0.0
    brow_right_y: float = 0.0
    happy: float = 0.0
    angry: float = 0.0
    sad: float = 0.0
    surprised: float = 0.0

    @classmethod
    def neutral(cls) -> ExpressionState:
        """返回无额外表情的中性状态。"""
        return cls()

    def clamp(self) -> ExpressionState:
        """把表情状态裁剪到标准范围。"""
        return ExpressionState(
            brow_left_y=_clamp(self.brow_left_y, -1.0, 1.0),
            brow_right_y=_clamp(self.brow_right_y, -1.0, 1.0),
            happy=_clamp(self.happy, 0.0, 1.0),
            angry=_clamp(self.angry, 0.0, 1.0),
            sad=_clamp(self.sad, 0.0, 1.0),
            surprised=_clamp(self.surprised, 0.0, 1.0),
        )

    def lerp(self, target: ExpressionState, alpha: float) -> ExpressionState:
        """向目标表情状态线性插值。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return ExpressionState(
            brow_left_y=_lerp(self.brow_left_y, target.brow_left_y, alpha),
            brow_right_y=_lerp(self.brow_right_y, target.brow_right_y, alpha),
            happy=_lerp(self.happy, target.happy, alpha),
            angry=_lerp(self.angry, target.angry, alpha),
            sad=_lerp(self.sad, target.sad, alpha),
            surprised=_lerp(self.surprised, target.surprised, alpha),
        )


@dataclass(frozen=True)
class BodyState:
    """身体运动状态。

    `sway_x/sway_y` 使用 `-1.0 .. 1.0`，`breath` 使用 `0.0 .. 1.0`。
    """

    sway_x: float = 0.0
    sway_y: float = 0.0
    breath: float = 0.0

    @classmethod
    def neutral(cls) -> BodyState:
        """返回身体居中、无呼吸增强的中性状态。"""
        return cls()

    def clamp(self) -> BodyState:
        """把身体状态裁剪到标准范围。"""
        return BodyState(
            sway_x=_clamp(self.sway_x, -1.0, 1.0),
            sway_y=_clamp(self.sway_y, -1.0, 1.0),
            breath=_clamp(self.breath, 0.0, 1.0),
        )

    def lerp(self, target: BodyState, alpha: float) -> BodyState:
        """向目标身体状态线性插值。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return BodyState(
            sway_x=_lerp(self.sway_x, target.sway_x, alpha),
            sway_y=_lerp(self.sway_y, target.sway_y, alpha),
            breath=_lerp(self.breath, target.breath, alpha),
        )


@dataclass(frozen=True)
class SpeechState:
    """说话状态。

    `active` 表示当前是否在说话；`text` 是可选文本计划；`volume/prosody` 使用
    `0.0 .. 1.0`，用于后续 TTS、嘴型或声音风格控制。
    """

    active: bool = False
    text: str | None = None
    volume: float = 0.0
    prosody: float = 0.0

    @classmethod
    def neutral(cls) -> SpeechState:
        """返回不说话的中性状态。"""
        return cls()

    def clamp(self) -> SpeechState:
        """把说话控制状态裁剪到标准范围。"""
        return SpeechState(
            active=self.active,
            text=self.text,
            volume=_clamp(self.volume, 0.0, 1.0),
            prosody=_clamp(self.prosody, 0.0, 1.0),
        )

    def lerp(self, target: SpeechState, alpha: float) -> SpeechState:
        """向目标说话状态插值。

        布尔值和文本不是连续量，因此在 `alpha >= 0.5` 时切换到目标状态。
        """
        alpha = _clamp(alpha, 0.0, 1.0)
        use_target_discrete = alpha >= 0.5
        return SpeechState(
            active=target.active if use_target_discrete else self.active,
            text=target.text if use_target_discrete else self.text,
            volume=_lerp(self.volume, target.volume, alpha),
            prosody=_lerp(self.prosody, target.prosody, alpha),
        )


@dataclass(frozen=True)
class AvatarState:
    """项目内部通用的 avatar 参数空间。

    这是模型输出与各类运行时之间的稳定中间层。它不包含 VTube Studio、Live2D
    Cubism 或具体皮套参数名；各后端通过 mapper 把它转换成自己的参数空间。
    """

    face: FacePose = field(default_factory=FacePose.neutral)
    eyes: EyeState = field(default_factory=EyeState.neutral)
    mouth: MouthState = field(default_factory=MouthState.neutral)
    expression: ExpressionState = field(default_factory=ExpressionState.neutral)
    body: BodyState = field(default_factory=BodyState.neutral)
    speech: SpeechState = field(default_factory=SpeechState.neutral)

    @classmethod
    def neutral(cls) -> AvatarState:
        """返回完整 avatar 的中性状态。"""
        return cls()

    def clamp(self) -> AvatarState:
        """把所有子状态裁剪到各自约定范围内，并返回新的状态。"""
        return AvatarState(
            face=self.face.clamp(),
            eyes=self.eyes.clamp(),
            mouth=self.mouth.clamp(),
            expression=self.expression.clamp(),
            body=self.body.clamp(),
            speech=self.speech.clamp(),
        )

    def lerp(self, target: AvatarState, alpha: float) -> AvatarState:
        """向目标 avatar 状态做线性插值，并返回新的状态。"""
        alpha = _clamp(alpha, 0.0, 1.0)
        return AvatarState(
            face=self.face.lerp(target.face, alpha),
            eyes=self.eyes.lerp(target.eyes, alpha),
            mouth=self.mouth.lerp(target.mouth, alpha),
            expression=self.expression.lerp(target.expression, alpha),
            body=self.body.lerp(target.body, alpha),
            speech=self.speech.lerp(target.speech, alpha),
        )

    @staticmethod
    def sample_random(rng: random.Random | None = None) -> AvatarState:
        """采样一个静态随机姿态。

        该采样器面向数据集构造，分布上偏向中性区域，并保留少量极端样本。
        """
        rng = rng or random.Random()

        is_blink = rng.random() < 0.12
        if is_blink:
            left_open = _sample_unit(rng, mode=0.08, extreme_prob=0.0)
            right_open = _sample_unit(rng, mode=0.08, extreme_prob=0.0)
        else:
            left_open = _sample_unit(rng, mode=0.92, extreme_prob=0.08)
            right_open = _sample_unit(rng, mode=0.92, extreme_prob=0.08)

        return AvatarState(
            face=FacePose(
                yaw=_sample_signed(rng, center_scale=0.55, extreme_prob=0.15),
                pitch=_sample_signed(rng, center_scale=0.35, extreme_prob=0.10),
                roll=_sample_signed(rng, center_scale=0.25, extreme_prob=0.08),
            ),
            eyes=EyeState(
                left_open=left_open,
                right_open=right_open,
                gaze_x=_sample_signed(rng, center_scale=0.45, extreme_prob=0.10),
                gaze_y=_sample_signed(rng, center_scale=0.30, extreme_prob=0.08),
            ),
            mouth=MouthState(
                open=_sample_unit(rng, mode=0.18, extreme_prob=0.15),
                smile=_sample_signed(rng, center_scale=0.25, extreme_prob=0.08),
                form=_sample_signed(rng, center_scale=0.25, extreme_prob=0.08),
            ),
            expression=ExpressionState(
                brow_left_y=_sample_signed(rng, center_scale=0.22, extreme_prob=0.06),
                brow_right_y=_sample_signed(rng, center_scale=0.22, extreme_prob=0.06),
                happy=_sample_unit(rng, mode=0.10, extreme_prob=0.05),
                angry=_sample_unit(rng, mode=0.05, extreme_prob=0.05),
                sad=_sample_unit(rng, mode=0.03, extreme_prob=0.03),
                surprised=_sample_unit(rng, mode=0.04, extreme_prob=0.04),
            ),
            body=BodyState.neutral(),
            speech=SpeechState.neutral(),
        ).clamp()

    def to_dict(self) -> dict[str, object]:
        """转换为适合写入 JSON 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AvatarState:
        """从字典恢复 `AvatarState`。"""
        face = dict(data.get("face", {}))
        eyes = dict(data.get("eyes", {}))
        mouth = dict(data.get("mouth", {}))
        expression = dict(data.get("expression", {}))
        body = dict(data.get("body", {}))
        speech = dict(data.get("speech", {}))
        return cls(
            face=FacePose(
                yaw=float(face.get("yaw", 0.0)),
                pitch=float(face.get("pitch", 0.0)),
                roll=float(face.get("roll", 0.0)),
            ),
            eyes=EyeState(
                left_open=float(eyes.get("left_open", 1.0)),
                right_open=float(eyes.get("right_open", 1.0)),
                gaze_x=float(eyes.get("gaze_x", 0.0)),
                gaze_y=float(eyes.get("gaze_y", 0.0)),
            ),
            mouth=MouthState(
                open=float(mouth.get("open", 0.0)),
                smile=float(mouth.get("smile", 0.0)),
                form=float(mouth.get("form", 0.0)),
            ),
            expression=ExpressionState(
                brow_left_y=float(expression.get("brow_left_y", 0.0)),
                brow_right_y=float(expression.get("brow_right_y", 0.0)),
                happy=float(expression.get("happy", 0.0)),
                angry=float(expression.get("angry", 0.0)),
                sad=float(expression.get("sad", 0.0)),
                surprised=float(expression.get("surprised", 0.0)),
            ),
            body=BodyState(
                sway_x=float(body.get("sway_x", 0.0)),
                sway_y=float(body.get("sway_y", 0.0)),
                breath=float(body.get("breath", 0.0)),
            ),
            speech=SpeechState(
                active=bool(speech.get("active", False)),
                text=speech.get("text"),
                volume=float(speech.get("volume", 0.0)),
                prosody=float(speech.get("prosody", 0.0)),
            ),
        ).clamp()
