from __future__ import annotations

from vtuber.avatar import (
    AvatarState,
    BodyState,
    ExpressionState,
    EyeState,
    FacePose,
    MouthState,
    SpeechState,
)


def test_neutral_state_defaults() -> None:
    assert AvatarState.neutral() == AvatarState(
        face=FacePose.neutral(),
        eyes=EyeState.neutral(),
        mouth=MouthState.neutral(),
        expression=ExpressionState.neutral(),
        body=BodyState.neutral(),
        speech=SpeechState.neutral(),
    )


def test_clamp_limits_nested_fields() -> None:
    state = AvatarState(
        face=FacePose(yaw=2.0, pitch=-2.0, roll=3.0),
        eyes=EyeState(left_open=-0.5, right_open=1.5, gaze_x=2.0, gaze_y=-2.0),
        mouth=MouthState(open=-1.0, smile=2.0, form=-2.0),
        expression=ExpressionState(
            brow_left_y=2.0,
            brow_right_y=-2.0,
            happy=2.0,
            angry=-1.0,
            sad=2.0,
            surprised=2.0,
        ),
        body=BodyState(sway_x=2.0, sway_y=-2.0, breath=2.0),
        speech=SpeechState(active=True, text="hello", volume=2.0, prosody=-1.0),
    )

    assert state.clamp() == AvatarState(
        face=FacePose(yaw=1.0, pitch=-1.0, roll=1.0),
        eyes=EyeState(left_open=0.0, right_open=1.0, gaze_x=1.0, gaze_y=-1.0),
        mouth=MouthState(open=0.0, smile=1.0, form=-1.0),
        expression=ExpressionState(
            brow_left_y=1.0,
            brow_right_y=-1.0,
            happy=1.0,
            angry=0.0,
            sad=1.0,
            surprised=1.0,
        ),
        body=BodyState(sway_x=1.0, sway_y=-1.0, breath=1.0),
        speech=SpeechState(active=True, text="hello", volume=1.0, prosody=0.0),
    )


def test_lerp_interpolates_nested_fields_and_clamps_alpha() -> None:
    start = AvatarState.neutral()
    end = AvatarState(
        face=FacePose(yaw=1.0, pitch=-1.0, roll=0.5),
        eyes=EyeState(left_open=0.0, right_open=0.5, gaze_x=1.0, gaze_y=-1.0),
        mouth=MouthState(open=1.0, smile=-1.0, form=0.5),
        expression=ExpressionState(brow_left_y=1.0, brow_right_y=-1.0, happy=1.0),
        body=BodyState(sway_x=1.0, sway_y=-1.0, breath=1.0),
        speech=SpeechState(active=True, text="target", volume=1.0, prosody=0.5),
    )

    assert start.lerp(end, 0.5) == AvatarState(
        face=FacePose(yaw=0.5, pitch=-0.5, roll=0.25),
        eyes=EyeState(left_open=0.5, right_open=0.75, gaze_x=0.5, gaze_y=-0.5),
        mouth=MouthState(open=0.5, smile=-0.5, form=0.25),
        expression=ExpressionState(brow_left_y=0.5, brow_right_y=-0.5, happy=0.5),
        body=BodyState(sway_x=0.5, sway_y=-0.5, breath=0.5),
        speech=SpeechState(active=True, text="target", volume=0.5, prosody=0.25),
    )
    assert start.lerp(end, -1.0) == start
    assert start.lerp(end, 2.0) == end
