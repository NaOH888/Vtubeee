from __future__ import annotations

from vtuber.avatar import (
    AvatarParameterMapping,
    AvatarProfile,
    AvatarState,
    EyeState,
    FacePose,
    MouthState,
)
from vtuber.vtube_studio import VTubeParameterMapper


def profile_with_vtube_parameters(
    *,
    head_yaw_degrees: float = 30.0,
    head_pitch_degrees: float = 30.0,
    head_roll_degrees: float = 30.0,
) -> AvatarProfile:
    """返回测试用 VTube Studio profile。"""
    return AvatarProfile(
        profile_id="test_vtube_profile",
        backend="vtube_studio",
        mappings=(
            AvatarParameterMapping(
                "face.yaw", "FaceAngleX", -1.0, 1.0, -head_yaw_degrees, head_yaw_degrees
            ),
            AvatarParameterMapping(
                "face.pitch", "FaceAngleY", -1.0, 1.0, -head_pitch_degrees, head_pitch_degrees
            ),
            AvatarParameterMapping(
                "face.roll", "FaceAngleZ", -1.0, 1.0, -head_roll_degrees, head_roll_degrees
            ),
            AvatarParameterMapping("mouth.open", "MouthOpen", 0.0, 1.0, 0.0, 1.0),
            AvatarParameterMapping("mouth.smile", "MouthSmile", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("mouth.form", "MouthX", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("eyes.left_open", "EyeOpenLeft", 0.0, 1.0, 0.0, 1.0),
            AvatarParameterMapping("eyes.right_open", "EyeOpenRight", 0.0, 1.0, 0.0, 1.0),
            AvatarParameterMapping("eyes.gaze_x", "EyeLeftX", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("eyes.gaze_y", "EyeLeftY", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("eyes.gaze_x", "EyeRightX", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("eyes.gaze_y", "EyeRightY", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping("expression.brow_left_y", "Brows", -1.0, 1.0, -1.0, 1.0),
            AvatarParameterMapping(
                "expression.brow_left_y", "BrowLeftY", -1.0, 1.0, -1.0, 1.0
            ),
            AvatarParameterMapping(
                "expression.brow_right_y", "BrowRightY", -1.0, 1.0, -1.0, 1.0
            ),
            AvatarParameterMapping("expression.angry", "FaceAngry", 0.0, 1.0, 0.0, 1.0),
            AvatarParameterMapping("speech.volume", "VoiceVolume", 0.0, 1.0, 0.0, 1.0),
            AvatarParameterMapping("speech.prosody", "VoiceFrequency", 0.0, 1.0, 0.0, 1.0),
        ),
    )


def test_mapper_converts_avatar_state_to_profile_vtube_parameters() -> None:
    mapper = VTubeParameterMapper(profile=profile_with_vtube_parameters())
    state = AvatarState(
        face=FacePose(yaw=0.5, pitch=-0.25, roll=1.0),
        mouth=MouthState(open=0.75, smile=-0.5),
        eyes=EyeState(left_open=0.25, right_open=0.5),
    )

    params = mapper.to_parameters(state)
    assert params == {
        "FaceAngleX": 15.0,
        "FaceAngleY": -7.5,
        "FaceAngleZ": 30.0,
        "MouthOpen": 0.75,
        "MouthSmile": -0.5,
        "MouthX": 0.0,
        "EyeOpenLeft": 0.25,
        "EyeOpenRight": 0.5,
        "EyeLeftX": 0.0,
        "EyeLeftY": 0.0,
        "EyeRightX": 0.0,
        "EyeRightY": 0.0,
        "Brows": 0.0,
        "BrowLeftY": 0.0,
        "BrowRightY": 0.0,
        "FaceAngry": 0.0,
        "VoiceVolume": 0.0,
        "VoiceFrequency": 0.0,
    }


def test_mapper_clamps_state_before_conversion() -> None:
    mapper = VTubeParameterMapper(
        profile=profile_with_vtube_parameters(
            head_yaw_degrees=20.0,
            head_pitch_degrees=10.0,
            head_roll_degrees=5.0,
        )
    )
    params = mapper.to_parameters(
        AvatarState(
            face=FacePose(yaw=2.0, pitch=-2.0, roll=2.0),
            mouth=MouthState(open=2.0, smile=-2.0),
            eyes=EyeState(left_open=-1.0, right_open=2.0),
        )
    )

    assert params["FaceAngleX"] == 20.0
    assert params["FaceAngleY"] == -10.0
    assert params["FaceAngleZ"] == 5.0
    assert params["MouthOpen"] == 1.0
    assert params["MouthSmile"] == -1.0
    assert params["EyeOpenLeft"] == 0.0
    assert params["EyeOpenRight"] == 1.0
    assert params == {
        "FaceAngleX": 20.0,
        "FaceAngleY": -10.0,
        "FaceAngleZ": 5.0,
        "MouthOpen": 1.0,
        "MouthSmile": -1.0,
        "MouthX": 0.0,
        "EyeOpenLeft": 0.0,
        "EyeOpenRight": 1.0,
        "EyeLeftX": 0.0,
        "EyeLeftY": 0.0,
        "EyeRightX": 0.0,
        "EyeRightY": 0.0,
        "Brows": 0.0,
        "BrowLeftY": 0.0,
        "BrowRightY": 0.0,
        "FaceAngry": 0.0,
        "VoiceVolume": 0.0,
        "VoiceFrequency": 0.0,
    }


def test_neutral_parameters_match_neutral_state() -> None:
    mapper = VTubeParameterMapper(profile=profile_with_vtube_parameters())

    assert mapper.neutral_parameters() == mapper.to_parameters(AvatarState.neutral())


def test_mapper_uses_profile_mappings_only() -> None:
    profile = AvatarProfile(
        profile_id="minimal_test",
        backend="vtube_studio",
        mappings=(
            AvatarParameterMapping(
                avatar_field="face.yaw",
                backend_parameter="HeadYaw",
                canonical_min=-1.0,
                canonical_max=1.0,
                backend_min=-45.0,
                backend_max=45.0,
            ),
        ),
    )
    mapper = VTubeParameterMapper(profile=profile)

    params = mapper.to_parameters(
        AvatarState(
            face=FacePose(yaw=0.5, pitch=1.0),
            mouth=MouthState(open=1.0),
        )
    )

    assert params == {"HeadYaw": 22.5}
    assert mapper.effective_fields() == ("face.yaw",)


def test_profile_can_be_derived_from_vtube_studio_parameter_settings() -> None:
    profile = AvatarProfile.from_vtube_studio_dict(
        {
            "ModelID": "hiyori_test",
            "Name": "hiyori",
            "ParameterSettings": [
                {
                    "Input": "FaceAngleY",
                    "InputRangeLower": -20.0,
                    "InputRangeUpper": 20.0,
                    "OutputLive2D": "ParamAngleY",
                    "OutputRangeLower": -30.0,
                    "OutputRangeUpper": 30.0,
                    "Smoothing": 15,
                    "Name": "Face Up/Down Rotation",
                },
                {
                    "Input": "MouthOpen",
                    "InputRangeLower": 0.0,
                    "InputRangeUpper": 1.0,
                    "OutputLive2D": "ParamMouthOpenY",
                    "OutputRangeLower": 0.0,
                    "OutputRangeUpper": 2.3,
                    "Smoothing": 0,
                    "Name": "Mouth Open",
                },
            ],
        }
    )
    mapper = VTubeParameterMapper(profile=profile)

    params = mapper.to_parameters(
        AvatarState(
            face=FacePose(pitch=0.5),
            mouth=MouthState(open=0.75),
        )
    )

    assert profile.profile_id == "hiyori_test"
    assert profile.effective_fields() == ("face.pitch", "mouth.open")
    assert params == {"FaceAngleY": 10.0, "MouthOpen": 0.75}
