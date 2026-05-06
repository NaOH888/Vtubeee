from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vtuber_ai.config import ModelConfig, ReactionConfig, StreamConfig

if torch is not None:
    from vtuber_ai.models import FusionReactionModel


@unittest.skipIf(torch is None, "torch is not installed")
def test_fusion_model_output_shapes() -> None:
    stream = StreamConfig(
        seq_len=4,
        video_feature_dim=8,
        audio_feature_dim=6,
        chat_feature_dim=5,
    )
    reaction = ReactionConfig(expression_dim=3, prosody_dim=2, speech_vocab_size=11)
    model_config = ModelConfig(hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    model = FusionReactionModel(stream, reaction, model_config)

    outputs = model(
        video=torch.randn(2, stream.seq_len, stream.video_feature_dim),
        audio=torch.randn(2, stream.seq_len, stream.audio_feature_dim),
        chat=torch.randn(2, stream.seq_len, stream.chat_feature_dim),
    )

    assert outputs["expression"].shape == (2, reaction.expression_dim)
    assert outputs["speech_active_logits"].shape == (2, 1)
    assert outputs["speech_token_logits"].shape == (2, reaction.speech_vocab_size)
    assert outputs["prosody"].shape == (2, reaction.prosody_dim)
