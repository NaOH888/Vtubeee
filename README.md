# VTuber AI

Research scaffold for an AI VTuber reaction model.

The main loop is:

1. Read continuous video features.
2. Read continuous audio features.
3. Read discrete chat events and bin them onto the same timeline.
4. Fuse the modalities over a recent context window.
5. Predict the next VTuber reaction:
   - expression parameters
   - speech activity
   - speech text token or text plan
   - prosody or voice-control parameters

This repository starts with a synthetic dataset and a compact PyTorch fusion model
so the training, inference, and shape contracts can be validated before connecting
real encoders, ASR, TTS, or avatar runtime code.

## Repository Layout

```text
src/vtuber_ai/
  config.py                 Project, stream, model, and training config.
  reactions.py              Reaction output dataclasses.
  streams/schema.py         Timestamped stream and chat event schemas.
  data/synthetic.py         Deterministic synthetic multimodal dataset.
  models/fusion.py          Transformer-based multimodal fusion model.
  training/train_synthetic.py
                            Minimal training entrypoint.
  inference/runtime.py      Small runtime predictor wrapper.
tests/
  test_schema.py
  test_model_shapes.py
docs/
  architecture.md
```

## Quick Start

Install the package in editable mode after installing PyTorch for your platform:

```powershell
python -m pip install -e ".[dev]"
```

Run the synthetic training smoke test:

```powershell
python -m vtuber_ai.training.train_synthetic --steps 20
```

Run tests:

```powershell
python -m pytest
```

## Design Notes

The model does not consume raw pixels or waveforms yet. It expects feature vectors:

- `video`: one vector per time step from a visual encoder.
- `audio`: one vector per time step from an audio encoder.
- `chat`: one vector per time step after discrete chat events are embedded and
  aggregated into timeline bins.

The first real-data milestone should replace `SyntheticMultimodalDataset` with a
dataset that yields the same tensors:

```text
video: [batch, time, video_feature_dim]
audio: [batch, time, audio_feature_dim]
chat:  [batch, time, chat_feature_dim]
```

The target remains the next-step reaction, not the current frame. This keeps the
training objective aligned with live response generation.
