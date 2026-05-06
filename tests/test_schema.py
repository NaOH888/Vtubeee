from __future__ import annotations

import unittest

from vtuber_ai.streams.schema import FrameFeature, MultimodalWindow, validate_monotonic_timestamps


def test_window_computes_end_time() -> None:
    window = MultimodalWindow(
        start_time=10.0,
        step_seconds=0.5,
        video=[[0.0], [1.0]],
        audio=[[0.0], [1.0]],
        chat=[[0.0], [1.0]],
        target_time=11.0,
    )

    assert window.seq_len == 2
    assert window.end_time == 11.0


def test_validate_monotonic_timestamps_rejects_reordered_items() -> None:
    items = [
        FrameFeature(timestamp=2.0, vector=[0.0]),
        FrameFeature(timestamp=1.0, vector=[0.0]),
    ]

    with unittest.TestCase().assertRaisesRegex(ValueError, "monotonic"):
        validate_monotonic_timestamps(items)
