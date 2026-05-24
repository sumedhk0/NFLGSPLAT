"""Team-color split + referee stripe detection on synthetic crops.

No GPU / no real imagery — we fabricate solid-color and striped BGR arrays
with known properties and assert the CPU classifiers behave.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.identity.team_color import (
    dominant_jersey_color,
    is_referee,
    split_two_teams,
)


def _solid(bgr: tuple[int, int, int], h: int = 80, w: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def _vertical_stripes(h: int = 80, w: int = 60, period: int = 6) -> np.ndarray:
    """Grayscale black/white vertical stripes (referee shirt)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        if (x // period) % 2 == 0:
            img[:, x] = (255, 255, 255)
    return img


def test_dominant_color_is_grayscale_for_white_jersey():
    color = dominant_jersey_color(_solid((255, 255, 255)))
    # White → very low saturation, high value.
    assert color[1] < 10
    assert color[2] > 240


def test_split_two_teams_separates_red_and_blue():
    reds = [dominant_jersey_color(_solid((0, 0, 200))) for _ in range(4)]
    blues = [dominant_jersey_color(_solid((200, 0, 0))) for _ in range(3)]
    colors = np.stack(reds + blues)
    labels = split_two_teams(colors)
    # The 4 reds share one label, the 3 blues the other.
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[-1]


def test_split_two_teams_degenerate_single_sample():
    labels = split_two_teams(np.array([[10.0, 20.0, 30.0]]))
    assert labels.tolist() == [0]


def test_is_referee_true_for_stripes():
    assert is_referee(_vertical_stripes()) is True


def test_is_referee_false_for_solid_jersey():
    assert is_referee(_solid((0, 0, 200))) is False
    # A solid gray shirt is low-saturation but has no stripe transitions.
    assert is_referee(_solid((120, 120, 120))) is False
