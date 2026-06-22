from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines


def _synthetic_field(W=1280, H=720):
    img = np.full((H, W, 3), (40, 120, 40), np.uint8)        # green field
    for x in (300, 600, 900):                                 # white vertical yard lines
        cv2.line(img, (x, 60), (x, H - 60), (240, 240, 240), 4)
    return img


def test_detect_lines_finds_vertical_yard_lines():
    img = _synthetic_field()
    feats = detect_lines(img, FieldDetectConfig())
    xs = sorted(round(0.5 * (s.p0[0] + s.p1[0])) for s in feats)
    assert len(feats) >= 3
    assert any(abs(x - 300) < 25 for x in xs)
    assert any(abs(x - 600) < 25 for x in xs)
    assert any(abs(x - 900) < 25 for x in xs)
