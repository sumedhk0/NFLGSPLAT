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


def test_detect_hashes_groups_two_rows():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_hashes
    img = np.full((720, 1280, 3), (40, 120, 40), np.uint8)
    for x in range(200, 1100, 90):                 # upper hash row at y=300
        cv2.rectangle(img, (x, 298), (x + 10, 306), (240, 240, 240), -1)
    for x in range(200, 1100, 90):                 # lower hash row at y=430
        cv2.rectangle(img, (x, 428), (x + 10, 436), (240, 240, 240), -1)
    pts = detect_hashes(img, FieldDetectConfig())
    ys = sorted(p[1] for p in pts)
    assert len(pts) >= 16
    assert max(ys) - min(ys) > 100
    assert any(abs(y - 302) < 15 for _, y in pts) and any(abs(y - 432) < 15 for _, y in pts)


def test_detect_lines_masks_player_box():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines
    img = np.full((720, 1280, 3), (40, 120, 40), np.uint8)
    cv2.line(img, (600, 60), (600, 660), (240, 240, 240), 4)            # real yard line
    cv2.rectangle(img, (300, 200), (360, 480), (250, 250, 250), -1)     # white jersey blob
    masked = detect_lines(img, FieldDetectConfig(), player_boxes=[(300, 200, 360, 480)])
    xs = [round(0.5 * (s.p0[0] + s.p1[0])) for s in masked]
    assert any(abs(x - 600) < 25 for x in xs)            # real line kept
    assert not any(abs(x - 330) < 25 for x in xs)        # jersey blob removed


def test_merge_collinear_preserves_slope():
    # Two pieces of the SAME slanted line (slope: x increases 100px over 400px of y).
    from nfl_gsplat.calibration.field_detect import _merge_collinear
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_identify import line_x_at
    a = YardLineSeg((400.0, 0.0), (450.0, 200.0))
    b = YardLineSeg((450.0, 200.0), (500.0, 400.0))
    merged = _merge_collinear([a, b])
    assert len(merged) == 1
    m = merged[0]
    # slope preserved: x at top ~400, x at bottom ~500 (old code collapsed both to ~450)
    assert abs(line_x_at(m, 0.0) - 400.0) < 2.0
    assert abs(line_x_at(m, 400.0) - 500.0) < 2.0
    # spans the union of member y-ranges
    ys = sorted([m.p0[1], m.p1[1]])
    assert ys[0] <= 1.0 and ys[1] >= 399.0


def test_merge_collinear_keeps_distinct_slanted_lines_apart():
    # Two parallel slanted lines ~60px apart must NOT merge.
    from nfl_gsplat.calibration.field_detect import _merge_collinear
    from nfl_gsplat.calibration.field_features import YardLineSeg
    a = YardLineSeg((400.0, 0.0), (500.0, 400.0))
    b = YardLineSeg((460.0, 0.0), (560.0, 400.0))
    assert len(_merge_collinear([a, b])) == 2
