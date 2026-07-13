"""Tests for the local per-frame detection path (``detect_only``).

No GPU / YOLO / video dependency — ``_predict`` and ``frame_source`` are
injected fakes. Mirrors the injection style used in tests/test_tracking.py.
"""
from __future__ import annotations


def test_detect_only_maps_boxes_to_rows(monkeypatch):
    import numpy as np
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS, TrackingConfig, detect_only

    # a fake per-frame detector: frame 0 has one box, frame 1 none
    class _Box:
        def __init__(self, xyxy, conf):
            import numpy as _np
            self.xyxy = _np.array([xyxy], float)
            self.conf = _np.array([conf], float)

    def fake_predict(bgr):
        return _Box([10.0, 20.0, 30.0, 60.0], 0.9) if bgr.sum() == 0 else None

    frames = [(0, np.zeros((8, 8, 3), np.uint8)), (1, np.ones((8, 8, 3), np.uint8))]
    df = detect_only("v.mp4", "endzone", TrackingConfig(),
                     frame_source=iter(frames), _predict=fake_predict)
    assert list(df.columns) == TRACK_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert (row.frame, row.cam, row.track_id) == (0, "endzone", -1)
    assert (row.bbox_x1, row.bbox_y2) == (10.0, 60.0)
    assert row.foot_v == 60.0 and row.foot_u == 20.0     # bottom-center
