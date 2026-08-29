"""Recovering a clip's offset, and refusing when the recovery is untrustworthy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.data.align_video import (VIDEO_FPS, Alignment, align_play,
                                         field_speed, image_speed,
                                         tracking_time)
from nfl_gsplat.data.helmet_dataset import PlayTracking


def _track(n=250, snap_at=150):
    """Players still, then moving at the snap -- the onset the match keys on."""
    times = (np.arange(n) - snap_at) / 10.0
    xy = np.zeros((n, 2, 2))
    moving = np.arange(n) >= snap_at
    xy[moving, 0, 0] = np.cumsum(np.full(moving.sum(), 0.5))
    xy[moving, 1, 0] = np.cumsum(np.full(moving.sum(), 0.4))
    return PlayTracking(game_key=1, play_id=2, players=("H1", "V2"),
                        times=times, xy=xy, snap_time=snap_at / 10.0)


def _labels(video, offset_s, n_frames=400, snap_track_s=0.0):
    """Helmets still until the snap, then moving -- in VIDEO frames."""
    rows = []
    for f in range(1, n_frames + 1):
        t = offset_s + f / VIDEO_FPS
        moved = max(0.0, t - snap_track_s) * 30.0
        for i, lab in enumerate(("H1", "V2")):
            rows.append({"video": video, "frame": f, "label": lab,
                         "left": 100.0 + i * 50 + moved, "top": 200.0,
                         "width": 20.0, "height": 20.0})
    return pd.DataFrame(rows)


def test_image_speed_is_zero_before_motion_and_positive_after():
    lab = _labels("v.mp4", offset_s=-2.0)
    sp = image_speed(lab)
    assert sp.iloc[5] == pytest.approx(0.0, abs=1e-9)
    assert sp.iloc[-5] > 0.0


def test_field_speed_tracks_the_snap():
    t, v = field_speed(_track())
    assert v[10] == pytest.approx(0.0, abs=1e-9)
    assert v[-10] > 1.0


def test_both_views_agreeing_yields_an_offset():
    track = _track()
    off = -3.0
    labels = pd.concat([_labels("1_000002_Sideline.mp4", off),
                        _labels("1_000002_Endzone.mp4", off)])
    got = align_play(labels, track, 1, 2)
    assert got.ok, got.reason
    assert got.offset_s == pytest.approx(off, abs=0.2)


def test_views_that_disagree_are_REFUSED():
    """The load-bearing check: two synchronised cameras cannot disagree, so a
    disagreement means the recovery failed and must not become ground truth."""
    track = _track()
    # Both views must actually CONTAIN the motion onset, or the second refuses
    # for want of signal and never reaches the disagreement check. Shifting the
    # endzone's snap instead makes it move, but at a different apparent offset.
    labels = pd.concat([_labels("1_000002_Sideline.mp4", -3.0),
                        _labels("1_000002_Endzone.mp4", -3.0,
                                snap_track_s=2.0)])
    got = align_play(labels, track, 1, 2)
    assert not got.ok
    assert "disagree" in got.reason


def test_a_single_view_is_not_enough():
    got = align_play(_labels("1_000002_Sideline.mp4", -3.0), _track(), 1, 2)
    assert not got.ok
    assert "one view" in got.reason


def test_a_flat_profile_is_refused_rather_than_fitted():
    """Nothing moves, so nothing distinguishes one offset from another."""
    track = _track()
    flat = []
    for v in ("1_000002_Sideline.mp4", "1_000002_Endzone.mp4"):
        for f in range(1, 400):
            for i, lab in enumerate(("H1", "V2")):
                flat.append({"video": v, "frame": f, "label": lab,
                             "left": 100.0 + i * 50, "top": 200.0,
                             "width": 20.0, "height": 20.0})
    got = align_play(pd.DataFrame(flat), track, 1, 2)
    assert not got.ok


def test_tracking_time_converts_a_frame():
    assert tracking_time(0, -1.0) == pytest.approx(-1.0)
    # 59.94 frames is one second; int() of it is not, and rounding that away
    # would hide a 16 ms error -- a full video frame.
    assert tracking_time(VIDEO_FPS, -1.0) == pytest.approx(0.0, abs=1e-9)
    assert tracking_time(60, 0.0) == pytest.approx(60 / VIDEO_FPS, abs=1e-9)


def test_alignment_reports_why_it_refused():
    a = Alignment(None, {}, "only one view has labels")
    assert not a.ok and a.reason
