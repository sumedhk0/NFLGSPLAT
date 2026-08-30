"""The plane-fit aligner: does it recover a known offset, and refuse a bad one?"""
import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.data.align_video import (
    VIDEO_FPS,
    align_play_plane,
    best_offset_plane,
    helmet_boxes_by_frame,
    plane_residual,
)
from nfl_gsplat.data.helmet_dataset import PlayTracking

# A plausible sideline camera as a plain field-metres -> pixels homography.
H_TRUE = np.array([[9.0, 0.4, 640.0],
                   [0.6, 4.5, 360.0],
                   [0.0, 0.002, 1.0]])


def project(H, xy):
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def synthetic(n_players=22, duration=26.0, offset=-1.4, H=H_TRUE):
    """A play whose video frames are ``offset`` seconds from the tracking clock."""
    rng = np.random.default_rng(0)
    # Spanning the clip on BOTH sides, as the real record does: it starts
    # well before the snap. Tracking that began at t=0 would put the true
    # offset itself outside the window and make the fixture untestable.
    times = np.arange(-duration / 2, duration / 2, 1.0 / 10.0)
    start = rng.uniform([-20, -12], [20, 12], size=(n_players, 2))
    vel = rng.uniform(-3.0, 3.0, size=(n_players, 2))
    # Independent per-player ACCELERATION, and it is not decoration. Players
    # drifting at constant velocity deform the group affinely, and a homography
    # absorbs most of an affine error -- on this fixture a full second of
    # misalignment then costs only 6 px, near the real 8 px noise floor. Real
    # plays cost 40-90 px because players cut and accelerate independently.
    # What the method actually keys on is NON-AFFINE deformation, so a fixture
    # without it tests an easier problem than the one that exists.
    acc = rng.uniform(-1.5, 1.5, size=(n_players, 2))
    t = times[:, None, None]
    xy = start[None] + vel[None] * t + 0.5 * acc[None] * t * t
    players = tuple(f"H{i:02d}" for i in range(n_players))
    track = PlayTracking(1, 1, players, times, xy, snap_time=0.0)

    rows = []
    for frame in range(1, 400):
        pos = track.at(offset + frame / VIDEO_FPS)
        uv = project(H, pos)
        for name, (u, v) in zip(players, uv):
            rows.append({"video": "1_000001_Sideline.mp4", "frame": frame,
                         "label": name, "left": u - 10, "top": v - 10,
                         "width": 20, "height": 20})
    return track, pd.DataFrame(rows)


def test_boxes_by_frame_centres_the_box():
    track, labels = synthetic()
    byf = helmet_boxes_by_frame(labels, {p: i for i, p in enumerate(track.players)})
    uv, cols = byf[1]
    assert uv.shape == (22, 2) and cols.shape == (22,)
    expect = project(H_TRUE, track.at(-1.4 + 1 / VIDEO_FPS))[cols]
    assert np.allclose(uv, expect, atol=1e-6)


def test_boxes_by_frame_drops_thin_frames():
    track, labels = synthetic()
    thin = labels[(labels.frame != 5) | (labels.label.isin(track.players[:3]))]
    byf = helmet_boxes_by_frame(thin, {p: i for i, p in enumerate(track.players)})
    assert 5 not in byf and 6 in byf


def test_residual_is_a_sharp_minimum_at_the_truth():
    """The whole method rests on this: wrong offsets must cost far more."""
    track, labels = synthetic(offset=-1.4)
    byf = helmet_boxes_by_frame(labels, {p: i for i, p in enumerate(track.players)})
    frames = sorted(byf)[::10]
    at_truth = plane_residual(byf, track, -1.4, frames)
    assert at_truth < 1e-3
    # On real footage the floor is ~8 px, set by helmet-height variation. A
    # wrong offset must cost clearly more than that floor or the method has
    # nothing to discriminate with. Asserting a specific px number instead
    # would just encode whatever this fixture's velocities happen to produce.
    for wrong in (-3.0, -0.4, 1.0):
        assert plane_residual(byf, track, wrong, frames) > 2 * 8.0


@pytest.mark.parametrize("offset", [-1.4, -5.25, 0.0])
def test_best_offset_recovers_the_truth(offset):
    track, labels = synthetic(offset=offset)
    byf = helmet_boxes_by_frame(labels, {p: i for i, p in enumerate(track.players)})
    got, residual, contrast = best_offset_plane(byf, track)
    assert abs(got - offset) < 0.05
    assert residual < 1.0 and contrast > 10.0


def test_align_play_accepts_when_the_views_agree():
    track, side = synthetic(offset=-1.4)
    endz = synthetic(offset=-1.4, H=H_TRUE.T.copy())[1]
    endz["video"] = "1_000001_Endzone.mp4"
    out = align_play_plane(pd.concat([side, endz]), track, 1, 1)
    assert out.ok and abs(out.offset_s - (-1.4)) < 0.05


def test_align_play_refuses_when_the_views_disagree():
    """Synchronised cameras cannot really disagree, so disagreement is failure."""
    track, side = synthetic(offset=-1.4)
    endz = synthetic(offset=-4.0, H=H_TRUE.T.copy())[1]
    endz["video"] = "1_000001_Endzone.mp4"
    out = align_play_plane(pd.concat([side, endz]), track, 1, 1)
    assert not out.ok and "disagree" in out.reason


def test_offsets_off_the_end_of_the_record_are_refused():
    """The clamp trap: a frozen configuration fits itself, so it must not score.

    ``PlayTracking.at`` clamps, so an offset that shoves the clip past the end
    of the record hands every frame the same endpoint positions. Scored naively
    that is a PERFECT homography fit and beats the truth.
    """
    track, labels = synthetic(offset=-1.4)
    byf = helmet_boxes_by_frame(labels, {p: i for i, p in enumerate(track.players)})
    frames = sorted(byf)[::10]
    way_past = track.times[-1] + 5.0
    assert not np.isfinite(plane_residual(byf, track, way_past, frames))
    assert np.isfinite(plane_residual(byf, track, -1.4, frames))
