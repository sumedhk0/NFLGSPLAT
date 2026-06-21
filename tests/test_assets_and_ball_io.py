"""Generic asset authoring (T1.6) + ball.npz round-trip (T1.7)."""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars.generic_assets import make_referee_avatar
from nfl_gsplat.avatars.library import AVATAR_KEYS, AvatarLibrary
from nfl_gsplat.ball.ball_io import build_and_write_ball_track, read_ball_npz, write_ball_npz
from nfl_gsplat.ball.kalman_3d import BallKalmanConfig
from nfl_gsplat.calibration.cameras_io import constant_track
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


# --- T1.6 referee avatar ----------------------------------------------------

def test_referee_avatar_has_canonical_schema():
    av = make_referee_avatar(num_gaussians=500)
    for k in AVATAR_KEYS:
        assert k in av
    assert av["canonical_xyz"].shape == (500, 3)
    # LBS weights are a valid convex one-hot over 22 joints.
    assert av["lbs_weights"].shape == (500, 22)
    assert np.allclose(av["lbs_weights"].sum(axis=1), 1.0)


def test_referee_avatar_loads_into_library(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    lib.put_referee_avatar(make_referee_avatar(num_gaussians=300))
    assert lib.has_referee_avatar()
    assert lib.get_referee_avatar()["canonical_xyz"].shape == (300, 3)


# --- T1.7 ball.npz ----------------------------------------------------------

def test_ball_npz_roundtrip(tmp_path):
    T = 8
    xyz = np.random.default_rng(0).normal(size=(T, 3))
    vel = np.random.default_rng(1).normal(size=(T, 3))
    visible = np.array([True] * 6 + [False] * 2)
    p = write_ball_npz(tmp_path / "ball.npz", xyz, vel, visible)
    d = read_ball_npz(p)
    assert np.allclose(d["xyz"], xyz.astype(np.float32))
    assert np.allclose(d["vel"], vel.astype(np.float32))
    assert d["visible"].tolist() == visible.tolist()


def _cam(off: float):
    intr = CameraIntrinsics(fx=1400, fy=1400, cx=960, cy=540, width=1920, height=1080)
    return intr, CameraPose(R=np.eye(3), t=np.array([off, 0.0, 60.0]))


def test_build_and_write_ball_track(tmp_path):
    intr_a, pose_a = _cam(-5.0)
    intr_b, pose_b = _cam(5.0)
    T = 10
    cams = {
        "a": constant_track(intr_a, pose_a, T),
        "b": constant_track(intr_b, pose_b, T),
    }
    dets = []
    for f in range(T):
        p = np.array([[-2.0 + 0.5 * f, 0.0, 3.0]])
        ua = project_points(p, intr_a.K(), pose_a.R, pose_a.t)[0]
        ub = project_points(p, intr_b.K(), pose_b.R, pose_b.t)[0]
        dets.append({"a": ua, "b": ub})
    out = build_and_write_ball_track(tmp_path / "ball.npz", dets, cams, BallKalmanConfig(fps=30.0))
    d = read_ball_npz(out)
    assert d["xyz"].shape == (10, 3) and d["vel"].shape == (10, 3)
    assert d["visible"].any()
