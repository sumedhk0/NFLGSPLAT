"""CPU-testable cores of the per-stage CLIs (the GPU/video seams are mocked or
exercised on PACE). Covers: camera loading, FK fit-forward, track windowing,
ball detection assembly, the pose numerical chain, and the play avatar loop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.avatars.build_one import reference_path
from nfl_gsplat.avatars.build_play import build_play_avatars, player_uids
from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.ball.run_ball import detections_to_frames
from nfl_gsplat.calibration.cameras_io import load_cameras
from nfl_gsplat.pose.forward_kinematics import (
    SMPLX_BODY_PARENTS,
    fk_forward,
    posed_joint_positions,
    pose_params_to_rotmats,
)
from nfl_gsplat.pose.run_pose import solve_joint_tfms
from nfl_gsplat.tracking.detect_track import window_tracks
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points
from nfl_gsplat.utils.io import write_json, write_npz


# --- cameras_io -------------------------------------------------------------

def test_load_cameras_parses_and_skips_bookkeeping(tmp_path):
    K = [[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]
    path = tmp_path / "cameras.json"
    write_json(path, {
        "sideline": {"K": K, "R": np.eye(3).tolist(), "t": [0, 0, 0], "width": 640, "height": 480},
        "endzone": {"K": K, "R": np.eye(3).tolist(), "t": [-1, 0, 0], "width": 640, "height": 480},
        "reprojection_error_px": {"sideline": 0.4, "endzone": 0.5},
    })
    cams = load_cameras(path)
    assert set(cams) == {"sideline", "endzone"}        # bookkeeping key dropped
    intr, pose = cams["sideline"]
    assert intr.fx == 500.0 and intr.width == 640
    assert pose.R.shape == (3, 3)


def test_load_cameras_missing_file_raises(tmp_path):
    from nfl_gsplat.errors import SetupError
    with pytest.raises(SetupError):
        load_cameras(tmp_path / "nope.json")


# --- forward kinematics fit-forward ----------------------------------------

def _rest_skeleton(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rest = np.zeros((22, 3))
    for i in range(1, 22):
        rest[i] = rest[SMPLX_BODY_PARENTS[i]] + rng.normal(0, 0.15, 3)
    return rest


def test_fk_forward_identity_pose_is_rest_plus_transl():
    rest = _rest_skeleton()
    fwd = fk_forward(rest, SMPLX_BODY_PARENTS)
    transl = np.array([1.0, -2.0, 5.0])
    p = np.zeros(69)
    p[66:69] = transl
    posed = fwd(p)
    assert np.allclose(posed, rest + transl, atol=1e-9)


def test_posed_joint_positions_single_rotation_moves_subtree():
    rest = _rest_skeleton(1)
    R = np.tile(np.eye(3), (22, 1, 1))
    # 90° about Z at the root rotates everything.
    R[0] = pose_params_to_rotmats(np.array([0, 0, np.pi / 2]), np.zeros((21, 3)))[0]
    posed = posed_joint_positions(rest, SMPLX_BODY_PARENTS, R)
    # Root stays at its rest location (no parent translation); a leaf moves.
    assert np.allclose(posed[0], rest[0], atol=1e-9)
    assert not np.allclose(posed[5], rest[5], atol=1e-3)


# --- detect_track.window_tracks --------------------------------------------

def test_window_tracks_keeps_inclusive_range():
    df = pd.DataFrame({"frame": list(range(10)), "cam": ["s"] * 10})
    out = window_tracks(df, 3, 6)
    assert out["frame"].tolist() == [3, 4, 5, 6]


# --- ball detection assembly -----------------------------------------------

def test_detections_to_frames_places_by_slot():
    s = pd.DataFrame({"frame": [10, 12], "u": [1.0, 2.0], "v": [3.0, 4.0]})
    e = pd.DataFrame({"frame": [12], "u": [5.0], "v": [6.0]})
    frames = detections_to_frames({"sideline": s, "endzone": e}, 10, 13)
    assert len(frames) == 4
    assert set(frames[0]) == {"sideline"}             # slot 0 == frame 10
    assert set(frames[2]) == {"sideline", "endzone"}  # slot 2 == frame 12
    assert frames[1] == {} and frames[3] == {}
    assert np.allclose(frames[2]["endzone"], [5.0, 6.0])


# --- pose numerical chain (triangulate → fuse → smooth → FK) ----------------

def _two_cameras():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    a = (intr, CameraPose(R=np.eye(3), t=np.zeros(3)))
    b = (intr, CameraPose(R=np.eye(3), t=np.array([-1.0, 0.0, 0.0])))  # 1 m baseline
    return {"sideline": a, "endzone": b}, K


def test_solve_joint_tfms_recovers_rest_plus_translation():
    rest = _rest_skeleton(2)
    cameras, _ = _two_cameras()
    transl = np.array([0.0, 0.0, 5.0])           # put joints in front of both cams
    T = 4
    world = np.broadcast_to(rest + transl, (T, 22, 3))   # static identity pose
    obs = {}
    for cam, (intr, pose) in cameras.items():
        uv = project_points(world.reshape(-1, 3), intr.K(), pose.R, pose.t).reshape(T, 22, 2)
        obs[cam] = {"uv": uv, "conf": np.full((T, 22), 0.95)}

    tfms = solve_joint_tfms(obs, cameras, rest, SMPLX_BODY_PARENTS)
    assert tfms.shape == (T, 22, 4, 4)
    assert np.isfinite(tfms).all()
    # Applying the recovered transforms to the rest skeleton reproduces world joints.
    homo = np.concatenate([rest, np.ones((22, 1))], axis=1)
    posed = np.einsum("jik,jk->ji", tfms[-1], homo)[:, :3]
    assert np.max(np.linalg.norm(posed - (rest + transl), axis=1)) < 0.1


# --- build_play -------------------------------------------------------------

def _fake_avatar(crop, cfg):
    n = 20
    return {
        "canonical_xyz": np.zeros((n, 3), np.float32),
        "canonical_rot": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "canonical_scale": np.zeros((n, 3), np.float32),
        "canonical_opacity": np.zeros(n, np.float32),
        "canonical_sh": np.zeros((n, 3, 1), np.float32),
        "lbs_weights": np.eye(22)[np.zeros(n, int)].astype(np.float32),
    }


def test_player_uids_excludes_generics_and_dedups():
    entities = [
        {"instance_id": "1", "player_uid": "qb_12", "entity_type": "player"},
        {"instance_id": "2", "player_uid": "qb_12", "entity_type": "player"},  # dup
        {"instance_id": "3", "player_uid": "__referee__", "entity_type": "referee"},
        {"instance_id": "4", "player_uid": "wr_81", "entity_type": "player"},
    ]
    assert player_uids(entities) == ["qb_12", "wr_81"]


def test_build_play_avatars_builds_each_player(tmp_path):
    root = tmp_path / "library"
    lib = AvatarLibrary(root, season=2024)
    entities = [
        {"instance_id": "1", "player_uid": "qb_12", "entity_type": "player"},
        {"instance_id": "3", "player_uid": "__referee__", "entity_type": "referee"},
        {"instance_id": "4", "player_uid": "wr_81", "entity_type": "player"},
    ]
    for uid in ("qb_12", "wr_81"):
        write_npz(reference_path(root, "", uid),
                  crop=np.zeros((32, 32, 3), np.uint8), betas=np.zeros(10, np.float32))

    built = build_play_avatars(entities, "2024", lib, generate_fn=_fake_avatar)
    assert built == ["qb_12", "wr_81"]
    assert lib.has_avatar("qb_12") and lib.has_avatar("wr_81")
    assert not lib.has_avatar("__referee__")     # referee is a generic asset, not built here
