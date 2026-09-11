"""Refit of SMPL-X pose parameters to fused world joints (pose.refit_fused)."""
import numpy as np
import pytest

from nfl_gsplat.pose.forward_kinematics import (NUM_BODY_JOINTS, SMPLX_BODY_PARENTS,
                                                fk_forward)
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params
from nfl_gsplat.pose.refit_fused import refit_player, render_blob, rigid_start


def _rest_skeleton(seed=0):
    """A body-sized tree on SMPL-X parents: each joint a short step from its parent."""
    rng = np.random.default_rng(seed)
    rest = np.zeros((NUM_BODY_JOINTS, 3))
    for j, p in enumerate(SMPLX_BODY_PARENTS):
        if p >= 0:
            rest[j] = rest[p] + rng.normal(scale=0.25, size=3)
    return rest


def _sequence(rest, T=6, seed=1, turn=np.pi):
    """Joints of a player FACING AWAY (global orient ``turn`` about Y), walking."""
    rng = np.random.default_rng(seed)
    fwd = fk_forward(rest)
    truth, joints = [], []
    for t in range(T):
        bp = 0.15 * rng.normal(size=63)
        go = np.array([0.0, turn, 0.0]) + 0.05 * rng.normal(size=3)
        tr = np.array([20.0 + 0.3 * t, -5.0, 0.9])
        p = _pack_params(bp, go, tr)
        truth.append(p)
        joints.append(fwd(p))
    return np.stack(truth), np.stack(joints)


def test_rigid_start_recovers_a_half_turn():
    rest = _rest_skeleton()
    _truth, joints = _sequence(rest, T=1)
    go, tr = rigid_start(rest, joints[0], np.ones(NUM_BODY_JOINTS, bool))
    # Starting there, the rest skeleton laid down rigidly is near the target.
    placed = fk_forward(rest)(_pack_params(np.zeros(63), go, tr))
    assert np.linalg.norm(placed - joints[0], axis=1).mean() < 0.12
    assert abs(np.linalg.norm(go) - np.pi) < 0.3


def test_refit_recovers_joints_with_holes_and_a_turned_player():
    rest = _rest_skeleton()
    _truth, joints = _sequence(rest, T=6)
    target = np.pad(joints, ((0, 0), (0, 105), (0, 0)), constant_values=np.nan)  # 127 like 05e
    target[:, 20] = np.nan                      # a wrist never seen
    target[2, 7:9] = np.nan                     # both ankles lost one frame
    cfg = SMPLXFitConfig(max_iter=80)
    body_pose, orient, transl, valid, rms = refit_player(target, rest, cfg=cfg)
    assert valid.all()
    assert body_pose.shape == (6, 21, 3) and orient.shape == (6, 3) and transl.shape == (6, 3)
    assert np.nanmax(rms) < 0.03, rms
    fwd = fk_forward(rest)
    for t in range(6):
        got = fwd(_pack_params(body_pose[t].reshape(-1), orient[t], transl[t]))
        seen = np.isfinite(target[t, :NUM_BODY_JOINTS]).all(1)
        assert np.linalg.norm(got[seen] - joints[t][seen], axis=1).max() < 0.05


def test_refit_refuses_a_player_mostly_unseen():
    from nfl_gsplat.errors import PoseFusionError

    rest = _rest_skeleton()
    _truth, joints = _sequence(rest, T=5)
    target = joints.copy()
    target[1:, :] = np.nan                      # four of five frames empty
    with pytest.raises(PoseFusionError):
        refit_player(target, rest)


def test_render_blob_is_world_mode_and_skips_invalid_frames():
    fs = [10, 16, 22]
    fits = {7: (fs, np.zeros((3, 21, 3)), np.zeros((3, 3)), np.arange(9.0).reshape(3, 3),
                np.array([True, False, True]))}
    blob = render_blob(fits, {7: np.zeros(10)}, stride=6)
    assert blob["world"] is True and blob["cam"] == "fused" and blob["stride"] == 6
    assert sorted(blob["frames"]) == [10, 22]
    rec = blob["frames"][22][7]
    assert rec["body_pose"].shape == (21, 3) and rec["transl"].tolist() == [6.0, 7.0, 8.0]
    assert "bbox" not in rec
