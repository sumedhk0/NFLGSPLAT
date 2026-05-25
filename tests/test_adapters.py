"""Contract tests for the three GPU model adapters (T1.2 / T1.5 / T2.4).

The real models run on PACE GPUs; here we monkeypatch each adapter's model seam
and assert the I/O glue maps raw outputs into our NPZ schemas correctly. These
are CPU-only and intentionally do NOT exercise torch / the third_party repos.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars import lhm_wrapper
from nfl_gsplat.avatars.library import AVATAR_KEYS
from nfl_gsplat.avatars.lhm_wrapper import LHMConfig
from nfl_gsplat.pose import smplestx_infer
from nfl_gsplat.pose.smplestx_infer import NUM_SMPLX_JOINTS, SMPLestXConfig


# --- T1.2 SMPLest-X ---------------------------------------------------------

def _raw_smplestx_sample():
    return {
        "betas": np.zeros(10),
        "body_pose": np.zeros((21, 3)),
        "global_orient": np.zeros(3),
        "transl": np.array([1.0, 2.0, 3.0]),
        "joints3d_cam": np.zeros((NUM_SMPLX_JOINTS, 3)),
        "joints2d": np.zeros((NUM_SMPLX_JOINTS, 2)),
        "confidence": np.ones(NUM_SMPLX_JOINTS),
    }


def test_smplestx_assembles_schema(monkeypatch):
    monkeypatch.setattr(smplestx_infer, "check_prerequisites", lambda cfg: None)
    monkeypatch.setattr(smplestx_infer, "_load_smplestx_model", lambda cfg: object())
    monkeypatch.setattr(
        smplestx_infer, "_smplestx_forward",
        lambda model, crops, bboxes, cfg: [_raw_smplestx_sample() for _ in range(len(crops))],
    )
    crops = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    bboxes = np.zeros((3, 4))
    out = smplestx_infer.infer_crops(crops, bboxes, SMPLestXConfig())
    assert out["betas"].shape == (3, 10)
    assert out["body_pose"].shape == (3, 21, 3)
    assert out["joints3d_cam"].shape == (3, NUM_SMPLX_JOINTS, 3)
    assert np.allclose(out["transl"][0], [1.0, 2.0, 3.0])


# --- T1.5 LHM++ -------------------------------------------------------------

def _raw_lhm(n=200, sh_degree=0):
    k = (sh_degree + 1) ** 2
    return {
        "xyz": np.random.default_rng(0).normal(size=(n, 3)),
        "rot": np.tile([1.0, 0, 0, 0], (n, 1)),
        "scale": np.full((n, 3), -3.0),
        "opacity": np.ones(n),
        "sh": np.zeros((n, 3, k)),
        "lbs_weights": np.eye(22)[np.random.default_rng(1).integers(0, 22, n)],
    }


def test_lhm_assembles_canonical_schema(monkeypatch):
    monkeypatch.setattr(lhm_wrapper, "pick_tier", lambda cfg, free_gb=None: "lhm_mini")
    monkeypatch.setattr(lhm_wrapper, "_load_lhm_model", lambda tier, cfg: object())
    monkeypatch.setattr(lhm_wrapper, "_forward_lhm", lambda model, crop, cfg: _raw_lhm())
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    av = lhm_wrapper.generate_avatar(crop, LHMConfig(model_choice="lhm_mini"))
    for k in AVATAR_KEYS:
        assert k in av
    assert av["canonical_xyz"].shape == (200, 3)
    assert av["canonical_rot"].shape == (200, 4)
    assert av["lbs_weights"].shape == (200, 22)
    assert av["tier"][0] == "lhm_mini"


# --- T2.4 3DGS-Avatar -------------------------------------------------------

def test_gdgs_train_hero_validates_and_returns(monkeypatch, tmp_path):
    from nfl_gsplat.avatars import gdgs_avatar_train as g
    from nfl_gsplat.utils.io import write_npz

    repo = tmp_path / "repo"
    repo.mkdir()
    out_dir = tmp_path / "hero"

    def fake_check_call(cmd):
        # Pretend the repo trained and exported a canonical-schema avatar.
        write_npz(
            out_dir / "avatar.npz",
            canonical_xyz=np.zeros((10, 3), np.float32),
            canonical_rot=np.tile([1, 0, 0, 0], (10, 1)).astype(np.float32),
            canonical_scale=np.zeros((10, 3), np.float32),
            canonical_opacity=np.zeros(10, np.float32),
            canonical_sh=np.zeros((10, 3, 1), np.float32),
            lbs_weights=np.eye(22)[np.zeros(10, int)].astype(np.float32),
        )

    monkeypatch.setattr(g.subprocess, "check_call", fake_check_call)
    cfg = g.GDGSAvatarConfig(repo_dir=repo)
    out = g.train_hero(tmp_path / "poses.npz", tmp_path / "crops", out_dir, cfg)
    assert out.exists() and out.name == "avatar.npz"
