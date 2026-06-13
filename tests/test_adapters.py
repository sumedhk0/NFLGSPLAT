"""Contract tests for the three GPU model adapters (T1.2 / T1.5 / T2.4).

The real models run on PACE GPUs; here we monkeypatch each adapter's model seam
and assert the I/O glue maps raw outputs into our NPZ schemas correctly. These
are CPU-only and intentionally do NOT exercise torch / the third_party repos.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars import lhm_wrapper
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


# --- T1.5 LHM++ (Option A: LHM-native avatars) ------------------------------

def _raw_lhm_native(n=200, n_query=120, k=4):
    rng = np.random.default_rng(0)
    return {
        "app_xyz": rng.normal(size=(n, 3)),
        "app_rot": np.tile([1.0, 0, 0, 0], (n, 1)),
        "app_scale": np.full((n, 3), -3.0),
        "app_opacity": np.ones(n),
        "app_sh": np.zeros((n, k, 3)),
        "query_points": rng.normal(size=(n_query, 3)),
        "neutral_transform": np.eye(4),
    }


def test_lhm_assembles_native_schema(monkeypatch):
    from nfl_gsplat.avatars.lhm_wrapper import LHM_NATIVE_KEYS, is_lhm_native

    monkeypatch.setattr(lhm_wrapper, "pick_tier", lambda cfg, free_gb=None: "lhm_mini")
    monkeypatch.setattr(lhm_wrapper, "_load_lhm_model", lambda tier, cfg: object())
    monkeypatch.setattr(
        lhm_wrapper, "_forward_lhm",
        lambda model, crop, cfg, betas=None: _raw_lhm_native(),
    )
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    av = lhm_wrapper.generate_avatar(crop, LHMConfig(model_choice="lhm_mini"))
    for k in LHM_NATIVE_KEYS:
        assert k in av
    assert is_lhm_native(av)
    assert av["app_xyz"].shape == (200, 3)
    assert av["query_points"].shape == (120, 3)
    assert av["tier"][0] == "lhm_mini"


def test_library_stores_lhm_native_roundtrip(tmp_path):
    from nfl_gsplat.avatars.library import AvatarLibrary

    av = {k: np.asarray(v, dtype=np.float32) for k, v in _raw_lhm_native().items()}
    av["tier"] = np.array(["lhm_mini"])
    lib = AvatarLibrary(root=tmp_path, season=2024)
    lib.put_avatar("p_42", av)
    loaded = lib.get_avatar("p_42")
    assert loaded["query_points"].shape == (120, 3)
    assert "lbs_weights" not in loaded  # native blob, not canonical


def test_avatar_batch_routes_lhm_native_to_animate_fn():
    from nfl_gsplat.compositing.scene import avatar_batch

    av = {k: np.asarray(v, dtype=np.float32) for k, v in _raw_lhm_native(n=50).items()}

    def fake_animate(avatar, smplx_params):
        m = avatar["app_xyz"].shape[0]
        return {
            "xyz": np.zeros((m, 3), np.float32), "rot": np.zeros((m, 4), np.float32),
            "scale": np.zeros((m, 3), np.float32), "opacity": np.zeros(m, np.float32),
            "sh": np.zeros((m, 1, 3), np.float32),
        }

    batch = avatar_batch(av, smplx_params={"betas": np.zeros((1, 10))},
                         animate_fn=fake_animate)
    assert batch.xyz.shape == (50, 3)


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
