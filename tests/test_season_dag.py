"""Season SLURM DAG plan assembly (T2.1) + per-uid avatar build (T1.8/S3)."""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from nfl_gsplat.avatars.build_one import build_one_avatar, reference_path
from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.season.dag import build_submission_plan
from nfl_gsplat.utils.io import write_npz


def _cfg():
    return OmegaConf.create({
        "season": 2024,
        "games": ["game_001", "game_002"],
        "slurm": {"account": "gatech", "partition": "gpu-h100", "gpu": "h100:1",
                  "cpus_per_task": 8, "mem": "64G", "time_field": "02:00:00",
                  "time_perception": "01:00:00", "time_avatar": "04:00:00",
                  "time_render": "01:00:00"},
    })


def test_plan_has_all_stages_in_order():
    plan = build_submission_plan(_cfg(), plays_dir="/nonexistent")
    text = "\n".join(plan)
    assert "field_recon.sbatch game_001" in text
    assert "perception_array.sbatch" in text
    assert "collect_uids" in text and "avatar_build_array.sbatch" in text
    assert "render_array.sbatch" in text
    # SLURM allocation flags are threaded through.
    assert "-A gatech" in text and "--gres=gpu:h100:1" in text
    # Ordering: field/perception before the collect+avatar tail before render.
    assert text.index("perception_array") < text.index("collect_uids") < text.index("render_array")


def test_plan_uses_play_count_when_list_present(tmp_path):
    (tmp_path / "game_001.txt").write_text("play_001\nplay_002\nplay_003\n")
    cfg = OmegaConf.create({**_cfg(), "games": ["game_001"]})
    plan = build_submission_plan(cfg, plays_dir=tmp_path)
    assert "--array=1-3" in "\n".join(plan)


# --- build_one (S3 per-uid task) -------------------------------------------

def _fake_avatar(crop, cfg):
    n = 50
    return {
        "canonical_xyz": np.zeros((n, 3), np.float32),
        "canonical_rot": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "canonical_scale": np.zeros((n, 3), np.float32),
        "canonical_opacity": np.zeros(n, np.float32),
        "canonical_sh": np.zeros((n, 3, 1), np.float32),
        "lbs_weights": np.eye(22)[np.zeros(n, int)].astype(np.float32),
    }


def test_build_one_generates_and_stores(tmp_path):
    root = tmp_path / "library"
    lib = AvatarLibrary(root, season=2024)
    # Perception writes the reference crop + betas for the uid.
    write_npz(reference_path(root, "2024", "qb_12"),
              crop=np.zeros((64, 64, 3), np.uint8), betas=np.arange(10, dtype=np.float32))

    build_one_avatar("2024", "qb_12", lib, generate_fn=_fake_avatar)
    assert lib.has_avatar("qb_12")
    assert np.allclose(lib.get_betas("qb_12"), np.arange(10))


def test_build_one_skips_when_cached(tmp_path):
    root = tmp_path / "library"
    lib = AvatarLibrary(root, season=2024)
    write_npz(reference_path(root, "2024", "wr_81"),
              crop=np.zeros((64, 64, 3), np.uint8), betas=None if False else np.zeros(10, np.float32))
    calls = []

    def gen(crop, cfg):
        calls.append(1)
        return _fake_avatar(crop, cfg)

    build_one_avatar("2024", "wr_81", lib, generate_fn=gen)
    build_one_avatar("2024", "wr_81", lib, generate_fn=gen)   # second is a no-op
    assert len(calls) == 1
