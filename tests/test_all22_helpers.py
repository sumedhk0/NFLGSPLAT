"""Pure helpers of scripts/08_reconstruct_all22.py (no video, no GPU)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "recon_all22", Path(__file__).resolve().parents[1] / "scripts" / "08_reconstruct_all22.py")
recon = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recon)


def test_mirrored_mount_is_the_half_turn_about_the_centre():
    assert recon.mirrored_mount((60.0, 0.0, 20.0)) == (-60.0, 0.0, 20.0)
    assert recon.mirrored_mount((-95, 4, 35)) == (95.0, -4.0, 35.0)


def test_sideline_candidate_from_track_uses_only_solved_frames(tmp_path):
    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.compositing.preview_cpu import look_at

    K = np.array([[9000.0, 0, 960], [0, 9000.0, 540], [0, 0, 1]])
    R, t = look_at(np.array([0.0, -100.0, 50.0]), np.array([0.0, 0.0, 0.0]))
    n = 5
    conf = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    track = CameraTrack(K=np.stack([K] * n), R=np.stack([R] * n), t=np.stack([t] * n),
                        conf=conf, width=1920, height=1080)
    write_camera_track(tmp_path / "cameras.npz", {"sideline": track, "endzone": track}, fps=59.94)
    cand = recon.sideline_candidate_from_track(tmp_path)
    assert sorted(cand["cams"]) == [1, 2, 4]
    assert np.allclose(cand["centre"], [0.0, -100.0, 50.0], atol=1e-6)
    assert cand["quality"]["player_cost"] == 0.0
    assert abs(cand["quality"]["fov_deg"] - recon.fov_deg(K, 1920)) < 1e-9


def test_select_by_rulers_keeps_agreeing_near_unit_scales_best_first():
    nan = float("nan")
    scales = [
        (1.148, {"hashes": 1.001, "numerals": 1.185}, 25),   # rulers disagree 18%
        (1.046, {"hashes": 1.000, "numerals": 1.052}, 27),   # the right camera
        (1.226, {"hashes": 3.019, "numerals": 1.255}, 16),   # nonsense
        (0.985, {"hashes": 0.99, "numerals": 0.98}, 20),     # also fine, closer to 1
        (nan, {}, 0),                                        # nothing read
        (1.02, {"numerals": 1.02}, 6),                       # one ruler, in range
        (1.6, {"numerals": 1.6}, 6),                         # one ruler, out of range
    ]
    assert recon.select_by_rulers(scales) == [3, 5, 1]
    assert recon.select_by_rulers(scales[:1]) == []

