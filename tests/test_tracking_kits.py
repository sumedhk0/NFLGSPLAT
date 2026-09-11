"""tracking.kits: kit labels per detection box; 08b's cross-camera pairing refuses cross-kit pairs."""
import importlib.util
import pathlib

import numpy as np
import pytest

from nfl_gsplat.identity import team_color as tc
from nfl_gsplat.tracking import kits


def test_labels_from_margins_gate_and_sign():
    m = np.array([0.9, -0.7, 0.2, -0.39, np.nan, 0.4, -0.4])
    lab = kits.labels_from_margins(m, margin=0.4)
    assert lab.tolist() == [1, 0, -1, -1, -1, 1, 0]


def test_kit_margins_reads_boxes_in_frame_order(monkeypatch, tmp_path):
    """Two frames, three boxes; the torso colours are stubbed so no video is read."""
    det = {3: np.array([[0, 0, 10, 40], [20, 0, 30, 40]], float), 7: np.array([[5, 5, 15, 45]], float)}
    sats = np.array([150.0, 40.0, 145.0])            # red, white, red in frame order

    def fake_colours(df, video, *, max_per_frame=64):
        assert df["frame"].tolist() == [3, 3, 7]
        return np.column_stack([np.zeros(3), sats, np.zeros(3)])

    monkeypatch.setattr(kits, "detection_colours", fake_colours)
    monkeypatch.setattr(kits, "MIN_COLOURED", 2)
    marg, (lo, hi) = kits.kit_margins(det, "unused.mp4")
    assert lo < hi
    assert (marg[3] > 0).tolist() == [True, False] and marg[7][0] > 0


def test_kit_margins_refuses_without_colour(monkeypatch):
    det = {0: np.array([[0, 0, 10, 40]], float)}
    monkeypatch.setattr(kits, "detection_colours",
                        lambda df, video, *, max_per_frame=64: np.full((1, 3), np.nan))
    with pytest.raises(kits.KitError):
        kits.kit_margins(det, "unused.mp4")


def test_votes_from_margins_majority_and_tie():
    keys = [("s", 1)] * 3 + [("s", 2)] * 2 + [("e", 1)]
    m = [0.5, 0.6, -0.9, 0.3, -0.7, np.nan]
    out = tc.votes_from_margins(keys, m)
    assert out[("s", 1)] == tc.SATURATED                # 2 votes to 1
    assert out[("s", 2)] == 0                          # a tie, mean margin negative
    assert ("e", 1) not in out                          # NaN never votes


def _load_08b():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "08b_export_play_dir.py"
    spec = importlib.util.spec_from_file_location("export_play_dir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fuse_frame_refuses_a_cross_kit_pair():
    """Two players 0.6 m apart on the turf (inside the pairing gap) seen by
    both cameras: without kits the nearer cross pair wins a coin flip; with
    kits the red box pairs with the red box."""
    m = _load_08b()
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    K = intrinsics(1920, 1080, fov_deg=14.0)
    Rs, ts = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    Re, te = look_at(np.array([-60.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0]))
    cam_s, cam_e = (K, Rs, ts), (K, Re, te)
    world = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])          # red at x=0, white at x=0.6

    def feet(cam, pts):
        K_, R_, t_ = cam
        p = (K_ @ (R_ @ pts.T + t_[:, None])).T
        return p[:, :2] / p[:, 2:3]

    fs, fe = feet(cam_s, world), feet(cam_e, world)
    kit_s = np.array([1, 0])
    kit_e = np.array([0, 1])                                        # endzone lists white first
    pts, src_s, src_e = m.fuse_frame(cam_s, fs, cam_e, fe[::-1], gap_m=1.5, kit_s=kit_s, kit_e=kit_e)
    pairs = {(int(a), int(b)) for a, b in zip(src_s, src_e) if a >= 0 and b >= 0}
    assert pairs == {(0, 1), (1, 0)}, pairs
