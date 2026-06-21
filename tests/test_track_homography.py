from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration import track_homography as th
from nfl_gsplat.calibration.decompose_homography import krt_to_homography
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.geometry import CameraIntrinsics


def _anchor_H(fx, yaw_deg, W=1920, H=1080):
    intr = CameraIntrinsics(fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H)
    y = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
    t = np.array([0.0, 5.0, 25.0])
    return krt_to_homography(intr.K(), R, t)


def test_blend_between_anchors_snaps_to_endpoints():
    Ha, Hb = _anchor_H(2500, 0.0), _anchor_H(2700, 12.0)
    fwd = [Ha.copy() for _ in range(5)]
    bwd = [Hb.copy() for _ in range(5)]
    blended = th.blend_chains(fwd, bwd)
    assert np.allclose(blended[0] / blended[0][2, 2], Ha / Ha[2, 2], atol=1e-9)
    assert np.allclose(blended[-1] / blended[-1][2, 2], Hb / Hb[2, 2], atol=1e-9)


def test_confidence_gap_detection_raises_with_range():
    conf = np.array([1.0, 0.9, 0.2, 0.1, 0.15, 0.95, 1.0])
    with pytest.raises(CalibrationError, match="frames 2-4"):
        th.check_confidence(conf, min_conf=0.5, max_gap=0)


def test_check_confidence_passes_when_above_threshold():
    th.check_confidence(np.array([0.8, 0.7, 0.9]), min_conf=0.5, max_gap=0)


def test_assemble_track_decomposes_each_frame():
    Ha, Hb = _anchor_H(2500, 0.0), _anchor_H(2600, 6.0)
    Hs = [Ha, (Ha + Hb) / 2, Hb]
    conf = np.ones(3)
    tr = th.assemble_track(Hs, conf, width=1920, height=1080)
    assert tr.num_frames == 3
    assert tr.K.shape == (3, 3, 3)
    assert 2000 < tr.K[0, 0, 0] < 3200


def test_anchor_frames_keep_full_confidence(monkeypatch):
    """Anchor frames must never drop below full confidence.

    The per-segment loop sets conf[anchor] = seg_conf[k], which can be
    < min_conf due to backward-tracked drift near the left anchor.  The fix
    forces conf[anchor] = 1.0 after the segment loop.  This regression test
    confirms that check_confidence does NOT raise even when seg_conf at an
    anchor position would otherwise be low (here, confidence of the
    backward-tracked far end is exactly what seg_conf assigns to conf[0]).
    """
    W, H, T = 1920, 1080, 5
    from nfl_gsplat.calibration.decompose_homography import krt_to_homography
    from nfl_gsplat.utils.geometry import CameraIntrinsics

    def Htrue(i):
        intr = CameraIntrinsics(2500, 2500, W / 2, H / 2, W, H)
        y = np.deg2rad(2.0 * i)
        R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
        return krt_to_homography(intr.K(), R, np.array([0.0, 5.0, 25.0]))

    anchors = {0: Htrue(0), T - 1: Htrue(T - 1)}

    def fake_est(video, a, b, masks, cfg):
        step = 1 if b > a else -1
        idxs = list(range(a + step, b + step, step))
        rel = [Htrue(k) @ np.linalg.inv(Htrue(a)) for k in idxs]
        # high confidence everywhere so only anchor-overwrite behavior is under test
        return rel, np.ones(len(idxs))

    monkeypatch.setattr(th, "_estimate_interframe_homographies", fake_est)
    tr = th.track_camera_sequence("v.mp4", anchors, num_frames=T, width=W, height=H)
    assert tr.conf[0] == 1.0 and tr.conf[T - 1] == 1.0


def test_low_confidence_near_anchor_does_not_raise(monkeypatch):
    """Low confidence at frames adjacent to anchors must not trigger CalibrationError.

    Before the fix, seg_conf[0] = min(1.0, c_bwd[-1]) could be < min_conf when
    backward tracking loses quality near the left anchor, overwriting conf[0]
    and causing a spurious CalibrationError at a PnP-solved anchor.
    """
    W, H, T = 1920, 1080, 7
    from nfl_gsplat.calibration.decompose_homography import krt_to_homography
    from nfl_gsplat.utils.geometry import CameraIntrinsics

    def Htrue(i):
        intr = CameraIntrinsics(2500, 2500, W / 2, H / 2, W, H)
        y = np.deg2rad(2.0 * i)
        R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
        return krt_to_homography(intr.K(), R, np.array([0.0, 5.0, 25.0]))

    anchors = {0: Htrue(0), T - 1: Htrue(T - 1)}
    LOW = 0.1   # well below default min_conf=0.35

    def fake_est(video, a, b, masks, cfg):
        step = 1 if b > a else -1
        idxs = list(range(a + step, b + step, step))
        rel = [Htrue(k) @ np.linalg.inv(Htrue(a)) for k in idxs]
        # Low confidence at the very end of each chain (i.e. adjacent to the
        # opposite anchor) — the slot that gets assigned to conf[anchor] pre-fix.
        conf_arr = np.ones(len(idxs))
        conf_arr[-1] = LOW
        return rel, conf_arr

    monkeypatch.setattr(th, "_estimate_interframe_homographies", fake_est)
    # Without the fix this would raise CalibrationError because conf[0] or
    # conf[T-1] would be overwritten with LOW < min_conf.
    tr = th.track_camera_sequence("v.mp4", anchors, num_frames=T, width=W, height=H)
    assert tr.conf[0] == 1.0 and tr.conf[T - 1] == 1.0


def test_track_camera_sequence_recovers_known_pan(monkeypatch):
    from nfl_gsplat.calibration.decompose_homography import krt_to_homography
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    W, H, T = 1920, 1080, 7
    def truth(i):
        intr = CameraIntrinsics(2500 + 20 * i, 2500 + 20 * i, W / 2, H / 2, W, H)
        y = np.deg2rad(2.0 * i)
        R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
        return intr, CameraPose(R=R, t=np.array([0.0, 5.0, 25.0]))
    Htrue = [krt_to_homography(truth(i)[0].K(), truth(i)[1].R, truth(i)[1].t) for i in range(T)]
    anchors = {0: Htrue[0], T - 1: Htrue[T - 1]}

    def fake_est(video, a, b, masks, cfg):
        step = 1 if b > a else -1
        idxs = list(range(a + step, b + step, step))
        rel = [Htrue[k] @ np.linalg.inv(Htrue[a]) for k in idxs]
        return rel, np.ones(len(idxs))
    monkeypatch.setattr(th, "_estimate_interframe_homographies", fake_est)

    tr = th.track_camera_sequence("v.mp4", anchors, num_frames=T, width=W, height=H)
    fld = np.array([[0, 0, 0], [25, 10, 0], [-20, -8, 0]], float)
    for i in range(T):
        it, pt = truth(i)
        ie, pe = tr.at(i)
        assert np.allclose(project_points(fld, it.K(), pt.R, pt.t),
                           project_points(fld, ie.K(), pe.R, pe.t), atol=2.0)
