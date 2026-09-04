"""Per-frame camera refinement to the painted yard lines (calibration.refine_paint)."""
import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.calibration import grid_fit as gf
from nfl_gsplat.calibration import refine_paint as rp
from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

W, H = 1920, 1080


def _camera():
    K = intrinsics(W, H, fov_deg=14.0)
    R, t = look_at(np.array([5.0, -100.0, 45.0]), np.array([0.0, 0.0, 0.0]))
    return K, R, t


def _perturb(K, R, t, *, deg=(1.2, -0.8, 0.9), focal=1.04):
    centre = -R.T @ t
    R2 = Rotation.from_euler("xyz", deg, degrees=True).as_matrix() @ R
    K2 = K.copy()
    K2[0, 0] *= focal
    K2[1, 1] *= focal
    return K2, R2, -R2 @ centre


def _painted_frame(K, R, t):
    import cv2

    img = np.full((H, W, 3), (40, 90, 40), np.uint8)
    P = K @ np.column_stack([R[:, :2], t])
    for x in np.arange(-45.72, 45.73, 4.572):
        a = P @ np.array([x, -24.0, 1.0])
        b = P @ np.array([x, 24.0, 1.0])
        if a[2] > 0 and b[2] > 0:
            cv2.line(img, tuple((a[:2] / a[2]).astype(int)), tuple((b[:2] / b[2]).astype(int)),
                     (240, 240, 240), 4)
    return img


def test_refinement_recovers_a_perturbed_camera():
    K, R, t = _camera()
    frame = _painted_frame(K, R, t)
    segs = detect_lines(frame, FieldDetectConfig(vertical_deg=gf.GRID_VERTICAL_DEG))
    K2, R2, t2 = _perturb(K, R, t)
    before, _ = gf.grid_distance_px(frame, K2, R2, t2)
    res = rp.refine_frame(segs, K2, R2, t2)
    assert res.applied and res.n_segments >= 5
    assert before > 15 and res.after_px < 2.0, (before, res.after_px)
    # Yard lines are a pencil of near-parallel lines: they pin the grid on the
    # paint but not the rotation about their own direction, which trades
    # against the focal length (0.78 deg and 4% measured after a 1.7 deg,
    # 4% perturbation). The rows (hashes, numerals) pin that; not in this
    # residual yet. The grid fit is what the render needs.
    # The priors on rotation and focal (the smallest motion along that valley)
    # bias the recovered rotation toward the start: 1.07 deg measured after a
    # 1.7 deg perturbation, against 0.78 without them.
    rot_err = np.degrees(np.linalg.norm(Rotation.from_matrix(res.R @ R.T).as_rotvec()))
    assert rot_err < 1.5, rot_err
    assert np.allclose(-res.R.T @ res.t, -R.T @ t, atol=1e-6)     # centre never moves


def test_too_few_segments_leaves_the_camera_alone():
    K, R, t = _camera()
    segs = detect_lines(np.zeros((H, W, 3), np.uint8), FieldDetectConfig())
    res = rp.refine_frame(segs, K, R, t)
    assert not res.applied and np.allclose(res.R, R) and np.allclose(res.K, K)
