"""08h: a frame camera from a homography by the ray fit; a near-identity step stays near identity."""
import importlib.util
import pathlib

import cv2
import numpy as np


def _mod():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "08h_endzone_track.py"
    spec = importlib.util.spec_from_file_location("endzone_track", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _H_for(R_t, f_t, R_ref, f_ref, cx, cy):
    K_ref = np.array([[f_ref, 0, cx], [0, f_ref, cy], [0, 0, 1.0]])
    K_t = np.array([[f_t, 0, cx], [0, f_t, cy], [0, 0, 1.0]])
    return K_ref @ R_ref @ R_t.T @ np.linalg.inv(K_t)


def test_ray_fit_recovers_a_pan_and_a_zoom():
    m = _mod()
    cx, cy, f_ref = 960.0, 540.0, 17600.0
    R_ref = cv2.Rodrigues(np.array([0.3, 0.0, 0.0]))[0]
    R_t = cv2.Rodrigues(np.array([0.02, 0.01, 0.003]))[0] @ R_ref
    H = _H_for(R_t, 19000.0, R_ref, f_ref, cx, cy)
    Rt, ft, rms = m.rot_focal_from_homography(H, f_ref, R_ref, cx, cy)
    assert abs(ft - 19000.0) < 5 and m._angle_deg(Rt, R_t) < 1e-3 and rms < 1e-3


def test_a_half_pixel_step_is_not_a_tenth_of_a_degree():
    """The matrix route turned a 0.4 px homography step into 0.18 deg (0.9 m
    on the ground at 300 m); the ray fit must keep it under 0.005 deg."""
    m = _mod()
    cx, cy, f_ref = 960.0, 540.0, 17600.0
    R_ref = cv2.Rodrigues(np.array([0.3, 0.0, 0.0]))[0]
    H = np.array([[0.99983, -0.00038, 0.08929], [-5e-05, 0.9993, -0.08202], [0.0, 0.0, 1.0]])
    Rt, ft, rms = m.rot_focal_from_homography(H, f_ref, R_ref, cx, cy)
    assert m._angle_deg(Rt, R_ref) < 0.005, m._angle_deg(Rt, R_ref)
    assert abs(ft / f_ref - 1.0) < 0.002
