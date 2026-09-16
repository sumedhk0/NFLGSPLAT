"""The loop's review tools: 05v (render strips) and 07m (a cache's own reprojection)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_render_strip_projects_the_camera_target_to_the_image_centre_and_crops_around_it():
    rs = _load("render_strips", "05v_render_strips.py")
    from nfl_gsplat.compositing.preview_cpu import intrinsics

    K = intrinsics(1920, 1080, fov_deg=50.0)
    target = np.array([10.0, -5.0, 1.0])
    eye = target + np.array([2.0, -26.0, 10.0])
    u, v = rs.project_point(K, eye, target, target)
    assert abs(u - 960) < 1e-6 and abs(v - 540) < 1e-6
    # a point behind the camera is None; a point 1 m to the side of the target lands off-centre
    assert rs.project_point(K, eye, target, eye + (eye - target)) is None
    assert abs(rs.project_point(K, eye, target, target + np.array([1.0, 0.0, 0.0]))[0] - 960) > 20
    x0, y0, x1, y1 = rs.crop_box(u, v, 160, 220)
    assert (x1 - x0, y1 - y0) == (160, 220) and x0 == 880 and y0 == 540 - int(round(220 * 0.55)) + 0


def test_render_strip_frame_path_falls_to_the_rendered_neighbour(tmp_path):
    rs = _load("render_strips", "05v_render_strips.py")
    (tmp_path / "frame_00300.png").write_bytes(b"")
    assert rs.frame_path(tmp_path, 300) == (300, tmp_path / "frame_00300.png")
    assert rs.frame_path(tmp_path, 299) == (300, tmp_path / "frame_00300.png")   # stride: 299 -> 300
    assert rs.frame_path(tmp_path, 301) == (300, tmp_path / "frame_00300.png")
    assert rs.frame_path(tmp_path, 305) == (None, None)


def test_fit_reprojection_rms_and_knee_angles():
    fr = _load("fit_reprojection", "07m_measure_fit_reprojection.py")
    pix = np.zeros((22, 2))
    uv = np.zeros((22, 2))
    uv[:6] = [3.0, 4.0]                       # 5 px off on six joints
    use = np.zeros(22, bool)
    use[:6] = True
    assert abs(fr.rms_px(pix, uv, use) - 5.0) < 1e-9
    assert np.isnan(fr.rms_px(pix, uv, use[:2].tolist() + [False] * 20))
    bp = np.zeros((21, 3))
    bp[3] = [np.radians(90), 0, 0]
    bp[4] = [0, 0, np.radians(30)]
    assert np.allclose(fr.knee_flexion_deg(bp), (90.0, 30.0))
