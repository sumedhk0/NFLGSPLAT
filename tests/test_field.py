"""Field-reconstruction tests.

Nerfstudio / ``splatfacto`` is GPU-only and is not exercised here. The
tests below cover:

1. OpenCV → OpenGL pose conversion round-trips (this is where bugs silently
   ruin an entire field reconstruction).
2. ``transforms.json`` has the right keys, per-camera intrinsics, and
   POSIX-style relative paths.
3. The CPU-only mock PLY satisfies the smoke-test contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.field.build_transforms import (
    build_transforms_json,
    opencv_pose_to_opengl_c2w,
)
from nfl_gsplat.field.train_field import (
    read_ply_gaussian_count,
    write_mock_field_ply,
)
from nfl_gsplat.utils.geometry import project_points
from nfl_gsplat.utils.io import read_json
from tests.fixtures.generate import (
    FIXTURE_HEIGHT,
    FIXTURE_WIDTH,
    _endzone_camera,
    _sideline_camera,
)


def test_opencv_to_opengl_inverts_cleanly():
    """The 4×4 OpenGL c2w should invert back to the OpenCV w2c pose
    (after undoing the Y/Z flip)."""
    intr, pose = _sideline_camera()
    R, t = pose.R, pose.t
    c2w_gl = opencv_pose_to_opengl_c2w(R, t)

    # Undo the Y/Z flip to get OpenCV c2w.
    flip = np.diag(np.array([1.0, -1.0, -1.0, 1.0]))
    c2w_cv = c2w_gl @ flip
    R_c2w_cv = c2w_cv[:3, :3]
    t_c2w_cv = c2w_cv[:3, 3]

    # Recover w2c.
    R_back = R_c2w_cv.T
    t_back = -R_c2w_cv.T @ t_c2w_cv
    np.testing.assert_allclose(R_back, R, atol=1e-10)
    np.testing.assert_allclose(t_back, t, atol=1e-10)


def test_opengl_c2w_points_camera_toward_target():
    """OpenGL c2w has the camera looking along -Z_cam. Transforming
    (0, 0, -1) from camera to world should give a vector pointing toward the
    camera's OpenCV forward axis."""
    intr, pose = _sideline_camera()
    c2w = opencv_pose_to_opengl_c2w(pose.R, pose.t)

    forward_opengl = np.array([0.0, 0.0, -1.0, 0.0])
    forward_world = c2w @ forward_opengl

    # Row 2 of the world→cam R is the OpenCV forward axis in world coords.
    expected_forward = pose.R[2, :]
    np.testing.assert_allclose(forward_world[:3], expected_forward, atol=1e-10)


def test_build_transforms_json_structure(tmp_path: Path):
    intr_s, pose_s = _sideline_camera()
    intr_e, pose_e = _endzone_camera()
    cameras = {
        "sideline": {
            "K": intr_s.K().tolist(), "R": pose_s.R.tolist(), "t": pose_s.t.tolist(),
            "width": intr_s.width, "height": intr_s.height,
        },
        "endzone": {
            "K": intr_e.K().tolist(), "R": pose_e.R.tolist(), "t": pose_e.t.tolist(),
            "width": intr_e.width, "height": intr_e.height,
        },
    }
    # Fake frame paths under a subdir of tmp_path.
    frames_dir = tmp_path / "frames"
    (frames_dir / "sideline").mkdir(parents=True)
    (frames_dir / "endzone").mkdir(parents=True)
    s_paths = [frames_dir / "sideline" / f"r00_{i:06d}.png" for i in range(3)]
    e_paths = [frames_dir / "endzone"  / f"r00_{i:06d}.png" for i in range(2)]
    for p in s_paths + e_paths:
        p.write_bytes(b"")

    out_json = tmp_path / "transforms.json"
    build_transforms_json(cameras, {"sideline": s_paths, "endzone": e_paths}, out_json)

    data = read_json(out_json)
    assert data["camera_model"] == "OPENCV"
    assert data["w"] == FIXTURE_WIDTH and data["h"] == FIXTURE_HEIGHT
    assert len(data["frames"]) == 5
    # Per-frame intrinsics must be present.
    for fr in data["frames"]:
        assert {"fl_x", "fl_y", "cx", "cy", "w", "h", "transform_matrix", "file_path"} <= fr.keys()
        # POSIX-style relative path, rooted at transforms.json's directory.
        assert "\\" not in fr["file_path"]
        assert fr["file_path"].startswith("frames/")


def test_transform_matrix_consistent_with_projection(tmp_path: Path):
    """If we project a 3D world point using the OpenCV (K, R, t), and also
    invert the stored OpenGL c2w back to OpenCV w2c and project again,
    we should get the same pixel."""
    intr, pose = _sideline_camera()
    cameras = {
        "sideline": {
            "K": intr.K().tolist(), "R": pose.R.tolist(), "t": pose.t.tolist(),
            "width": intr.width, "height": intr.height,
        },
    }
    frame = tmp_path / "frames" / "sideline" / "r00_000000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"")
    out_json = tmp_path / "transforms.json"
    build_transforms_json(cameras, {"sideline": [frame]}, out_json)

    data = read_json(out_json)
    c2w_gl = np.array(data["frames"][0]["transform_matrix"])

    # Undo the Y/Z flip, invert to get w2c, re-project a known point.
    flip = np.diag(np.array([1.0, -1.0, -1.0, 1.0]))
    c2w_cv = c2w_gl @ flip
    R_c2w = c2w_cv[:3, :3]
    t_c2w = c2w_cv[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w

    world_pt = np.array([[5.0, 3.0, 1.0]])
    uv_direct = project_points(world_pt, intr.K(), pose.R, pose.t)
    uv_rebuilt = project_points(world_pt, intr.K(), R_w2c, t_w2c)
    np.testing.assert_allclose(uv_direct, uv_rebuilt, atol=1e-8)


def test_mock_field_ply_has_expected_gaussian_count(tmp_path: Path):
    ply = write_mock_field_ply(tmp_path / "field.ply", num_gaussians=60_000, seed=0)
    assert ply.exists()
    assert ply.stat().st_size > 0
    assert read_ply_gaussian_count(ply) == 60_000


def test_mock_field_ply_header_has_required_gs_fields(tmp_path: Path):
    ply = write_mock_field_ply(tmp_path / "field.ply", num_gaussians=100, seed=1)
    header = ply.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
    for prop in (
        "property float x", "property float y", "property float z",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
    ):
        assert prop in header, f"PLY header missing: {prop}"
