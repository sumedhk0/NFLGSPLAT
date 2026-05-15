"""Build a nerfstudio-format ``transforms.json`` from calibrated camera poses.

Nerfstudio consumes a JSON of ``(image path, c2w 4×4 matrix, intrinsics)``
tuples. We bypass COLMAP entirely: our PnP-solved calibration is authoritative.

Two coordinate conventions are in play here:

- **OpenCV** (what ``solve_pnp.py`` produces): camera-space is right/down/forward,
  and ``(R, t)`` is ``world→camera``.
- **Nerfstudio / OpenGL** (what ``transforms.json`` expects): camera-space is
  right/up/backward, and the matrix stored per-frame is ``camera→world``.

Conversion (OpenCV world→cam → OpenGL cam→world)::

    c2w_opencv = [[R.T, -R.T @ t], [0, 0, 0, 1]]
    c2w_opengl = c2w_opencv @ diag(1, -1, -1, 1)

i.e. negate the second and third columns of the 4×4 (flipping camera Y and Z).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from nfl_gsplat.utils.io import write_json


def opencv_pose_to_opengl_c2w(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Convert an OpenCV world→cam (R, t) to a 4×4 OpenGL cam→world matrix."""
    R_w2c = np.asarray(R_w2c, dtype=np.float64)
    t_w2c = np.asarray(t_w2c, dtype=np.float64).reshape(3)

    # OpenCV cam→world.
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w

    # OpenCV → OpenGL: flip camera Y and Z.
    flip = np.diag(np.array([1.0, -1.0, -1.0, 1.0]))
    return c2w @ flip


def build_transforms_json(
    cameras: Mapping[str, Mapping],
    frames: Mapping[str, Sequence[Path | str]],
    out_path: Path | str,
    *,
    root_dir: Path | str | None = None,
) -> Path:
    """Write a nerfstudio-style ``transforms.json``.

    ``cameras[cam]`` must contain ``K (3×3)``, ``R (3×3)``, ``t (3,)``,
    ``width``, ``height``. Per-frame pose is the single calibrated extrinsic
    for now (pan/zoom keyframe interpolation is a Phase-post hardening item).

    ``frames[cam]`` is the ordered list of image paths for that camera.
    Paths are stored **relative to ``root_dir``** (default: parent of
    ``out_path``) since nerfstudio resolves ``file_path`` relative to the
    transforms.json directory.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root_dir) if root_dir is not None else out_path.parent

    first_cam = next(iter(cameras))
    ref = cameras[first_cam]
    width = int(ref["width"])
    height = int(ref["height"])
    # All cameras must share (width, height) for a single transforms file.
    # Nerfstudio supports per-frame intrinsics, so we store fl_x etc. per frame.

    out: dict = {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        # Global ``fl_x`` etc. are required by some loaders; we set them from the
        # first camera but also override per-frame below.
        "fl_x": float(ref["K"][0][0]),
        "fl_y": float(ref["K"][1][1]),
        "cx":   float(ref["K"][0][2]),
        "cy":   float(ref["K"][1][2]),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [],
    }

    for cam_name, cam in cameras.items():
        K = np.asarray(cam["K"], dtype=np.float64)
        R = np.asarray(cam["R"], dtype=np.float64)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
        c2w = opencv_pose_to_opengl_c2w(R, t)

        for img in frames.get(cam_name, []):
            img_path = Path(img)
            try:
                rel = img_path.relative_to(root)
            except ValueError:
                rel = img_path
            out["frames"].append({
                "file_path": str(rel).replace("\\", "/"),
                "transform_matrix": c2w.tolist(),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx":   float(K[0, 2]),
                "cy":   float(K[1, 2]),
                "w": int(cam["width"]),
                "h": int(cam["height"]),
                "camera": cam_name,
            })

    write_json(out_path, out)
    return out_path
