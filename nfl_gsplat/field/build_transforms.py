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


from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.utils.logging import get_logger  # noqa: E402

_LOG = get_logger(__name__)


def build_transforms_json(
    tracks: Mapping,
    frames_by_cam: Mapping[str, Sequence[tuple[int, "str | Path"]]],
    out_path: "Path | str",
    *,
    camera_model: str = "OPENCV",
    root_dir: "Path | str | None" = None,
    min_conf: float | None = 1e-9,
) -> Path:
    """Write a nerfstudio-style ``transforms.json``.

    ``tracks[cam]`` is a :class:`~nfl_gsplat.calibration.cameras_io.CameraTrack`;
    per-frame intrinsics and pose come from ``tracks[cam].at(frame_index)``.
    ``frames_by_cam[cam]`` is an ordered sequence of ``(frame_index, image_path)``
    pairs; ``frame_index`` is passed to ``.at()`` so pan/tilt/zoom is captured
    per frame.

    Paths are stored **relative to ``root_dir``** (default: parent of
    ``out_path``) since nerfstudio resolves ``file_path`` relative to the
    transforms.json directory.

    Frames whose track confidence is at or below ``min_conf`` are DROPPED. A
    zero-confidence frame does not carry "a slightly worse camera" -- it carries
    a camera interpolated from its nearest valid neighbour, which during an
    endzone pan was measured 8 px off the paint. Training on those teaches the
    splat a field that is not there, and nothing downstream would flag it,
    because the pose is present and well-formed. Pass ``min_conf=None`` to keep
    every frame regardless.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root_dir) if root_dir is not None else out_path.parent

    first_cam = next(iter(tracks))
    ref_track = tracks[first_cam]
    ref_intr, _ = ref_track.at(0)

    out: dict = {
        "camera_model": camera_model,
        "w": ref_track.width,
        "h": ref_track.height,
        # Global fl_x etc. are required by some loaders; set from the first
        # camera's frame-0 intrinsics, but also overridden per-frame below.
        "fl_x": ref_intr.fx,
        "fl_y": ref_intr.fy,
        "cx": ref_intr.cx,
        "cy": ref_intr.cy,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [],
    }

    dropped: dict[str, list[int]] = {}
    for cam_name, frame_list in frames_by_cam.items():
        if cam_name not in tracks:
            continue
        track = tracks[cam_name]
        for frame_index, img in frame_list:
            if min_conf is not None:
                idx = max(0, min(int(frame_index), track.num_frames - 1))
                if float(track.conf[idx]) <= min_conf:
                    dropped.setdefault(cam_name, []).append(int(frame_index))
                    continue
            intr, pose = track.at(int(frame_index))
            K = intr.K()
            c2w = opencv_pose_to_opengl_c2w(pose.R, pose.t)
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
                "w": intr.width,
                "h": intr.height,
                "camera": cam_name,
            })

    if dropped:
        for cam_name, idxs in sorted(dropped.items()):
            _LOG.warning(
                "build_transforms: dropped %d/%d %s frame(s) whose camera is "
                "unverified (conf <= %g); they would have trained on a pose "
                "interpolated from a neighbour. First few: %s",
                len(idxs), len(frames_by_cam[cam_name]), cam_name, min_conf,
                idxs[:5])
    if not out["frames"]:
        raise SetupError(
            "build_transforms: every extracted frame was dropped as "
            "unverified, so there is nothing to train on. Re-run the endzone "
            "calibration (02_autocalibrate --mode mosaic-endzone "
            "--refine-stride N) and extract frames at indices it verified, or "
            "pass min_conf=None to train on unverified poses deliberately.")
    write_json(out_path, out)
    return out_path


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import typer

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import PlayDir
    from nfl_gsplat.utils.logging import get_logger

    _log = get_logger(__name__)
    app = typer.Typer(add_completion=False)

    @app.command()
    def main(
        play_dir: Path = typer.Option(..., "--play-dir"),
        config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT,
    ) -> None:
        cfg = load_cli_config(config, config_override, set_)  # noqa: F841 — kept for uniform CLI signature
        pdir = PlayDir.from_dir(play_dir)

        # Load per-frame calibrated camera tracks from cameras.npz.
        tracks = load_camera_track(pdir.cameras_npz)

        # Discover frames written by extract_static_frames under field/frames/{cam}/.
        field_dir = pdir.dir / "field"
        frames_root = field_dir / "frames"
        cam_frames: dict[str, list[Path]] = {
            cam: sorted((frames_root / cam).glob("*.png"))
            for cam in tracks
            if (frames_root / cam).is_dir()
        }

        # Build per-frame (frame_index, path) pairs; index is extracted from the
        # file stem written by extract_static_frames, e.g. "r00_000042" → 42.
        frames_by_cam: dict[str, list[tuple[int, Path]]] = {
            cam: [(int(img.stem.split("_")[-1]), img) for img in img_list]
            for cam, img_list in cam_frames.items()
            if cam in tracks
        }

        out_json = field_dir / "transforms.json"
        build_transforms_json(tracks, frames_by_cam, out_json, root_dir=field_dir)
        # Count what was WRITTEN, not what was offered: unverified frames are
        # dropped inside, and reporting the offered total read as though every
        # extracted frame had made it into the training set.
        import json as _json
        written = len(_json.loads(out_json.read_text())["frames"])
        offered = sum(len(v) for v in frames_by_cam.values())
        _log.info("build_transforms: %d/%d frames -> %s",
                  written, offered, out_json)

    app()


if __name__ == "__main__":
    _main()
