"""Load the per-play ``cameras.json`` into geometry objects.

``scripts/02_calibrate_cameras.py`` writes ``<play folder>/cameras.json``
as ``{cam: {K, R, t, width, height}, "reprojection_error_px": {...}}``. Every
3D stage (cross-cam re-ID, triangulation, ball Kalman) needs the same
``{cam: (CameraIntrinsics, CameraPose)}`` mapping, so it lives here once instead
of being re-parsed in each stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
from nfl_gsplat.utils.io import read_json


def _camera_from_entry(entry: dict) -> tuple[CameraIntrinsics, CameraPose]:
    K = np.asarray(entry["K"], dtype=np.float64)
    R = np.asarray(entry["R"], dtype=np.float64)
    t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
    intr = CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=int(entry["width"]), height=int(entry["height"]),
    )
    return intr, CameraPose(R=R, t=t)


def load_cameras(path: Path | str) -> dict[str, tuple[CameraIntrinsics, CameraPose]]:
    """Return ``{cam: (CameraIntrinsics, CameraPose)}`` from a cameras.json.

    Non-camera bookkeeping keys (anything whose value is not a dict carrying a
    ``"K"``) are skipped, so ``reprojection_error_px`` and friends are ignored.
    Raises :class:`SetupError` if the file is missing or has no cameras.
    """
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"cameras.json not found at {path}. Calibrate first: "
            "`python scripts/02_calibrate_cameras.py --play-dir <play folder>` — see SETUP.md §3."
        )
    raw = read_json(path)
    cams: dict[str, tuple[CameraIntrinsics, CameraPose]] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict) and "K" in entry:
            cams[name] = _camera_from_entry(entry)
    if not cams:
        raise SetupError(f"{path}: no cameras found (expected per-cam K/R/t entries).")
    return cams
