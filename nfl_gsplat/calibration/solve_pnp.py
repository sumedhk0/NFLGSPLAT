"""PnP camera calibration from annotated NFL field landmarks.

Pipeline
--------
1. Load annotation JSON (``[{"name", "uv", "frame"}]``).
2. Map each ``name`` to a 3D world point via :data:`NFL_LANDMARKS`.
3. Solve the extrinsics with ``cv2.solvePnP`` (iterative, intrinsic guess).
4. If ≥10 correspondences exist, upgrade to ``cv2.calibrateCamera`` which
   also refines the intrinsics.
5. Optionally run a small bundle adjustment with ``scipy.optimize.least_squares``
   to jointly refine ``(fx, fy, cx, cy, r, t)``.
6. Compute reprojection RMS; raise :class:`CalibrationError` if above threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.errors import CalibrationError, SetupError
from nfl_gsplat.utils.geometry import (
    CameraIntrinsics,
    CameraPose,
    project_points,
    reprojection_rms,
)
from nfl_gsplat.utils.io import read_json


@dataclass(frozen=True)
class CalibrationResult:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    rms_px: float
    num_correspondences: int
    refined_with_ba: bool

    def as_json(self) -> dict:
        return {
            "K": self.intrinsics.K().tolist(),
            "fx": self.intrinsics.fx,
            "fy": self.intrinsics.fy,
            "cx": self.intrinsics.cx,
            "cy": self.intrinsics.cy,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "R": self.pose.R.tolist(),
            "t": self.pose.t.tolist(),
            "rms_px": self.rms_px,
            "num_correspondences": self.num_correspondences,
            "refined_with_ba": self.refined_with_ba,
        }


def _default_intrinsics_guess(width: int, height: int) -> CameraIntrinsics:
    f = float(max(width, height))
    return CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                            width=width, height=height)


def _load_correspondences(
    annotations_json: Path | str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = Path(annotations_json)
    if not path.exists():
        raise SetupError(
            f"calibration annotations not found at {path}. "
            f"Run `python scripts/02_calibrate_cameras.py --play-dir <play folder>` — see SETUP.md §3."
        )
    entries = read_json(path)
    if not isinstance(entries, list) or len(entries) == 0:
        raise SetupError(f"empty annotations file {path} — see SETUP.md §3.")

    world_pts: list[np.ndarray] = []
    uv_pts: list[np.ndarray] = []
    names: list[str] = []
    for e in entries:
        name = e["name"]
        if name not in NFL_LANDMARKS:
            raise CalibrationError(
                f"annotation refers to unknown landmark {name!r} in {path}."
            )
        world_pts.append(NFL_LANDMARKS[name])
        uv_pts.append(np.asarray(e["uv"], dtype=np.float64))
        names.append(name)
    return (
        np.stack(world_pts, axis=0).astype(np.float64),
        np.stack(uv_pts, axis=0).astype(np.float64),
        names,
    )


def _pack(
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    *, refine_intrinsics: bool,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Pack optimizable parameters. When ``refine_intrinsics`` is True we
    optimise only the focal length (aspect ratio locked to 1, principal
    point fixed). That matches the coplanar-landmark case on NFL field
    views, which otherwise leaves fy and (cx, cy) under-determined."""
    rvec, _ = cv2.Rodrigues(R)
    if refine_intrinsics:
        params = np.concatenate([[K[0, 0]], rvec.ravel(), t.ravel()])
    else:
        params = np.concatenate([rvec.ravel(), t.ravel()])
    fixed = (float(K[1, 1] / K[0, 0]), float(K[0, 2]), float(K[1, 2]))
    return params, fixed


def _unpack(
    params: np.ndarray,
    fixed: tuple[float, float, float],
    *,
    refine_intrinsics: bool,
    K_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aspect, cx, cy = fixed
    if refine_intrinsics:
        fx = float(params[0])
        rvec = params[1:4]
        t = params[4:7]
    else:
        assert K_prior is not None
        fx = float(K_prior[0, 0])
        rvec = params[0:3]
        t = params[3:6]
    fy = fx * aspect
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return K, R, t


def _residuals(
    params: np.ndarray,
    world_pts: np.ndarray, uv_gt: np.ndarray,
    fixed: tuple[float, float, float],
    *,
    refine_intrinsics: bool,
    K_prior: np.ndarray | None,
) -> np.ndarray:
    K, R, t = _unpack(params, fixed,
                      refine_intrinsics=refine_intrinsics, K_prior=K_prior)
    uv_pred = project_points(world_pts, K, R, t)
    finite = np.isfinite(uv_pred).all(axis=1)
    res = np.where(finite[:, None], uv_pred - uv_gt, np.full_like(uv_gt, 1e3))
    return res.ravel()


def solve_pnp_from_annotations(
    annotations_json: Path | str,
    *,
    image_size: tuple[int, int],
    max_reproj_px: float = 5.0,
    min_landmarks: int = 6,
    bundle_adjustment: bool = True,
    refine_intrinsics: bool = True,
    initial_intrinsics: CameraIntrinsics | None = None,
) -> CalibrationResult:
    """Solve K, R, t from annotated landmarks. See module docstring.

    Parameters
    ----------
    annotations_json
        Path to ``[{"name", "uv", "frame"}]`` JSON produced by ``annotate_gui.py``.
    image_size
        ``(width, height)`` of the annotated frame.
    max_reproj_px
        Fail with :class:`CalibrationError` if final RMS exceeds this.
    min_landmarks
        Fail with :class:`CalibrationError` if fewer correspondences exist.
    bundle_adjustment
        Run scipy least-squares refinement after the OpenCV PnP seed.
    initial_intrinsics
        Optional prior. If ``None``, guess ``fx = fy = max(W, H)`` centered principal point.
    """
    width, height = image_size
    world_pts, uv_gt, names = _load_correspondences(annotations_json)
    n = world_pts.shape[0]
    if n < min_landmarks:
        raise CalibrationError(
            f"only {n} landmark(s) annotated in {annotations_json}; need ≥{min_landmarks}. "
            f"Named landmarks used: {names}. Re-run annotate_gui.py and add more."
        )

    intr = initial_intrinsics or _default_intrinsics_guess(width, height)
    K = intr.K()
    dist = np.zeros(5, dtype=np.float64)

    obj = world_pts.reshape(-1, 1, 3).astype(np.float32)
    img = uv_gt.reshape(-1, 1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K.astype(np.float32), dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=False,
    )
    if not ok:
        raise CalibrationError(
            f"cv2.solvePnP failed on {n} correspondences from {annotations_json}."
        )

    if n >= 10 and refine_intrinsics:
        # Field landmarks are coplanar (Z=0), which makes refining fy and
        # (cx, cy) independently under-determined. Lock aspect ratio and
        # principal point so only fx (and extrinsics) move.
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_ZERO_TANGENT_DIST
        )
        K_cc = K.astype(np.float64).copy()
        _, K_cc, _, rvecs_cc, tvecs_cc = cv2.calibrateCamera(
            [obj.astype(np.float32)],
            [img.astype(np.float32)],
            (width, height),
            K_cc, dist, flags=flags,
        )
        K = K_cc
        rvec = rvecs_cc[0]
        tvec = tvecs_cc[0]

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    refined = False
    if bundle_adjustment:
        p0, fixed = _pack(K, R, t, refine_intrinsics=refine_intrinsics)
        res = least_squares(
            _residuals, p0,
            args=(world_pts, uv_gt, fixed),
            kwargs={"refine_intrinsics": refine_intrinsics, "K_prior": K},
            method="trf", loss="soft_l1", f_scale=1.0, max_nfev=200,
        )
        K, R, t = _unpack(res.x, fixed,
                          refine_intrinsics=refine_intrinsics, K_prior=K)
        refined = True

    rms = reprojection_rms(world_pts, uv_gt, K, R, t)
    if rms > max_reproj_px:
        raise CalibrationError(
            f"reprojection RMS {rms:.2f} px exceeds threshold {max_reproj_px:.2f} px "
            f"on {n} landmarks from {annotations_json}. Re-annotate or add more "
            f"landmarks; see SETUP.md §3."
        )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return CalibrationResult(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy,
                                    width=width, height=height),
        pose=CameraPose(R=R.astype(np.float64), t=t.astype(np.float64)),
        rms_px=float(rms),
        num_correspondences=n,
        refined_with_ba=refined,
    )


def solve_pair(
    annotations: Sequence[Path | str],
    image_sizes: Sequence[tuple[int, int]],
    *,
    max_reproj_px: float = 5.0,
    bundle_adjustment: bool = True,
    refine_intrinsics: bool = True,
    initial_intrinsics: Sequence[CameraIntrinsics | None] | None = None,
) -> list[CalibrationResult]:
    """Solve multiple cameras independently. (Per-camera for now; a future
    pass could couple them via correspondences of shared world points.)"""
    priors = list(initial_intrinsics) if initial_intrinsics else [None] * len(annotations)
    return [
        solve_pnp_from_annotations(
            a, image_size=s,
            max_reproj_px=max_reproj_px,
            bundle_adjustment=bundle_adjustment,
            refine_intrinsics=refine_intrinsics,
            initial_intrinsics=pri,
        )
        for a, s, pri in zip(annotations, image_sizes, priors)
    ]
