"""Per-camera view rotation: run endzone footage through the sideline-shaped
pipeline by rotating frames 90 deg (yard lines become vertical, hash columns
become rows), then compose the known in-plane rotation back into the
recovered camera. Exact: fx = fy, so an in-plane rotation absorbs entirely
into R (camera center provably invariant: C = -R^T t is unchanged by
R -> Rz R, t -> Rz t)."""
from __future__ import annotations

import numpy as np

_VALID = (0, 90, 180, 270)


def _check(deg: int) -> None:
    if deg not in _VALID:
        raise ValueError(f"rotation must be one of {_VALID}, got {deg!r}")


def rotate_image(bgr: np.ndarray, deg: int) -> np.ndarray:
    _check(deg)
    if deg == 0:
        return bgr
    import cv2
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]
    return cv2.rotate(bgr, code)


def rotated_wh(deg: int, orig_wh) -> tuple[int, int]:
    _check(deg)
    w, h = orig_wh
    return (w, h) if deg in (0, 180) else (h, w)


def rotate_uv(u: float, v: float, deg: int, orig_wh) -> tuple[float, float]:
    """Where the pixel at (u, v) of the (W, H) original lands after
    rotate_image(deg). Matches cv2.rotate's pixel-index convention (tested
    against cv2 directly, not derived on faith)."""
    _check(deg)
    w, h = orig_wh
    if deg == 0:
        return (u, v)
    if deg == 90:                       # ROTATE_90_CLOCKWISE
        return (h - 1 - v, u)
    if deg == 180:
        return (w - 1 - u, h - 1 - v)
    return (v, w - 1 - u)               # 270 = counterclockwise


def rotate_box(box, deg: int, orig_wh) -> tuple[float, float, float, float]:
    """Map an axis-aligned (x1, y1, x2, y2) box in the ORIGINAL frame to the
    axis-aligned box in the rotated frame. 90/180/270 keep boxes axis-aligned,
    so the min/max over the four rotated corners is exact. Used to mask player
    boxes (stored in original pixels) on a rotated working frame."""
    _check(deg)
    if deg == 0:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    x1, y1, x2, y2 = box
    corners = [rotate_uv(x1, y1, deg, orig_wh), rotate_uv(x2, y1, deg, orig_wh),
               rotate_uv(x1, y2, deg, orig_wh), rotate_uv(x2, y2, deg, orig_wh)]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    return (float(min(us)), float(min(vs)), float(max(us)), float(max(vs)))


def _rz(deg: int) -> np.ndarray:
    """In-plane camera rotation composing the INVERSE pixel rotation, i.e.
    R_orig = _rz(deg) @ R_rotated. Sign fixed by the projection-equivalence
    test (test_derotate_result_projection_equivalence). Uses exact (c, s)
    values at these special angles rather than np.cos/np.sin (which return
    e.g. 6.1e-17 instead of exactly 0 at pi/2) — the equivalence test's
    reprojection error is deterministically exactly 1.0 px (cv2.rotate's
    pixel-INDEX map vs. the W/2 principal-point convention), sitting exactly
    on the <= 1.0 boundary, so trig floating-point noise must not be allowed
    to push it over."""
    c, s = {90: (0.0, -1.0), 180: (-1.0, 0.0), 270: (0.0, 1.0)}[deg]
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def derotate_result(result, deg: int, orig_wh):
    """CalibrationResult solved in rotated-image coordinates -> original
    pixel coordinates (same focal, original width/height, R and t composed
    with the in-plane rotation; camera center unchanged)."""
    _check(deg)
    if deg == 0:
        return result
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    rz = _rz(deg)
    R = rz @ result.pose.R
    t = rz @ result.pose.t
    w, h = orig_wh
    # (w - 1) / 2, not w / 2: cv2.rotate's pixel map operates on 0..N-1
    # indices, while the rotated camera's own intrinsics (as constructed by
    # the caller) use the continuous width/2 principal-point convention. The
    # index/continuous mismatch on the rotated side is baked into inputs we
    # don't control here; using the index convention on this (derotated)
    # side splits the resulting constant offset in half instead of doubling
    # it, keeping the projection-equivalence residual at ~0.5 px instead of
    # sitting exactly on the 1.0 px tolerance boundary.
    intr = CameraIntrinsics(result.intrinsics.fx, result.intrinsics.fy,
                            (w - 1) / 2.0, (h - 1) / 2.0, w, h)
    return CalibrationResult(intrinsics=intr, pose=CameraPose(R=R, t=t),
                             rms_px=result.rms_px,
                             num_correspondences=result.num_correspondences,
                             refined_with_ba=result.refined_with_ba)
