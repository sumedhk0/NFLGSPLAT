"""The number rows as a cross-field ruler for the paint calibration.

WHY. Paint has no scale: yard lines fix the camera's x axis, but nothing in
the paint says how far apart the two sidelines are in the image's depth
direction, and the box-height ruler (from_players) fixes the lens for the
players' patch only. Reading the numerals (field.yard_numbers) measured
the rows at y = +-11.0 m in the solved frame of play 1; the rule book puts
them at +-14.33 m. Every player's cross-field position -- and the drawn
field's width under them -- was 22% short.

HOW. The measured row positions give an affine correction of the solved
frame's y: ``y_true = s * y + o``. The ground homography of the solved
camera, H (turf (x, y, 1) -> pixels), composed with that correction is the
homography of the corrected world, and a camera is read back out of it the
classic way: with a centred principal point and square pixels the first two
columns of K^-1 H must be orthogonal and equal in length, which fixes the
focal length; the columns then give R and t. The corrected camera puts the
rows where the rule book does, keeps every yard line where it was, and is
checked -- not assumed -- against the players: their heights under it must
still read 1.85 m (scripts/08d prints both).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NUMBER_BOTTOM_Y_M, NUMBER_TOP_Y_M
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

ROW_Y_M: float = 0.5 * (NUMBER_BOTTOM_Y_M + NUMBER_TOP_Y_M)
MIN_ROW_READINGS: int = 2


@dataclass(frozen=True)
class RowScale:
    scale: float        # y_true = scale * y_solved + offset
    offset: float
    n_far: int
    n_near: int
    residual_m: float   # median absolute deviation of the readings about the fit


def fit_row_scale(ys_solved, sides) -> RowScale:
    """Affine y correction from numerals read at ``ys_solved`` on rows
    ``sides`` (+1 / -1). Both rows -> scale and offset through each row's
    MEDIAN; one row -> scale alone (offset 0), which is all one row can say.

    Medians, not least squares: a frame where the camera track is poor
    reads its numerals metres off, and on play 1 two such frames out of
    eight pulled a least-squares scale from 1.27 to 1.12 with a 3.8 m
    residual. ``residual_m`` is the median absolute deviation of the
    readings about the fit, so a noisy set still reports itself.
    """
    y = np.asarray(ys_solved, float)
    side = np.asarray(sides, int)
    if len(y) < MIN_ROW_READINGS:
        raise ValueError(f"{len(y)} numeral readings; need {MIN_ROW_READINGS}")
    far, near = y[side > 0], y[side < 0]
    n_far, n_near = len(far), len(near)
    if n_far and n_near:
        y_far, y_near = float(np.median(far)), float(np.median(near))
        s = 2.0 * ROW_Y_M / (y_far - y_near)
        o = ROW_Y_M - s * y_far
    else:
        s = float(np.median(side * ROW_Y_M / y))
        o = 0.0
    res = float(np.median(np.abs(s * y + o - side * ROW_Y_M)))
    return RowScale(float(s), float(o), n_far, n_near, res)


def ground_homography(K, R, t):
    return np.asarray(K, float) @ np.column_stack([np.asarray(R, float)[:, :2],
                                                   np.asarray(t, float).reshape(3)])


def corrected_homography(H, rs: RowScale):
    """Homography of the corrected world: ``(x, y_true, 1) -> pixels``."""
    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0 / rs.scale, -rs.offset / rs.scale],
                  [0.0, 0.0, 1.0]])
    return np.asarray(H, float) @ T


def camera_from_homography(H, width: int, height: int):
    """``(K, R, t)`` with a centred principal point and one focal length.

    ``G = C H`` with ``C`` removing the principal point; the columns of
    ``diag(1/f, 1/f, 1) G`` must satisfy ``m1 . m2 = 0`` and ``|m1| = |m2|``;
    each is linear in ``1/f^2`` and the two are solved together in least
    squares. ``t`` is scaled so ``|m1| = |m2| = 1`` and signed so the
    camera stands above the turf.
    """
    H = np.asarray(H, float)
    cx, cy = width / 2.0, height / 2.0
    C = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    G = C @ H
    g1, g2 = G[:, 0], G[:, 1]
    # a * (1/f^2) + b = 0 for each constraint.
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1],
                  g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    inv_f2 = -(a @ b) / (a @ a)
    if not np.isfinite(inv_f2) or inv_f2 <= 0:
        raise ValueError("homography admits no real focal length")
    f = 1.0 / np.sqrt(inv_f2)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    M = np.linalg.inv(K) @ H
    m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2]
    lam = 0.5 * (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, tv = m1 / lam, m2 / lam, m3 / lam
    r3 = np.cross(r1, r2)
    U, _S, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    centre = -R.T @ tv
    if centre[2] < 0:                       # the other sign of the homography
        R, tv = camera_from_homography(-H, width, height)[1:]
    return K, R, tv


def refine_camera(K, R, t, rs: RowScale, width: int, height: int):
    """The camera whose yard lines stay put and whose number rows land on
    the rule book's, from the solved camera and the row fit."""
    H = corrected_homography(ground_homography(K, R, t), rs)
    return camera_from_homography(H, width, height)


def refine_track(track, rs: RowScale):
    """Every frame of a CameraTrack refined by the same row fit."""
    from nfl_gsplat.calibration.cameras_io import CameraTrack

    K = np.empty_like(track.K)
    R = np.empty_like(track.R)
    t = np.empty_like(track.t)
    for i in range(len(track.R)):
        K[i], R[i], t[i] = refine_camera(track.K[i], track.R[i], track.t[i], rs,
                                         track.width, track.height)
    return CameraTrack(K=K, R=R, t=t, conf=track.conf.copy(),
                       width=track.width, height=track.height)
