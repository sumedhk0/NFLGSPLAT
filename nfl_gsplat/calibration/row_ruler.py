"""Paint rows at a known y as a cross-field ruler for the paint calibration.

WHY. Paint has no scale: yard lines fix the camera's x axis, but nothing in
the paint says how far apart the two sidelines are in the image's depth
direction, and the box-height ruler (from_players) fixes the lens for the
players' patch only. Two kinds of paint sit at a y the rule book fixes:
the HASH MARKS at +-2.82 m (70 ft 9 in from each sideline) and the NUMERAL
rows at +-12.50 m (bottom 12 yd in, 6 ft tall, top toward midfield).
Reading both on play 1 put the hashes at +-2.65 m and the numerals at
+-11.2 m in the solved frame: the paint solve's cross-field axis was about
6% short.

TWO RULERS, ON PURPOSE. The first version of this module used the numerals
alone, against a row constant that was wrong by the glyph height (14.33 m
instead of 12.50), and "corrected" the camera by 27% -- the hash marks
showed it up. A correction of a calibration needs a second ruler before it
is applied: fit_rows takes every known row it is given, reports each
ruler's implied scale, and 08d refuses to apply when they disagree.

HOW. The measured row positions give an affine correction of the solved
frame's y: ``y_true = s * y + o``. The ground homography of the solved
camera, H (turf (x, y, 1) -> pixels), composed with that correction is the
homography of the corrected world, and a camera is read back out of it the
classic way: with a centred principal point and square pixels the first two
columns of K^-1 H must be orthogonal and equal in length, which fixes the
focal length; the columns then give R and t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.field_landmarks import (HASH_OFFSET_M,
                                                    NUMBER_BOTTOM_Y_M,
                                                    NUMBER_TOP_Y_M)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

ROW_Y_M: float = 0.5 * (NUMBER_BOTTOM_Y_M + NUMBER_TOP_Y_M)     # numeral rows, 12.50
HASH_Y_M: float = HASH_OFFSET_M                                  # hash rows, 2.82
MIN_ROW_READINGS: int = 2
# Hash-row search: strips on the 1-yard lines between two 5-yard lines,
# where the only paint is the hash tick, thresholded and summed across
# strips; the brightest row on each side of midfield within this band.
HASH_BAND_M: float = 6.0
HASH_STRIP_W_M: float = 1.0
HASH_PX_PER_M: float = 40.0
HASH_WHITE: int = 170


@dataclass(frozen=True)
class RowScale:
    scale: float        # y_true = scale * y_solved + offset
    offset: float
    n_far: int
    n_near: int
    residual_m: float   # median absolute deviation of the readings about the fit
    by_ruler: dict | None = None    # ruler name -> implied scale, when several


def _line_through_medians(y, yt):
    """(scale, offset) through the median reading of each distinct true row."""
    rows = {float(v): float(np.median(y[yt == v])) for v in np.unique(yt)}
    if len(rows) >= 2:
        A = np.column_stack([list(rows.values()), np.ones(len(rows))])
        (s, o), *_ = np.linalg.lstsq(A, np.array(list(rows)), rcond=None)
        return float(s), float(o)
    (v, m), = rows.items()
    return float(v / m), 0.0


def fit_rows(ys_solved, ys_true, *, rulers=None) -> RowScale:
    """Affine y correction from paint rows read at ``ys_solved`` whose true
    positions are ``ys_true`` (signed). Each distinct true row contributes
    its MEDIAN reading, and the line through those medians is fitted in
    least squares (two distinct rows -> exact; one -> scale alone).
    ``rulers`` names each reading's ruler ("numerals", "hashes"); the scale
    each ruler implies on its own is reported in ``by_ruler`` so that
    disagreement is visible.

    Medians, not least squares over readings: a frame where the camera
    track is poor reads its rows metres off, and on play 1 two such frames
    out of eight pulled a least-squares scale from 1.27 to 1.12 with a
    3.8 m residual.
    """
    y = np.asarray(ys_solved, float)
    yt = np.asarray(ys_true, float)
    if len(y) < MIN_ROW_READINGS:
        raise ValueError(f"{len(y)} row readings; need {MIN_ROW_READINGS}")
    s, o = _line_through_medians(y, yt)
    res = float(np.median(np.abs(s * y + o - yt)))
    by = None
    if rulers is not None:
        rulers = np.asarray(rulers)
        by = {str(name): _line_through_medians(y[rulers == name], yt[rulers == name])[0]
              for name in np.unique(rulers)}
    return RowScale(s, o, int((yt > 0).sum()), int((yt < 0).sum()), res, by)


def fit_row_scale(ys_solved, sides) -> RowScale:
    """Numeral rows only: readings at ``ys_solved`` on rows ``sides`` (+1/-1)."""
    side = np.asarray(sides, int)
    return fit_rows(ys_solved, side * ROW_Y_M)


def measure_hash_rows(image, K, R, t, *, band_m: float = HASH_BAND_M):
    """``[(y_solved, side)]`` for the hash rows on each side of midfield.

    Strips along the 1-yard lines between the 5-yard lines in view, top
    down; the hash tick is the only paint there. White pixels are summed
    across strips per row and the brightest row within ``band_m`` of
    midfield on each side is the hash row. A side with no paint is absent.
    """
    import cv2

    from nfl_gsplat.calibration.field_landmarks import HALF_WIDTH_M, YARD_TO_M
    from nfl_gsplat.field.yard_numbers import lines_in_view, rectify

    h_m = 2.0 * HALF_WIDTH_M
    ppm = HASH_PX_PER_M
    prof = None
    for x in lines_in_view(K, R, t, image.shape[1], image.shape[0]):
        for k in (-2, -1, 1, 2):
            strip = rectify(image, K, R, t, (x + k * YARD_TO_M, 0.0),
                            w_m=HASH_STRIP_W_M, h_m=h_m, px_per_m=ppm)
            white = (cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) > HASH_WHITE).mean(axis=1)
            prof = white if prof is None else prof + white
    if prof is None:
        return []
    y_axis = h_m / 2.0 - np.arange(len(prof)) / ppm
    out = []
    for side in (1, -1):
        sel = (np.sign(y_axis) == side) & (np.abs(y_axis) < band_m)
        if not sel.any() or prof[sel].max() <= 0:
            continue
        i = int(np.argmax(np.where(sel, prof, -1.0)))
        # A tick is 2 ft tall: a plateau in the profile, whose first maximum
        # is an edge, not the row. The centroid within half a metre is.
        win = sel & (np.abs(y_axis - y_axis[i]) < 0.5)
        w = prof[win]
        out.append((float((y_axis[win] * w).sum() / w.sum()), side))
    return out


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
