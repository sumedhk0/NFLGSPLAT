"""Absolute field position from the painted yard numbers.

WHY. Calibration from paint (calibration.calibrate_clip) fixes the camera
against the yard-line grid, but every 5-yard line looks the same, and the
field is symmetric end to end: the solved world frame sits on SOME yard line
with SOME direction along the field. The rendered field then draws generic
numerals and the play could be at either 20. The numbers on the turf resolve
it: "10 20 30 40 50 40 30 20 10", 6 ft tall, on every 10-yard line, 12 yards
in from each sideline.

HOW. Whole-frame OCR on a 12-degree broadcast lens reads foreshortened,
arrow-split numerals badly ('70', '2107' on play 1). The camera is known, so
instead each place a numeral CAN be -- every 5-yard line, both number rows
-- is warped to an upright, top-down patch through the calibration and read
there, right way up and turned 180 (the two rows face opposite sidelines).
A reading is (line x in the solved frame, numeral). The unknown is a shift
along the field, a multiple of 5 yards, that puts every reading on a line
labelled with its numeral. Each reading votes for the shifts that do; two
different numerals pin one, one numeral leaves two (either half), a lone
50 is unique.

WHAT THE NUMERALS CANNOT SAY. Whether the solved frame is turned 180 degrees
about the centre. The field is invariant under that turn -- numerals, their
arrows and both rows map onto themselves -- so no reading distinguishes the
two, and the first version of this module, which voted over (turn, shift),
tied every time. Which end zone is which is play metadata (possession,
yard line), not paint. The transform keeps a `turn` for the day that data
is wired in; this module always sets it False.

Applied ONCE to the camera track (08d), so every stage downstream is in the
rule-book frame with nothing else to change.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M,
                                                    HALF_WIDTH_M,
                                                    NUMBER_BOTTOM_Y_M,
                                                    NUMBER_TOP_Y_M,
                                                    YARD_LINE_SPACING_M,
                                                    YARD_TO_M)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

NUMERALS = ("10", "20", "30", "40", "50")
# The patch: two 4-ft digits plus their gap, the 6-ft height with a margin
# so a slightly-off calibration still holds the glyphs.
PATCH_W_M: float = 3.2
PATCH_H_M: float = 2.4
PATCH_PX_PER_M: float = 40.0
MIN_PATCH_HEIGHT_PX: float = 24.0
MIN_CONF: float = 0.45
# A reading's implied shift must land this close to the 5-yard grid.
GRID_TOL_M: float = 1.2
WIN_MARGIN: float = 1.5


@dataclass(frozen=True)
class Reading:
    x_m: float          # yard line in the solved frame
    side: int           # +1 / -1 number row
    numeral: int
    conf: float
    turned: bool        # read after a 180-degree turn of the patch
    y_m: float = float("nan")   # where along the line it was found (strip reads)


@dataclass(frozen=True)
class FieldTransform:
    turn: bool          # 180 degrees about Z (x -> -x, y -> -y); never from paint
    shift_m: float      # added along x after the turn
    votes: float
    runner_up: float
    n_readings: int

    def apply_points(self, xy):
        xy = np.asarray(xy, float)
        out = -xy if self.turn else xy.copy()
        out[..., 0] += self.shift_m
        return out


def number_row_y() -> float:
    return 0.5 * (NUMBER_BOTTOM_Y_M + NUMBER_TOP_Y_M)


def yard_line_xs():
    """x of every 5-yard line between the goal lines, solved-frame units
    assumed equal to metres (calibration is metric)."""
    n = int(round(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M))
    return -GOAL_LINE_X_M + YARD_LINE_SPACING_M * np.arange(1, n)


def numeral_at(x_abs_m: float) -> int | None:
    """The numeral painted at absolute field x, if any."""
    yards = int(round((GOAL_LINE_X_M - abs(x_abs_m)) / YARD_TO_M))
    return yards if yards % 10 == 0 and 0 < yards <= 50 else None


def ground_homography(K, R, t):
    """3x3 mapping ``(x, y, 1)`` on the turf to homogeneous pixels."""
    P = np.asarray(K, float) @ np.column_stack([np.asarray(R, float)[:, :2],
                                                np.asarray(t, float).reshape(3)])
    return P


def patch_to_world(centre_xy, w_m: float, h_m: float, px_per_m: float):
    """Affine ``(u, v, 1) -> (x, y, 1)``: patch pixel to turf, +y up in the patch."""
    cx, cy = centre_xy
    return np.array([[1.0 / px_per_m, 0.0, cx - w_m / 2.0],
                     [0.0, -1.0 / px_per_m, cy + h_m / 2.0],
                     [0.0, 0.0, 1.0]])


def project_corners(K, R, t, centre_xy, w_m: float, h_m: float):
    """Pixel corners of a turf rectangle, ``[4, 2]`` or None if behind the camera."""
    H = ground_homography(K, R, t)
    cx, cy = centre_xy
    pts = np.array([[cx - w_m / 2, cy - h_m / 2, 1.0], [cx + w_m / 2, cy - h_m / 2, 1.0],
                    [cx + w_m / 2, cy + h_m / 2, 1.0], [cx - w_m / 2, cy + h_m / 2, 1.0]])
    q = pts @ H.T
    if np.any(q[:, 2] <= 1e-9):
        return None
    return q[:, :2] / q[:, 2:3]


def rectify(image, K, R, t, centre_xy, *, w_m: float = PATCH_W_M, h_m: float = PATCH_H_M,
            px_per_m: float = PATCH_PX_PER_M):
    """Top-down patch of the turf rectangle, ``[h_px, w_px, C]``."""
    import cv2

    A = patch_to_world(centre_xy, w_m, h_m, px_per_m)
    M = ground_homography(K, R, t) @ A                  # patch pixel -> image pixel
    size = (int(round(w_m * px_per_m)), int(round(h_m * px_per_m)))
    return cv2.warpPerspective(image, M, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def candidate_patches(K, R, t, width: int, height: int, *, margin_px: int = 4):
    """``[(x_m, side)]`` of every numeral slot fully in view and tall enough."""
    out = []
    y_row = number_row_y()
    for x in yard_line_xs():
        for side in (1, -1):
            corners = project_corners(K, R, t, (x, side * y_row), PATCH_W_M, PATCH_H_M)
            if corners is None:
                continue
            inside = ((corners[:, 0] >= margin_px) & (corners[:, 0] < width - margin_px)
                      & (corners[:, 1] >= margin_px) & (corners[:, 1] < height - margin_px))
            if not inside.all():
                continue
            tall = corners[:, 1].max() - corners[:, 1].min()
            if tall < MIN_PATCH_HEIGHT_PX:
                continue
            out.append((float(x), side))
    return out


def read_patch(reader, patch, *, min_conf: float = MIN_CONF):
    """``(numeral, conf, turned)`` for the better of the patch and its 180 turn, or None."""
    import cv2

    best = None
    for turned, img in ((False, patch), (True, cv2.rotate(patch, cv2.ROTATE_180))):
        for _box, text, conf in reader.readtext(img, allowlist="0123456789", min_size=8,
                                                text_threshold=0.5, low_text=0.3):
            if text in NUMERALS and conf >= min_conf and (best is None or conf > best[1]):
                best = (int(text), float(conf), turned)
    return best


def read_numbers(image, K, R, t, reader, *, min_conf: float = MIN_CONF) -> list[Reading]:
    """Every numeral read off one frame at the rule-book row positions.

    Only right when the calibration's cross-field scale is; on play 1 it
    was not (the far row projected 250 px above the numerals), so the
    strips of :func:`read_line_strips` are what 08d uses. Kept as the check
    that a calibration puts the rows where the rule book does.
    """
    h, w = image.shape[:2]
    out = []
    for x, side in candidate_patches(K, R, t, w, h):
        patch = rectify(image, K, R, t, (x, side * number_row_y()))
        got = read_patch(reader, patch, min_conf=min_conf)
        if got is not None:
            out.append(Reading(x, side, got[0], got[1], got[2], side * number_row_y()))
    return out


STRIP_MARGIN_M: float = 4.0
STRIP_W_M: float = 4.5
# The calibration's cross-field scale is unknown (paint has no scale), so
# glyphs in a top-down strip may be squashed; each stretch is tried.
Y_STRETCHES = (1.0, 1.5, 2.0, 2.5)
TENS = "12345"
MIN_DIGIT_CONF: float = 0.6


def lines_in_view(K, R, t, width: int, height: int):
    """Yard lines whose midfield-row point projects inside the frame."""
    H = ground_homography(K, R, t)
    out = []
    for x in yard_line_xs():
        q = H @ np.array([x, 0.0, 1.0])
        if q[2] <= 1e-9:
            continue
        u, v = q[:2] / q[2]
        if 0 <= u < width and 0 <= v < height:
            out.append(float(x))
    return out


def tens_digit(text: str):
    """The numeral a box stands for, or None.

    The yard line runs between a numeral's two digits and the reader returns
    them as separate boxes, with the direction arrow beside them read as a
    '7' or '1'. The tens digit alone names the numeral (every numeral is
    d0, d in 1..5); a '0' says nothing and an arrow must not be a '1', so a
    lone '1' counts only when nothing better is read.
    """
    if text in NUMERALS:
        return int(text)
    if len(text) == 1 and text in TENS:
        return 10 * int(text)
    return None


def read_line_strips(image, K, R, t, reader, *, min_conf: float = MIN_DIGIT_CONF,
                     px_per_m: float = PATCH_PX_PER_M) -> list[Reading]:
    """Numerals found ANYWHERE along each yard line in view.

    One top-down strip per line, sideline to sideline (plus a margin), read
    upright and turned at several vertical stretches; a hit's row in the
    strip gives the y it sits at, so the calibration's cross-field error is
    measured rather than assumed away. ``side`` is the sign of that y.

    A whole row faces one sideline, so its orientation is decided once, by
    the summed score of every hit in that row, and each line then keeps its
    best hit in that orientation. Deciding per line let an upside-down "30"
    read as an upright "20" at 0.9 and win over the turned '3' at 1.0.
    """
    import cv2

    h_m = 2.0 * (HALF_WIDTH_M + STRIP_MARGIN_M)
    hits: list[tuple] = []                       # (x, side, turned, score, numeral, conf, y)
    for x in lines_in_view(K, R, t, image.shape[1], image.shape[0]):
        strip = rectify(image, K, R, t, (x, 0.0), w_m=STRIP_W_M, h_m=h_m, px_per_m=px_per_m)
        for stretch in Y_STRETCHES:
            st = cv2.resize(strip, None, fx=1.0, fy=stretch, interpolation=cv2.INTER_CUBIC)
            for turned, img in ((False, st), (True, cv2.rotate(st, cv2.ROTATE_180))):
                for box, text, conf in reader.readtext(img, allowlist="0123456789", min_size=8,
                                                       text_threshold=0.4, low_text=0.3):
                    numeral = tens_digit(text)
                    if numeral is None or conf < min_conf:
                        continue
                    v = float(np.mean([pt[1] for pt in box]))
                    if turned:
                        v = img.shape[0] - v
                    y = h_m / 2.0 - v / (px_per_m * stretch)
                    score = (float(conf) + (0.5 if text in NUMERALS else 0.0)
                             - (0.5 if text == "1" else 0.0))
                    hits.append((x, int(np.sign(y)), turned, score, numeral, float(conf), y))
    out = []
    for side in (1, -1):
        row = [h for h in hits if h[1] == side]
        if not row:
            continue
        facing = max((False, True), key=lambda tr: sum(h[3] for h in row if h[2] == tr))
        best: dict[float, tuple] = {}
        for h in row:
            if h[2] == facing and (h[0] not in best or h[3] > best[h[0]][3]):
                best[h[0]] = h
        for x, _side, turned, _score, numeral, conf, y in best.values():
            out.append(Reading(x, side, numeral, conf, turned, y))
    return out


def _snap(s: float) -> float | None:
    k = round(s / YARD_LINE_SPACING_M)
    snapped = k * YARD_LINE_SPACING_M
    return snapped if abs(s - snapped) <= GRID_TOL_M else None


def solve_transform(readings: list[Reading], *, win_margin: float = WIN_MARGIN):
    """The shift most readings agree on, or None when ambiguous.

    Ambiguous means the winner has no clear margin over the runner-up --
    which is what one numeral alone gives (its line on either half).
    """
    votes: Counter = Counter()
    for r in readings:
        for sign in (-1.0, 1.0):
            x_abs = sign * (50 - r.numeral) * YARD_TO_M
            if r.numeral == 50 and sign > 0:
                continue                            # midfield is one line
            s = _snap(x_abs - r.x_m)
            if s is not None:
                votes[s] += r.conf
    if not votes:
        return None
    ranked = votes.most_common(2)
    s, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if second > 0 and top < win_margin * second:
        _LOG.info("yard numbers: ambiguous, %.2f vs %.2f votes", top, second)
        return None
    return FieldTransform(False, float(s), float(top), float(second), len(readings))


def transform_camera(R, t, tf: FieldTransform):
    """Camera ``(R', t')`` seeing the same pixels in the transformed frame.

    Points move by ``X' = Rz X + sv``; ``x_cam = R X + t = R Rz (X' - sv) + t``.
    """
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    Rz = np.diag([-1.0, -1.0, 1.0]) if tf.turn else np.eye(3)
    sv = np.array([tf.shift_m, 0.0, 0.0])
    R2 = R @ Rz
    return R2, t - R2 @ sv


def transform_track(track, tf: FieldTransform):
    """A CameraTrack with every frame's camera moved into the rule-book frame."""
    from nfl_gsplat.calibration.cameras_io import CameraTrack

    R = np.empty_like(track.R)
    t = np.empty_like(track.t)
    for i in range(len(track.R)):
        R[i], t[i] = transform_camera(track.R[i], track.t[i], tf)
    return CameraTrack(K=track.K.copy(), R=R, t=t, conf=track.conf.copy(),
                       width=track.width, height=track.height)
