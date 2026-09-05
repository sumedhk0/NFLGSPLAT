"""Calibrate from the painted field alone -- no tracking, no player identity.

This is the production path. Everything in ``from_helmets`` needs per-frame
tracking to fit a camera, which exists only for the Helmet Assignment set;
arbitrary game footage has nothing but the picture. What the picture does have
is paint at surveyed positions, and that is enough to recover a camera.

WHAT THE PAINT DETERMINES, AND WHAT IT CANNOT. The long lines -- the sidelines
and the two hash rows -- sit at known Y, so they fix world Y outright. The yard
lines fix the X DIRECTION and the X SCALE, but not the X ORIGIN: every 5-yard
line looks like every other, so the recovered camera is correct only up to a
whole-yard shift along the field. That shift does not matter for reconstruction
-- the play merely sits at the wrong yard number -- and it is a useful check
rather than a defect, because a correct solve returns a shift very close to a
whole number of 5-yard steps. Measured on 57583/82: -6.06 steps.

FIT LINES AS LINES. Intersecting the detections to manufacture points is worse
than useless here: a yard line meets the far sideline at a shallow angle, so
the crossing lands far outside the image and one pixel of line noise becomes
tens of pixels of point error. A homography carries lines directly, l' ~ H^-T L,
so ``homography_from_lines`` fits what was actually measured. Switching to this
took the per-frame focal from an 8.7x spread to an IQR of 7%.

THE LADDER'S GAP IS AMBIGUOUS and must be resolved, not assumed. Weak
perspective makes evenly spaced yard lines fit any constant index step, so the
ladder cannot tell 5 yards from 15 by fit quality (the repo's fit_yard_ladder
returns gaps of 3 on this footage). A wrong gap stretches world X against a
world Y the long lines have already fixed, which no square-pixel camera can
express, so the gap is chosen by which one yields a consistent camera.
Measured: gap 1 places players 1.73 m from truth, gaps 2 and 3 give 8.17 m and
12.33 m.

HOW GOOD IS IT. ``cameras_from_paint_pooled`` shares ONE camera centre across
the play and REFUSES any camera it cannot justify. Over 59 sideline views of
the helmet set:

    no gates              32 accepted   4.44 m median
    grid-consistency      36 accepted   4.11 m
    + rms and mount       12 accepted   1.29 m     <- what it does now

against 0.35 m for a camera fitted from tracking. So it accepts about a fifth
of views and places players a bit over a metre out on those. That is the honest
state of the tracking-free path: usable as an initialiser, roughly four times
looser than having tracking, and -- importantly -- it now knows which views it
should not be trusted on.

Pooling also repairs catastrophes rather than merely polishing good cases: one
play went from 82.35 m to 0.93 m, because a shared centre lets the good frames
outvote the bad instead of every frame keeping its own mistake.

Free evidence that the LABELLING is right even when the pose is loose: the
recovered X shift lands within 0.15 of a whole 5-yard step on 20 of 35 views.
Nothing forces that, so a near-integer shift means the yard lines were
identified correctly and the error is in the fit, not the assignment.

THE ENDZONE VIEW IS NOT SOLVED. Its residuals run ~199 px against the sideline's
13 px and most views fail outright, usually with too few frames yielding a
focal. That camera looks ALONG the field, so its yard lines crowd toward the
vanishing point and little of the field's width is in frame.

THE SHARED-CENTRE SOLVE NEEDED TWO THINGS THAT WERE MISSING AT FIRST. An early
attempt failed on every view: the multi-start refit found no frames consistent
with any candidate camera. The causes were starvation, not the method -- twelve
frames per play where the solve needs ten consistent ones, and no seed, so it
searched a plausibility grid whose points are tens of metres apart. Feeding it
thirty frames and SEEDING it with the median of the per-frame paint centres
turns it from never working into the best result here.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    _solve_focal,
    homography_to_rt,
)
from nfl_gsplat.calibration.field_landmarks import (
    HALF_WIDTH_M,
    HASH_OFFSET_M,
    YARD_LINE_SPACING_M,
)
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# The field's long lines, from the far side of the field to the near side. Order
# matters: an assignment of detected rows to these must preserve it, because a
# camera on one side of the field sees them in this order down the image.
CROSS_Y_M: tuple[float, ...] = (+HALF_WIDTH_M, +HASH_OFFSET_M,
                                -HASH_OFFSET_M, -HALF_WIDTH_M)

# Real yard lines cross most of the frame. Short segments come off players and
# the painted numbers, and one of them in the ladder shifts every index after
# it.
#
# As a FRACTION of image height, not a pixel count. The original 300 px was set
# on 1280x720 broadcast clips; production All-22 is 1920x1080, where the same
# number silently means something else. 0.42 reproduces the old behaviour at
# 720 and scales.
MIN_YARD_LINE_FRAC: float = 0.42
MIN_YARD_LINE_PX: float = 300.0        # kept for callers that pass pixels

# Two rows within this many pixels at the image centre are the same line
# detected twice.
ROW_MERGE_PX: float = 25.0

GAP_CANDIDATES: tuple[int, ...] = (1, 2, 3)

# A frame whose best assignment cannot land this fraction of hash marks on the
# yard grid has been labelled wrong, and its homography is not worth keeping.
#
# The check earns its place on production All-22, where the detector also
# returns stadium structure as "long lines": with four rows there is only ONE
# order-preserving assignment, so nothing can be chosen and a wrong labelling
# goes through unopposed. It scored 9.8% here while a correct labelling scores
# near 100%, and the cameras that came out sat 300-500 m from the field.
MIN_GRID_CONSISTENCY: float = 0.60

# Beyond this many lines the exponential ladder search is not worth its cost.
_MAX_LADDER_LINES: int = 8
_LADDER_CACHE: dict = {}


def seg_line(seg) -> np.ndarray:
    """Normalised homogeneous image line through a detected segment."""
    ln = np.cross([seg.p0[0], seg.p0[1], 1.0], [seg.p1[0], seg.p1[1], 1.0])
    return ln / max(float(np.hypot(ln[0], ln[1])), 1e-9)


def seg_length(seg) -> float:
    return float(np.hypot(seg.p1[0] - seg.p0[0], seg.p1[1] - seg.p0[1]))


def line_at_x(line, x: float):
    """Row where a line crosses column ``x``, or None if it is vertical."""
    if abs(line[1]) < 1e-9:
        return None
    return float(-(line[0] * x + line[2]) / line[1])


def fit_rows(points, *, tol_px: float = 5.0, min_marks: int = 10,
             max_rows: int = 3, max_slope: float = 0.35, seed: int = 0):
    """Rows through hash-mark blobs, by sequential RANSAC.

    The blob detector also fires on crowd and stadium structure -- measured, 167
    blobs of which 95 were on the field -- so a row has to be found by consensus
    rather than by averaging whatever was returned.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    rows: list[tuple[np.ndarray, int]] = []
    used = np.zeros(len(pts), bool)
    rng = np.random.default_rng(seed)
    for _ in range(max_rows):
        idx = np.flatnonzero(~used)
        if len(idx) < min_marks:
            break
        best, best_n = None, 0
        for _ in range(400):
            a, b = rng.choice(idx, 2, replace=False)
            p, q = pts[a], pts[b]
            if abs(q[0] - p[0]) < 40:
                continue
            if abs((q[1] - p[1]) / (q[0] - p[0])) > max_slope:
                continue
            ln = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
            ln = ln / max(float(np.hypot(ln[0], ln[1])), 1e-9)
            n = int((np.abs(ln[0] * pts[idx, 0] + ln[1] * pts[idx, 1]
                            + ln[2]) < tol_px).sum())
            if n > best_n:
                best, best_n = ln, n
        if best is None or best_n < min_marks:
            break
        inl = (np.abs(best[0] * pts[:, 0] + best[1] * pts[:, 1]
                      + best[2]) < tol_px) & (~used)
        sel = pts[inl]
        centre = sel.mean(0)
        _u, _s, vt = np.linalg.svd(sel - centre)
        nrm = np.array([-vt[0][1], vt[0][0]])
        ln = np.array([nrm[0], nrm[1], -float(nrm @ centre)])
        rows.append((ln / float(np.hypot(ln[0], ln[1])), int(inl.sum()),
                     pts[inl].copy()))
        used |= inl
    return rows


def homography_from_lines(world_lines, image_lines):
    """Ground(z=0)->image homography from LINE correspondences.

    Lines transform as ``l' ~ H^-T L``, so the ordinary DLT is run on the line
    vectors to get ``G = H^-T`` and the homography recovered as ``G^-T``.
    """
    rows = []
    for world, img in zip(world_lines, image_lines):
        world = np.asarray(world, float)
        img = np.asarray(img, float)
        img = img / max(float(np.linalg.norm(img)), 1e-12)
        zero = np.zeros(3)
        rows.append(np.r_[zero, -img[2] * world, img[1] * world])
        rows.append(np.r_[img[2] * world, zero, -img[0] * world])
    A = np.asarray(rows)
    if len(A) < 8:
        return None
    _u, _s, vt = np.linalg.svd(A)
    G = vt[-1].reshape(3, 3)
    if abs(np.linalg.det(G)) < 1e-12:
        return None
    return np.linalg.inv(G).T


def long_lines(features, width: int, height: int):
    """Distinct near-horizontal field lines: hash rows plus any sideline."""
    blobs = (np.asarray(features.hashes, float).reshape(-1, 2)
             if len(features.hashes) else np.zeros((0, 2)))
    # Score grid consistency only on blobs that a ROW claimed. The detector also
    # fires on crowd and stadium structure -- measured, 95 of 167 blobs were on
    # the field -- and counting the rest in the denominator caps the metric near
    # 0.4 even when the labelling is perfect, which put the accept threshold
    # right at the achievable ceiling and threw away most usable frames.
    candidates = list(fit_rows(blobs))
    candidates += [(seg_line(s), 0, np.zeros((0, 2))) for s in features.sidelines]
    out = []
    for line, n, marks in sorted(candidates, key=lambda z: -z[1]):
        v = line_at_x(line, width / 2.0)
        if v is None or not (-200 < v < height + 200):
            continue
        if all(abs(v - line_at_x(o[0], width / 2.0)) > ROW_MERGE_PX for o in out):
            out.append((line, n, v, marks))
    out.sort(key=lambda z: z[2])          # top of the image first
    return out


def yard_ladder(features, *, min_len_px: float | None = None,
                image_height: int | None = None, image_width: int | None = None):
    """Detected yard lines, filtered and ordered across the image.

    The length bar is a fraction of the SHORT side; see detect_lines for why
    height alone silently blinds a frame that has been turned a quarter turn.
    """
    if min_len_px is None:
        ref = (min(image_height, image_width)
               if image_height and image_width else image_height)
        min_len_px = (MIN_YARD_LINE_FRAC * ref if ref else MIN_YARD_LINE_PX)
    kept = [s for s in features.yard_lines if seg_length(s) >= min_len_px]
    return sorted(kept, key=lambda s: (s.p0[0] + s.p1[0]) / 2.0)


# Swapping image u and v turns a vertical line family into a horizontal one.
# The endzone camera looks along the field, so ITS yard lines run across the
# image while the sidelines and hash rows run away from it -- exactly the
# opposite of the sideline camera, and the reason field_detect's "yard_lines"
# are the field's LONG lines in that view. Rather than duplicate every routine
# for the other orientation, the frame is transposed and the result mapped back.
_SWAP_UV = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class _Transposed:
    """A features object with u and v swapped, cheap and read-only."""

    def __init__(self, features):
        self.yard_lines = [_SwapSeg(s) for s in features.sidelines]
        self.sidelines = [_SwapSeg(s) for s in features.yard_lines]
        self.hashes = [(float(p[1]), float(p[0])) for p in features.hashes]


class _SwapSeg:
    __slots__ = ("p0", "p1")

    def __init__(self, seg):
        self.p0 = (seg.p0[1], seg.p0[0])
        self.p1 = (seg.p1[1], seg.p1[0])


def spacing_ratio(ladder) -> float:
    """Largest over smallest adjacent spacing across the image centre.

    Near 1 means the lines are evenly spaced, which is what a nearly affine
    view produces and the condition under which the index step is NOT
    identifiable. Well above 1 means real perspective, where it is.
    """
    mids = sorted((s.p0[0] + s.p1[0]) / 2.0 for s in ladder)
    gaps = np.diff(mids)
    gaps = gaps[gaps > 1e-6]
    if len(gaps) < 2:
        return 1.0
    return float(gaps.max() / max(gaps.min(), 1e-9))


def ladder_indices(ladder, *, min_ratio: float = 1.5):
    """Yard-grid index per detected line, allowing for MISSING lines.

    Numbering detections 0,1,2,... assumes every yard line between the first
    and last was found, which fails wherever one is hidden under players. The
    repo's ladder fit handles that -- but ONLY where perspective identifies the
    step, and it does not say when it cannot. Under weak perspective evenly
    spaced lines fit any constant step equally well and it returns a confident
    wrong answer: on the sideline view it gave 12,10,9,6,3,0 for five evenly
    spaced lines, and using those took placement from 3.24 m to 11.79 m.

    So the choice is made on the evidence that decides it. Even spacing means
    the step is unidentifiable AND that nothing is missing, so consecutive
    numbering is both forced and correct. Uneven spacing means real perspective,
    where the ladder fit is informative -- measured, an endzone frame's rows sat
    at 116/208/306/532/660 px, a 2.5x spread whose 226 px gap really is two
    spacings.
    """
    if len(ladder) < 4:
        return list(range(len(ladder)))
    if spacing_ratio(ladder) < min_ratio:
        return list(range(len(ladder)))
    key = tuple(round((s.p0[0] + s.p1[0]) / 2.0, 2) for s in ladder)
    hit = _LADDER_CACHE.get(key)
    if hit is not None:
        return hit
    if len(ladder) > _MAX_LADDER_LINES:
        return list(range(len(ladder)))
    try:
        from nfl_gsplat.calibration.field_model_fit import fit_yard_ladder

        idx = fit_yard_ladder(list(ladder))
    except Exception:
        _LADDER_CACHE[key] = list(range(len(ladder)))
        return _LADDER_CACHE[key]
    if len(idx) != len(ladder) or len(set(idx)) != len(idx):
        _LADDER_CACHE[key] = list(range(len(ladder)))
        return _LADDER_CACHE[key]
    lo = min(idx)
    _LADDER_CACHE[key] = [int(i - lo) for i in idx]
    return _LADDER_CACHE[key]


def frame_homography(features, width: int, height: int, *, gap: int = 1,
                     transposed: bool = False, image=None,
                     use_hash_points: bool = True, fov_band=None):
    """``(H, world_y, residual_px)`` for one frame, X ORIGIN ARBITRARY.

    The assignment of detected long lines to the field's known Y values is
    searched over, order-preserving, and scored by how well the resulting
    homography reproduces every line.
    """
    from itertools import combinations

    if transposed:
        features = _Transposed(features)
        width, height = height, width
        if image is not None:
            image = np.transpose(image, (1, 0, 2) if image.ndim == 3 else (1, 0))

    gray = None
    if image is not None:
        import cv2

        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else np.asarray(image))

    rows = long_lines(features, width, height)
    ladder = yard_ladder(features, image_height=height,
                         image_width=width)
    if len(rows) < 2 or len(ladder) < 3:
        return None

    steps = ladder_indices(ladder)
    blobs = (np.asarray(features.hashes, float).reshape(-1, 2)
             if len(features.hashes) else np.zeros((0, 2)))
    # Score grid consistency only on blobs that a ROW claimed. The detector also
    # fires on crowd and stadium structure -- measured, 95 of 167 blobs were on
    # the field -- and counting the rest in the denominator caps the metric near
    # 0.4 even when the labelling is perfect, which put the accept threshold
    # right at the achievable ceiling and threw away most usable frames.

    ladder_lines = [seg_line(seg) for seg in ladder]
    if gray is not None:
        ladder_lines = [refine_line(gray, ln) for ln in ladder_lines]

    best = None
    # SUBSETS of the detected rows, not just all of them. The detector picks up
    # stadium edges as long lines, and with every row forced into the
    # assignment there is exactly one candidate and therefore no choice to make.
    # Allowing the weakest rows to be dropped restores the competition that
    # grid consistency and camera plausibility are there to judge.
    # The full set, then single drops only. Every subset was tried at first and
    # it multiplied the per-frame cost by about ten for no measured gain: the
    # detector returns at most one or two spurious rows, so dropping more than
    # one at a time is discarding real evidence rather than noise.
    n_rows = min(len(rows), len(CROSS_Y_M))
    row_choices = [[rows[i] for i in range(n_rows)]]
    if n_rows > 2:
        for drop in range(n_rows):
            row_choices.append([rows[i] for i in range(n_rows) if i != drop])
    for chosen_rows in row_choices:
      for pick in combinations(range(len(CROSS_Y_M)), len(chosen_rows)):
        world_y = [CROSS_Y_M[i] for i in pick]
        world_lines, image_lines = [], []
        for k, ln in zip(steps, ladder_lines):
            world_lines.append([1.0, 0.0, -k * gap * YARD_LINE_SPACING_M])
            image_lines.append(ln)
        for row, y in zip(chosen_rows, world_y):
            ln = row[0]
            if gray is not None:
                ln = refine_line(gray, ln)
            world_lines.append([0.0, 1.0, -y])
            image_lines.append(ln)
        H = homography_from_lines(world_lines, image_lines)
        if H is None:
            continue

        # Second pass: the line fit is only needed to say WHICH yard and which
        # row each hash mark is, after which the marks themselves are far
        # better constraints -- tens of points instead of a handful of lines,
        # and a 1-yard ruler laid along the field.
        n_points = 0
        if use_hash_points and len(blobs):
            import cv2

            wpts, ipts = hash_point_correspondences(H, blobs, world_y)
            if len(wpts) >= MIN_HASH_POINTS:
                H2, mask = cv2.findHomography(wpts, ipts, cv2.RANSAC, 6.0)
                if H2 is not None and int(mask.sum()) >= MIN_HASH_POINTS:
                    H = H2
                    n_points = int(mask.sum())
        errs = []
        Hinv_T = np.linalg.inv(H).T
        for world, img in zip(world_lines, image_lines):
            pred = Hinv_T @ np.asarray(world, float)
            norm = float(np.hypot(pred[0], pred[1]))
            if norm < 1e-12:
                continue
            pred = pred / norm
            if float(pred @ np.asarray(img)) < 0:
                pred = -pred
            a = line_at_x(pred, width / 2.0)
            b = line_at_x(np.asarray(img), width / 2.0)
            if a is not None and b is not None:
                errs.append(abs(a - b))
        if not errs:
            continue
        res = float(np.median(errs))
        # Prefer the assignment whose hash marks land on the yard grid; fall
        # back to the line residual only to break ties among equally consistent
        # candidates, since it cannot tell a wrong assignment from a right one.
        row_marks = [m for _l, _n, _v, m in chosen_rows if len(m)]
        scored = (np.vstack(row_marks) if row_marks else np.zeros((0, 2)))
        grid = grid_consistency(H, scored, world_y) if len(scored) else 0.0
        # Physical possibility first -- an assignment implying a camera that
        # cannot exist is wrong however well it fits. Then grid consistency for
        # the X scale, focal agreement for Y, and the line residual for ties.
        possible = assignment_is_possible(H, width, height, fov_band=fov_band)
        # Physical possibility, then grid consistency, then MORE ROWS -- a
        # labelling that uses four field lines is better constrained than one
        # using two, so among equally consistent candidates the fuller one
        # wins. Focal agreement and the line residual break what is left.
        key = (not possible, -round(grid, 2), -len(chosen_rows),
               focal_disagreement(H, width, height), res)
        if best is None or key < best[4]:
            best = (H, world_y, res, n_points, key, grid, len(scored))
    if best is None:
        return None
    # Refuse a frame whose best labelling still cannot put hash marks on the
    # yard grid: a homography fitted to the wrong lines is worse than none.
    if best[6] >= MIN_HASH_POINTS and best[5] < MIN_GRID_CONSISTENCY:
        return None
    if transposed:
        best = (_SWAP_UV @ best[0], *best[1:])
    return best[0], best[1], best[2], best[3], best[5]


def best_frame_homography(features, width: int, height: int, *, gap: int = 1,
                          image=None, use_hash_points: bool = True, fov_band=None):
    """``(H, world_y, residual, transposed)`` -- tries both view orientations."""
    out = []
    for flag in (False, True):
        got = frame_homography(features, width, height, gap=gap, fov_band=fov_band,
                               transposed=flag, image=image,
                               use_hash_points=use_hash_points)
        if got is not None:
            # Same rule as within a frame: grid consistency first, residual
            # only as a tie-break.
            out.append(((-got[4], got[2]), got[0], got[1], flag, got[4]))
    if not out:
        return None
    out.sort(key=lambda z: z[0])
    _key, H, world_y, flag, grid = out[0]
    return H, world_y, _key[1], flag, grid


def cameras_from_paint(features_by_frame, width: int, height: int, *,
                       gap: int = 1, transposed=None, images=None,
                       use_hash_points: bool = True):
    """``({frame: (K, R, t)}, focal, residual)`` in the GROUND frame.

    The focal is pooled over frames and then held fixed, for the same reason as
    in ``from_helmets``: one frame constrains it far more weakly than the play
    as a whole.
    """
    per_frame, focals, residuals = {}, [], []
    for frame, feats in features_by_frame.items():
        image = None if images is None else images.get(frame)
        if transposed is None:
            got = best_frame_homography(feats, width, height, gap=gap,
                                        image=image,
                                        use_hash_points=use_hash_points)
        else:
            got = frame_homography(feats, width, height, gap=gap,
                                   transposed=transposed, image=image,
                                   use_hash_points=use_hash_points)
        if got is None:
            continue
        H, _world_y, res = got[0], got[1], got[2]
        per_frame[frame] = H
        residuals.append(res)
        try:
            focals.append(_solve_focal(H, width / 2.0, height / 2.0))
        except ValueError:
            continue
    if len(focals) < 5:
        raise CalibrationError(
            f"only {len(focals)} of {len(features_by_frame)} frames yielded a "
            "focal from paint. Too little of the field is visible -- check that "
            "the yard lines and at least two long lines are being detected.")
    focal = float(np.median(focals))
    K = np.array([[focal, 0.0, width / 2.0],
                  [0.0, focal, height / 2.0],
                  [0.0, 0.0, 1.0]])
    cams = {}
    for frame, H in per_frame.items():
        R, t = homography_to_rt(H, K)
        cams[frame] = (K, R, t)
    _LOG.info("paint calibration: %d frames, focal %.0f px, line residual %.1f px",
              len(cams), focal, float(np.median(residuals)))
    return cams, focal, float(np.median(residuals))


def choose_gap(features_by_frame, width: int, height: int,
               candidates=GAP_CANDIDATES):
    """Pick the yard-line index step whose camera is most self-consistent.

    Scored by the spread of the per-frame focal: a wrong gap stretches world X
    against an already-fixed world Y, and the square-pixel camera that has to
    absorb it disagrees from frame to frame.
    """
    best = None
    for gap in candidates:
        focals = []
        for feats in features_by_frame.values():
            got = best_frame_homography(feats, width, height, gap=gap)
            if got is None:
                continue
            try:
                focals.append(_solve_focal(got[0], width / 2.0, height / 2.0))
            except ValueError:
                continue
        if len(focals) < 5:
            continue
        f = np.asarray(focals)
        spread = float(np.subtract(*np.percentile(f, [75, 25])) / max(np.median(f), 1e-9))
        if best is None or spread < best[1]:
            best = (gap, spread, float(np.median(f)))
    if best is None:
        raise CalibrationError("no yard-line gap produced a usable camera.")
    return best


# Height to meet when turning a HELMET detection into a field position, with a
# camera whose world z=0 is the turf.
#
# NOT the same number as the 1.5 m turf-drop measured in 07e, and the difference
# is real rather than an inconsistency. That one is where the helmet PLANE sits
# relative to the paint, fitted to painted lines. This one is what best matches
# TRACKING, which marks a player's position on the ground while their helmet
# leans forward of it -- so it absorbs lean as well as height and comes out
# lower. Swept over 10 plays with paint cameras: 0.6 m -> 1.72, 1.0 -> 0.98,
# 1.2 -> 0.81, 1.4 -> 0.82, 1.5 -> 1.03, 2.0 -> 2.02. A clean minimum, and
# using it instead of 1.5 is worth 21%.
PLACEMENT_HELMET_HEIGHT_M: float = 1.25


def ray_to_plane(K, R, t, uv, z_plane: float):
    """Where the ray through pixel ``uv`` meets a horizontal plane, or None."""
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    centre = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    if abs(d[2]) < 1e-12:
        return None
    s = (z_plane - centre[2]) / d[2]
    if s <= 0:
        return None
    return (centre + s * d)[:2]

# --------------------------------------------------------------------------
# Hash marks as POINTS, and sub-pixel line refinement.
# --------------------------------------------------------------------------

# How close a hash blob must land to the 1-yard grid, in metres, to be believed
# as that grid point. Loose enough for a rough starting homography, tight enough
# that a blob which is really a shoe or a shadow is not snapped onto the field.
HASH_SNAP_TOL_M: float = 0.45

# Hash marks are painted one per yard along each hash row.
HASH_PITCH_M: float = 0.9144

MIN_HASH_POINTS: int = 8


# A broadcast camera's physical envelope. Used to REJECT a row assignment whose
# implied camera cannot exist, which is the only thing that separates two
# labellings that both fit the lines perfectly.
# Measured on the broadcast clips, a real game camera runs about 14-16 degrees
# horizontally, and All-22 is wider still because it exists to show all 22
# players. A 4.6 degree lens at 122 m sees ten metres of field and cannot show a
# formation, so the old 4.0 lower bound admitted cameras that were physically
# impossible for the job. 8 degrees is still generous to a long lens.
PLAUSIBLE_FOV_DEG = (8.0, 75.0)


def implied_fov_deg(H, width: int, height: int) -> float:
    """Horizontal field of view implied by a plane homography, or NaN."""
    try:
        focal = _solve_focal(np.asarray(H, float), width / 2.0, height / 2.0)
    except ValueError:
        return float("nan")
    if not np.isfinite(focal) or focal <= 1e-6:
        return float("nan")
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * focal))))


def assignment_is_possible(H, width: int, height: int, fov_band=None) -> bool:
    """Could a real camera have produced this homography of the field?

    ``fov_band`` (lo, hi) degrees narrows the plausible lens to what is KNOWN
    before the solve -- the same mount's lens on another play of the game,
    or the players' boxes. Measured on play 3 (2026-09-05): in two-row
    frames the reader labelled the far sideline's ticks and the far hash row
    as the TWO SIDELINES, stretching the field 2.3x across; the implied 41
    degree lens sat inside the generous (8, 75) band and grid consistency,
    an X instrument, still scored 0.95, so it won over the hash-pair
    labelling at 14.9 degrees. With the band from play 1 (11.7 deg x 1/1.6
    .. 1.6) the sideline-pair labelling is not possible and the right one
    wins.

    This is what separates a wrong row labelling from a right one when both fit
    the lines. Calling the two HASH ROWS the two sidelines stretches world Y by
    24.384/2.8194 = 8.65, and the solve absorbs it by moving the camera 8.65x
    further away behind a much longer lens. Measured on All-22, that produced
    cameras 300-500 m from the field at a 4-6 degree field of view -- a fit no
    broadcast mount could have made, and the only signal that says so.
    """
    fov = implied_fov_deg(H, width, height)
    if not np.isfinite(fov):
        return True          # cannot tell; do not penalise
    lo, hi = fov_band if fov_band is not None else PLAUSIBLE_FOV_DEG
    return lo <= fov <= hi


def focal_disagreement(H, width: int, height: int) -> float:
    """How badly the two focal constraints disagree under ``H``. Lower is better.

    A plane homography gives the focal twice over, from the orthogonality of
    the plane's two axes and from their equal length. Under a CORRECT world
    labelling the two agree; under a wrong one they cannot, because mislabelling
    the rows rescales world Y against world X and no square-pixel camera can
    absorb that.

    This is the complement to ``grid_consistency``, which pins the X scale via
    the 1-yard hash pitch but is blind to a pure Y rescale -- measured, a fit
    that called the hash rows the sidelines still landed every hash mark on the
    grid. Together they cover both axes.
    """
    H = np.asarray(H, float)
    T = np.array([[1.0, 0.0, -width / 2.0],
                  [0.0, 1.0, -height / 2.0],
                  [0.0, 0.0, 1.0]])
    G = T @ H
    h1, h2 = G[:, 0], G[:, 1]
    denom_o = h1[2] * h2[2]
    denom_n = h1[2] ** 2 - h2[2] ** 2
    scale2 = max(abs(h1[2]), abs(h2[2]), 1e-30) ** 2
    vals = []
    for num, den in ((-(h1[0] * h2[0] + h1[1] * h2[1]), denom_o),
                     ((h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2),
                      denom_n)):
        if abs(den) / scale2 <= 1e-6:
            continue
        v = num / den
        if v > 0:
            vals.append(np.sqrt(v))
    if len(vals) < 2:
        return np.inf
    return float(abs(vals[0] - vals[1]) / max(min(vals), 1e-9))


def grid_consistency(H, blobs, row_y_values, *, tol_m: float = HASH_SNAP_TOL_M,
                     pitch_m: float = HASH_PITCH_M):
    """What fraction of hash blobs land on the 1-yard grid under ``H``.

    This is the score that picks the row assignment and the yard-line step, and
    it works where the line residual does not. A wrong assignment -- calling the
    far sideline a hash row, or stepping the ladder by 10 yards instead of 5 --
    still fits the LINES perfectly, because a homography has enough freedom to
    put a few lines wherever it is told. What it cannot do is also land ninety
    independent hash marks on a ruler they do not belong to.

    Measured: the line residual picked assignments that placed players tens of
    metres out, on plays where the grid score rejects the same assignment.
    """
    blobs = np.asarray(blobs, float).reshape(-1, 2)
    if len(blobs) < MIN_HASH_POINTS or H is None:
        return 0.0
    world, _img = hash_point_correspondences(H, blobs, row_y_values,
                                             tol_m=tol_m, pitch_m=pitch_m)
    return float(len(world)) / float(len(blobs))


def hash_point_correspondences(H, blobs, row_y_values, *,
                               tol_m: float = HASH_SNAP_TOL_M,
                               pitch_m: float = HASH_PITCH_M):
    """``(world_xy, image_uv)`` from hash-mark blobs snapped to the yard grid.

    Rows alone give two line constraints out of ninety-odd detected marks. But
    the marks are painted ONE PER YARD, so each is a point with known field
    coordinates as soon as it is known which row and which yard it belongs to.
    A rough homography answers both: send the blob to the field, snap its Y to
    the nearest hash row and its X to the nearest yard, and keep it only if it
    landed close enough that the snap is not a guess.

    This is also what pins the X SCALE. The yard-line ladder cannot -- evenly
    spaced lines fit any constant step -- but the 1-yard pitch of the hash marks
    is a ruler laid along the field.
    """
    blobs = np.asarray(blobs, float).reshape(-1, 2)
    if len(blobs) == 0 or H is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    try:
        Hinv = np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return np.zeros((0, 2)), np.zeros((0, 2))

    q = np.c_[blobs, np.ones(len(blobs))] @ Hinv.T
    with np.errstate(invalid="ignore", divide="ignore"):
        world = q[:, :2] / q[:, 2:3]

    rows = np.asarray(sorted(row_y_values), float)
    if len(rows) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    dy = np.abs(world[:, 1][:, None] - rows[None, :])
    row_idx = np.argmin(dy, axis=1)
    snapped_y = rows[row_idx]
    snapped_x = np.round(world[:, 0] / pitch_m) * pitch_m

    good = (np.isfinite(world).all(axis=1)
            & (dy.min(axis=1) <= tol_m)
            & (np.abs(world[:, 0] - snapped_x) <= tol_m))
    return (np.column_stack([snapped_x[good], snapped_y[good]]), blobs[good])


def refine_line(gray, line, *, band_px: float = 6.0, n_samples: int = 40,
                min_support: int = 8):
    """Re-fit a detected line to the sub-pixel ridge of the paint under it.

    Hough returns a line quantised to its accumulator; the paint itself is a
    bright ridge several pixels wide whose CENTRE can be located far more
    precisely. Sampling perpendicular profiles and taking the
    intensity-weighted centroid of each moves the line onto that ridge.
    """
    gray = np.asarray(gray)
    h, w = gray.shape[:2]
    line = np.asarray(line, float)
    n = np.hypot(line[0], line[1])
    if n < 1e-9:
        return line
    line = line / n
    normal = line[:2]
    direction = np.array([-normal[1], normal[0]])
    # A point on the line, then walk along it.
    p0 = -line[2] * normal
    ts = np.linspace(-max(w, h), max(w, h), n_samples)
    offsets = np.arange(-band_px, band_px + 1.0)

    pts = []
    for t in ts:
        centre = p0 + t * direction
        samples = centre[None, :] + offsets[:, None] * normal[None, :]
        cols = np.round(samples[:, 0]).astype(int)
        rows_i = np.round(samples[:, 1]).astype(int)
        inside = (cols >= 0) & (cols < w) & (rows_i >= 0) & (rows_i < h)
        if inside.sum() < 3:
            continue
        vals = gray[rows_i[inside], cols[inside]].astype(np.float64)
        weight = vals - vals.min()
        total = weight.sum()
        if total <= 1e-6:
            continue
        off = float((offsets[inside] * weight).sum() / total)
        pts.append(centre + off * normal)
    if len(pts) < min_support:
        return line
    pts = np.asarray(pts)
    centroid = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centroid)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    out = np.array([nrm[0], nrm[1], -float(nrm @ centroid)])
    return out / max(float(np.hypot(out[0], out[1])), 1e-9)


# Reprojection gate for the shared-centre solve on PAINT, in pixels.
#
# Paint is looser than helmets -- line residuals run ~13 px on the sideline view
# and points sampled from a per-frame homography inherit that -- so the helmet
# gate of 25 px rejects everything. But loose is worse than useless: swept over
# 15 plays, a 150 px gate gave a 0.95 m median placement on the 8 plays it
# accepted, while a 400 px gate accepted 11 and scored 5.72 m, no better than
# not pooling at all. The tight gate REFUSES rather than returning a camera it
# cannot stand behind, which is the trade this codebase takes everywhere.
PAINT_AUDIT_PX: float = 150.0

# ...at this image height. The gate is a PIXEL tolerance, so the same world
# error projects to more pixels on a taller frame: 150 px on the 720p broadcast
# clips is 225 px on 1080p All-22, and using the raw number there quietly
# tightens the gate by half without anyone choosing to.
PAINT_AUDIT_REF_HEIGHT: int = 720

# A solved camera is only returned if its own residual is under this (at the
# reference height) and its mount is physically possible. Both are TRACKING
# FREE, which is the point: production has nothing to compare against, so the
# camera must be judged on its own evidence or not at all.
#
# The threshold is an operating point, measured over 37 solved views:
#
#     gate                 views   median   within 2 m
#     none                    37    4.11 m    14
#     rms<=25 + plausible     22    3.00 m     9
#     rms<=18 + plausible     13    1.30 m     8
#     rms<=14 + plausible      6    1.12 m     5
#
# 18 keeps most of the good plays while dropping most of the bad, and the
# alternative -- returning all 37 and letting the caller find out -- is the
# thing this codebase refuses to do everywhere else.
PAINT_MAX_RMS_PX: float = 18.0


def align_x_origins(homographies, width: int, height: int, *,
                    spacing_m: float = YARD_LINE_SPACING_M, search: int = 25):
    """Put every frame's homography on a COMMON world X origin.

    Each frame numbers its own detected yard lines from zero, so as the camera
    pans and a different line becomes the leftmost one, that frame's world
    slides by a whole number of 5-yard steps. Every frame is then internally
    consistent -- its hash marks still land on its own grid -- while no single
    rigid camera can satisfy them all, which is exactly what was seen on
    All-22: grid consistency 0.97 per frame alongside a 170-240 px shared-centre
    residual that shortening the window did nothing to fix.

    The shift is unobservable within one frame and obvious across many: only one
    choice per frame puts the camera where the other frames put it. So each
    frame's offset is chosen to bring its implied centre to the median, which
    needs no ground truth and no tracking.

    OFF BY DEFAULT, because it helps only where the problem exists. On All-22 it
    fires on 12-29 frames per clip and cut one clip's residual from 158 to 97
    px. On the helmet clips the camera barely pans, the origins already agree,
    and aligning to a median of NOISY per-frame decompositions moves frames that
    were right: pooled views fell 12 to 4 and paired error went 0.96 -> 1.43 m.

    That is the third time in this work that a correction has done harm where
    the thing it corrects was already fine -- see also the team prior in
    identity and per-game seeding here. The rule is the same each time: the
    correction must be better than what it is correcting.
    """
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt

    def centre_of(H):
        try:
            _K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)

    frames = list(homographies)
    centres = {f: centre_of(homographies[f]) for f in frames}
    known = [c for c in centres.values() if c is not None and np.isfinite(c).all()]
    if len(known) < 3:
        return dict(homographies), 0
    target = np.median(np.stack(known), axis=0)

    out, moved = {}, 0
    for f in frames:
        H = homographies[f]
        best, best_d, best_k = H, np.inf, 0
        for k in range(-search, search + 1):
            T = np.array([[1.0, 0.0, k * spacing_m],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
            cand = np.asarray(H, float) @ T
            c = centre_of(cand)
            if c is None or not np.isfinite(c).all():
                continue
            d = float(np.linalg.norm(c - target))
            if d < best_d:
                best, best_d, best_k = cand, d, k
        out[f] = best
        moved += int(best_k != 0)
    _LOG.info("x-origin alignment: %d/%d frames shifted", moved, len(frames))
    return out, moved


def _enforce_quality(quality, centre, max_rms_px, scale):
    """Refuse a camera that cannot be justified on its own evidence."""
    if not quality["plausible_mount"]:
        raise CalibrationError(
            "paint solve produced a camera no broadcast mount could be: centre "
            f"{np.round(centre, 1)}, {quality['fov_deg']:.1f} deg field of "
            "view. Refusing it rather than returning it.")
    limit = max_rms_px if max_rms_px is not None else PAINT_MAX_RMS_PX * scale
    if not (quality["rms_px"] <= limit):
        raise CalibrationError(
            f"paint solve residual {quality['rms_px']:.0f} px exceeds "
            f"{limit:.0f} px; the camera is not trustworthy. See "
            "PAINT_MAX_RMS_PX for the measured operating curve.")


def _solve_pooled(raw, world, width, height, seed_centre, audit_px, fixed_centre=None):
    """The shared-centre solve, given one homography per frame."""
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center

    frame_data, centres = {}, []
    for frame, H in raw.items():
        q = np.c_[world[:, :2], np.ones(len(world))] @ np.asarray(H, float).T
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        keep = (np.isfinite(uv).all(axis=1)
                & (uv[:, 0] > -width) & (uv[:, 0] < 2 * width)
                & (uv[:, 1] > -height) & (uv[:, 1] < 2 * height))
        if keep.sum() < 8:
            continue
        frame_data[int(frame)] = (world[keep], uv[keep])
        try:
            _K, R, t = homography_to_krt(H, width=width, height=height)
            centres.append(-np.asarray(R).T @ np.asarray(t).reshape(3))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if len(frame_data) < 10:
        raise CalibrationError(
            f"only {len(frame_data)} frames gave a paint homography; "
            "need 10 for a shared-centre solve.")

    seeds = []
    if seed_centre is not None:
        # A camera from another play of the SAME GAME. Measured to make paint
        # slightly WORSE (5.23 m vs 4.40 m) because a paint camera carries
        # metres of its own error, while the same trick on tracking-fitted
        # cameras rescued plays. Pass one only if it is good.
        seeds.append(np.asarray(seed_centre, float).reshape(3))
    if centres:
        seeds.append(np.median(np.stack(centres), axis=0))
    seeds = seeds or None

    results, mirrored = solve_fixed_center(
        {}, (width, height), init_results=[None] * (max(frame_data) + 1),
        _frame_data_override=frame_data, view_deg=0, audit_drop_px=audit_px,
        seed_centers=seeds, fixed_center=fixed_centre)

    cams = {}
    for frame in frame_data:
        r = results[frame]
        if r is None:
            continue
        K = np.asarray(r.intrinsics.K() if callable(r.intrinsics.K)
                       else r.intrinsics.K, float)
        cams[frame] = (K, np.asarray(r.pose.R, float),
                       np.asarray(r.pose.t, float).reshape(3))
    if not cams:
        raise CalibrationError("shared-centre paint solve kept no frames.")
    centre = np.median([
        np.asarray(results[f].pose.center_world()
                   if callable(results[f].pose.center_world)
                   else results[f].pose.center_world, float)
        for f in cams], axis=0)
    focal = float(np.median([c[0][0, 0] for c in cams.values()]))
    quality = solve_quality(results, cams, frame_data, centre, focal, width)
    return cams, focal, centre, mirrored, quality


def cameras_from_paint_pooled(features_by_frame, width: int, height: int, *,
                              gap: int = 1, images=None,
                              audit_px: float | None = None,
                              grid: int = 6, half_x_m: float = 30.0,
                              use_hash_points: bool = True, seed_centre=None, fixed_centre=None,
                              fov_band=None,
                              require_quality: bool = True,
                              max_rms_px: float | None = None,
                              align_origins: bool = False,
                              propagate: bool = False,
                              verify_top: int = 2):
    """One camera centre for the whole play, seeded from the paint itself.

    Each frame's homography is fitted from that frame's paint alone and carries
    that frame's mistakes into the pose. The camera is on a tripod -- confirmed
    on this footage, since off-plane structure fits the same frame-to-frame
    homography as the field -- so the play shares one centre, and imposing that
    lets the good frames outvote the bad. It is the same constraint that took
    the helmet-fitted cameras from 6.17 m to 0.13 m.

    Set propagate to label the field ONCE and carry that labelling to every
    frame through image-to-image homographies, instead of labelling each frame
    on its own. Independent labelling lets the world slide between frames in
    both origin and scale, which is what makes excellent single frames
    irreconcilable; see propagate_field_homographies.
    """
    scale = height / PAINT_AUDIT_REF_HEIGHT
    if audit_px is None:
        audit_px = PAINT_AUDIT_PX * scale

    xs = np.linspace(-half_x_m, half_x_m, grid)
    ys = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, grid)
    world = np.array([[x, y, 0.0] for x in xs for y in ys])

    raw, scored = {}, {}
    for frame, feats in features_by_frame.items():
        image = None if images is None else images.get(frame)
        got = best_frame_homography(feats, width, height, gap=gap, image=image, fov_band=fov_band,
                                    use_hash_points=use_hash_points)
        if got is not None:
            raw[int(frame)] = got[0]
            scored[int(frame)] = (got[4], -got[2])       # grid, then -residual
    if len(raw) < 3:
        raise CalibrationError(
            f"only {len(raw)} frames could be labelled from paint; need 3.")

    if propagate and images is not None:
        # Try several reference frames and keep the one whose CAMERA comes out
        # best. The per-frame proxies cannot tell a correct labelling from a
        # Y-rescaled one, and picking on them alone was unstable: the same clip
        # gave a 26.9 degree camera from one frame set and 2.7 degrees from
        # another, purely on which frame won.
        # Two stages. Screen EVERY labelled frame cheaply, then run the full
        # solve only on the few that survive. Trying a handful chosen by
        # per-frame proxies was unstable -- one clip gave 5.1 px on one frame
        # sample and 63 px on another purely on which candidate was offered.
        links = neighbour_chain(images, raw)
        screened = []
        for ref in sorted(raw):
            plausible, spread, carried = screen_reference(
                links, raw, ref, width, height, features=features_by_frame)
            if plausible and np.isfinite(spread):
                screened.append((spread, ref, carried))
        screened.sort(key=lambda z: z[0])
        if not screened:
            raise CalibrationError(
                "no reference labelling implied a camera that could exist.")

        best, best_key = None, None
        for _spread, ref, carried in screened[:verify_top]:
            try:
                got = _solve_pooled(carried, world, width, height,
                                    seed_centre, audit_px, fixed_centre=fixed_centre)
            except CalibrationError:
                continue
            q = got[4]
            # Plausibility is a FILTER, never a tie-break. Ranking implausible
            # candidates by rms selects the most degenerate one available: a
            # camera far enough away compresses the whole field into a handful
            # of pixels, so its reprojection error goes to zero for the same
            # reason it is useless. Measured, that is exactly what won -- 1.5 px
            # rms from a camera 1981 m down the field.
            if not q["plausible_mount"]:
                continue
            if best_key is None or q["rms_px"] < best_key:
                best, best_key = got, q["rms_px"]
        if best is None:
            raise CalibrationError(
                "no reference labelling produced a camera that could exist; "
                "every candidate was rejected as an impossible mount.")
        cams, focal, centre, mirrored, quality = best
    else:
        if align_origins:
            raw, _moved = align_x_origins(raw, width, height)
        cams, focal, centre, mirrored, quality = _solve_pooled(
            raw, world, width, height, seed_centre, audit_px, fixed_centre=fixed_centre)

    if require_quality:
        _enforce_quality(quality, centre, max_rms_px, scale)
    _LOG.info("pooled paint: %d frames, focal %.0f, centre %s, mirrored=%s, "
              "rms %.1f px", len(cams), focal, np.round(centre, 1), mirrored,
              quality["rms_px"])
    return cams, focal, centre, mirrored, quality


# A camera this far from the field, or this high, is not a broadcast mount.
PLAUSIBLE_HEIGHT_M = (8.0, 90.0)
PLAUSIBLE_RANGE_M = (25.0, 220.0)


def solve_quality(results, cams, frame_data, centre, focal, width: int) -> dict:
    """Tracking-free signals for whether a paint camera can be believed.

    None of these look at ground truth, which is the point: in production there
    is nothing to compare against, so a camera has to be judged on its own
    evidence or not at all. Reported rather than acted on here, so their
    predictive value can be measured before anything is gated on them.
    """
    rms = [float(results[f].rms_px) for f in cams
           if getattr(results[f], "rms_px", None) is not None]
    centre = np.asarray(centre, float)
    height = float(centre[2])
    ground_range = float(np.hypot(centre[0], centre[1]))
    # Focal as a field of view is easier to sanity-check than raw pixels.
    fov_deg = float(np.degrees(2.0 * np.arctan(width / (2.0 * max(focal, 1e-6)))))
    return {
        "rms_px": float(np.median(rms)) if rms else float("nan"),
        "kept_frac": float(len(cams) / max(len(frame_data), 1)),
        "height_m": height,
        "range_m": ground_range,
        "fov_deg": fov_deg,
        # The LENS counts as part of the mount. A camera can sit in exactly the
        # right place behind a lens that could never cover a football field,
        # which is what a residual Y-scale error looks like once the position
        # has been forced to be sensible.
        "plausible_mount": bool(
            PLAUSIBLE_HEIGHT_M[0] <= height <= PLAUSIBLE_HEIGHT_M[1]
            and PLAUSIBLE_RANGE_M[0] <= ground_range <= PLAUSIBLE_RANGE_M[1]
            and PLAUSIBLE_FOV_DEG[0] <= fov_deg <= PLAUSIBLE_FOV_DEG[1]),
    }


# --------------------------------------------------------------------------
# Label once, propagate. The fix for per-frame labelling drift.
# --------------------------------------------------------------------------

# Lowe ratio and RANSAC tolerance for frame-to-frame matching. Measured on both
# feeds, these give 73-87% inliers at frame gaps of 15 to 40.
MATCH_RATIO: float = 0.75
MATCH_RANSAC_PX: float = 3.0
MIN_MATCHES: int = 40


def image_to_image(image_a, image_b, *, ratio: float = MATCH_RATIO,
                   ransac_px: float = MATCH_RANSAC_PX,
                   min_matches: int = MIN_MATCHES):
    """Homography carrying image A onto image B, or None.

    Valid whatever the camera did between the two frames, because it is fitted
    to the picture rather than to a world model. Measured to be reliable on this
    footage: 73-87% inliers at gaps of 15-40 frames.
    """
    import cv2

    sift = cv2.SIFT_create(nfeatures=4000)
    ga = (cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
          if image_a.ndim == 3 else image_a)
    gb = (cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
          if image_b.ndim == 3 else image_b)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < min_matches:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)
    if H is None or int(mask.sum()) < min_matches:
        return None
    return H


def rank_references(per_frame, scored, width: int, height: int, *, limit: int = 4):
    """Candidate reference frames, best proxy first.

    A ranking rather than a single pick, because the per-frame proxies cannot
    reliably identify a correct labelling. Grid consistency is blind to a wrong
    row assignment (a pure Y rescale leaves every hash mark on the X ruler) and
    the plausibility of one frame's own decomposition is noisy. Measured, the
    single-pick version was unstable: the same clip gave a 26.9 degree camera
    from one frame set and a 2.7 degree one from another, purely on which frame
    won. The caller tries these in order and keeps whichever produces the best
    CAMERA, which is the thing actually wanted.
    """
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt

    ranked = []
    for frame, H in per_frame.items():
        try:
            _K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            continue
        centre = -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)
        fov = implied_fov_deg(H, width, height)
        ok = (PLAUSIBLE_HEIGHT_M[0] <= centre[2] <= PLAUSIBLE_HEIGHT_M[1]
              and PLAUSIBLE_RANGE_M[0] <= float(np.hypot(*centre[:2]))
              <= PLAUSIBLE_RANGE_M[1]
              and (not np.isfinite(fov)
                   or PLAUSIBLE_FOV_DEG[0] <= fov <= PLAUSIBLE_FOV_DEG[1]))
        ranked.append(((not ok, [-v for v in scored.get(frame, (0.0, 0.0))]), frame))
    ranked.sort(key=lambda z: (z[0][0], z[0][1]))
    out = [f for _k, f in ranked][:limit]
    return out or ([max(scored, key=lambda f: scored[f])] if scored else [])


def choose_reference(per_frame, scored, width: int, height: int):
    """The frame whose labelling should be trusted with the whole clip.

    Grid consistency is NOT enough on its own, and this is the trap that made
    propagation actively harmful at first. A wrong row assignment -- hash rows
    called sidelines -- still scores 1.00, because rescaling world Y leaves
    every hash mark on the X ruler. Picking the reference on grid alone chose
    such a frame and then propagated its error perfectly: the frames agreed with
    each other beautifully (centre spread fell by half) on a camera 3.7 km away
    and 1.5 km underground.

    So a candidate must also DECOMPOSE to a camera that could exist. That is
    the one check a wrong Y scale cannot survive.
    """
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt

    ranked = []
    for frame, H in per_frame.items():
        try:
            _K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            continue
        centre = -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)
        height_ok = PLAUSIBLE_HEIGHT_M[0] <= centre[2] <= PLAUSIBLE_HEIGHT_M[1]
        range_ok = (PLAUSIBLE_RANGE_M[0] <= float(np.hypot(*centre[:2]))
                    <= PLAUSIBLE_RANGE_M[1])
        fov = implied_fov_deg(H, width, height)
        fov_ok = (not np.isfinite(fov)
                  or PLAUSIBLE_FOV_DEG[0] <= fov <= PLAUSIBLE_FOV_DEG[1])
        if not (height_ok and range_ok and fov_ok):
            continue
        ranked.append((scored.get(frame, (0.0, 0.0)), frame))
    if not ranked:
        # Nothing decomposes to a possible camera; fall back to the best label
        # and let the quality gate refuse whatever comes out.
        return max(scored, key=lambda f: scored[f]) if scored else None
    ranked.sort(reverse=True)
    return ranked[0][1]





# How close a predicted line must come to a detected one, as a fraction of image
# height, to count as explained.
AGREE_TOL_FRAC: float = 0.02

# Frames used to score one candidate reference. A propagated labelling puts
# every frame on the same world, so a handful says as much as all of them, and
# scoring all of them made the search quadratic.
_SCREEN_FRAMES: int = 5


def paint_agreement(H, features, width: int, height: int,
                    *, tol_frac: float = AGREE_TOL_FRAC) -> float:
    """How much of the WHOLE field model the picture agrees with, 0 to 1.

    The residual of a fit says how well the lines it was given are explained,
    which is not the question -- a wrong labelling fits its own lines perfectly.
    This asks the held-out question instead: with this homography, every line on
    a football field has a predicted place in the image, so how many of the
    predicted ones are actually there, and how much of the detected paint is
    accounted for?

    A labelling that calls the hash rows the sidelines has to predict real
    sidelines somewhere no paint exists, and leaves the true hash rows
    unexplained. Nothing about its own fit objects; this does.

    Needed because plausibility alone does not settle it: the same sideline clip
    produced a 14.4 degree camera 146 m out and a 41.7 degree one 58 m out, both
    passing every physical check, on different frame samples.
    """
    H = np.asarray(H, float)
    tol = tol_frac * height
    try:
        Hinv_T = np.linalg.inv(H).T
    except np.linalg.LinAlgError:
        return 0.0

    detected = [seg_line(s) for s in features.yard_lines]
    detected += [seg_line(s) for s in features.sidelines]
    if not detected:
        return 0.0

    def crossing(line):
        return line_at_x(line, width / 2.0)

    det_v = [v for v in (crossing(ln) for ln in detected) if v is not None]

    predicted = []
    for y in CROSS_Y_M:
        pred = Hinv_T @ np.array([0.0, 1.0, -y])
        n = float(np.hypot(pred[0], pred[1]))
        if n < 1e-12:
            continue
        v = crossing(pred / n)
        if v is not None and -0.2 * height <= v <= 1.2 * height:
            predicted.append(v)
    if not predicted:
        return 0.0

    # Recall: predicted long lines that are actually visible in the picture.
    hits = sum(1 for v in predicted
               if det_v and min(abs(v - d) for d in det_v) <= tol)
    recall = hits / len(predicted)

    # Precision: detected paint the model can account for, using the hash
    # marks, which are the densest evidence available.
    blobs = (np.asarray(features.hashes, float).reshape(-1, 2)
             if len(features.hashes) else np.zeros((0, 2)))
    precision = (grid_consistency(H, blobs, list(CROSS_Y_M))
                 if len(blobs) >= MIN_HASH_POINTS else recall)
    if recall <= 0 or precision <= 0:
        return 0.0
    return float(2.0 * recall * precision / (recall + precision))


def neighbour_chain(images, frames):
    """``{frame: H}`` carrying each frame onto the NEXT one, computed once.

    The links between neighbours do not depend on which frame is later chosen
    as the reference, so computing them once and composing turns an O(n^2) pile
    of SIFT matching into O(n). Without this, screening every candidate
    reference re-matched the whole clip per candidate and took longer than the
    solve it was meant to make affordable.
    """
    order = sorted(frames)
    links = {}
    for a, b in zip(order, order[1:]):
        H = image_to_image(images[a], images[b])
        if H is not None:
            links[a] = H
    _LOG.info("frame chain: %d/%d neighbour links", len(links), max(len(order) - 1, 1))
    return links


def carry_with_chain(links, per_frame, reference):
    """Propagate one labelling using precomputed neighbour links."""
    order = sorted(per_frame)
    if reference not in per_frame:
        return {}
    pivot = order.index(reference)
    to_ref = {reference: np.eye(3)}

    # Forward: frame i -> reference, walking back down the links.
    for i in range(pivot + 1, len(order)):
        prev, cur = order[i - 1], order[i]
        H = links.get(prev)
        if H is None or prev not in to_ref:
            break
        try:
            to_ref[cur] = to_ref[prev] @ np.linalg.inv(H)
        except np.linalg.LinAlgError:
            break
    # Backward.
    for i in range(pivot - 1, -1, -1):
        cur, nxt = order[i], order[i + 1]
        H = links.get(cur)
        if H is None or nxt not in to_ref:
            break
        to_ref[cur] = to_ref[nxt] @ H

    H_ref = np.asarray(per_frame[reference], float)
    out = {}
    for f, G in to_ref.items():
        try:
            out[f] = np.linalg.inv(G) @ H_ref
        except np.linalg.LinAlgError:
            continue
    return out


def screen_reference(links, raw, reference, width: int, height: int,
                     features=None):
    """Cheap score for one candidate reference, without running the solve.

    Propagating a labelling makes every frame share one world, so if that
    labelling is right the frames must all imply the SAME camera -- and a camera
    that could exist. Both are readable from the per-frame decompositions alone,
    which costs a matrix inverse per frame instead of a full bundle.

    Returns ``(plausible, cost, carried)``; lower cost is better. This
    exists because the full solve is far too slow to try on every frame, while
    the proxies that are cheap enough (grid consistency, per-frame residual)
    cannot see a wrong labelling at all. Measured, choosing among only four
    candidates by those proxies was unstable: one endzone clip solved at 5.1 px
    on one frame sample and 63 px on another.
    """
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt

    carried = carry_with_chain(links, raw, reference)
    if len(carried) < 3:
        return False, np.inf, {}
    centres = []
    for H in carried.values():
        try:
            _K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            continue
        centres.append(-np.asarray(R, float).T @ np.asarray(t, float).reshape(3))
    if len(centres) < 3:
        return False, np.inf, carried
    C = np.stack(centres)
    med = np.median(C, axis=0)
    fov = implied_fov_deg(carried[reference], width, height)
    plausible = bool(
        PLAUSIBLE_HEIGHT_M[0] <= med[2] <= PLAUSIBLE_HEIGHT_M[1]
        and PLAUSIBLE_RANGE_M[0] <= float(np.hypot(*med[:2]))
        <= PLAUSIBLE_RANGE_M[1]
        and (not np.isfinite(fov)
             or PLAUSIBLE_FOV_DEG[0] <= fov <= PLAUSIBLE_FOV_DEG[1]))
    # Centre spread cannot be the score: propagation forces every frame onto
    # one world, so their centres agree for ANY reference and the number
    # measures decomposition noise rather than whether the labelling is right.
    # Model agreement is the held-out question, and it does discriminate.
    if features:
        # On a SUBSAMPLE. Every candidate reference is screened against every
        # frame otherwise, which is quadratic and dominated the runtime; the
        # agreement of a propagated labelling barely varies between frames,
        # because they all share one world by construction.
        usable = [f for f in sorted(carried) if f in features]
        step = max(1, len(usable) // _SCREEN_FRAMES)
        agree = float(np.median([
            paint_agreement(carried[f], features[f], width, height)
            for f in usable[::step][:_SCREEN_FRAMES]] or [0.0]))
    else:
        agree = 0.0
    return plausible, 1.0 - agree, carried


def propagate_field_homographies(images, per_frame, *, reference=None):
    """One labelling, carried to every frame by image-to-image homographies.

    THE PROBLEM THIS SOLVES. Labelling each frame independently lets its world
    slide: as the camera pans, a different yard line becomes that frame's
    "index 0", and a frame that loses a line under players can read the spacing
    as 10 yards where its neighbour read 5. Each frame is then self-consistent
    -- its hash marks land on its own grid, measured at 0.97 -- while no rigid
    camera satisfies them all. That is exactly the All-22 failure: excellent
    frames, a 100-240 px shared-centre residual.

    Labelling ONCE removes both the origin and the scale ambiguity by
    construction, because every frame inherits the same world. The camera was
    separately confirmed to only ROTATE on this footage -- off-plane structure
    fits the same homography as the field, at the same rate as on the broadcast
    clips -- so an image-to-image homography is exact and chaining is sound.

    Chained through neighbours rather than matched directly to a distant
    reference, since a frame at the other end of a pan may share no view with it
    at all.
    """
    frames = sorted(f for f in per_frame if per_frame[f] is not None)
    if len(frames) < 2:
        return dict(per_frame)
    if reference is None:
        reference = frames[len(frames) // 2]
    if reference not in per_frame or per_frame[reference] is None:
        return dict(per_frame)

    # Walk outward from the reference, composing neighbour-to-neighbour maps.
    to_ref = {reference: np.eye(3)}
    order = sorted(frames)
    pivot = order.index(reference)
    for direction in (1, -1):
        i = pivot
        while 0 <= i + direction < len(order):
            cur, nxt = order[i], order[i + direction]
            G = image_to_image(images[nxt], images[cur])
            if G is None or cur not in to_ref:
                break
            to_ref[nxt] = to_ref[cur] @ G
            i += direction

    H_ref = np.asarray(per_frame[reference], float)
    out = {}
    for f in frames:
        if f not in to_ref:
            continue
        try:
            # world -> image_f  =  (image_f -> ref)^-1  applied to world -> ref
            out[f] = np.linalg.inv(to_ref[f]) @ H_ref
        except np.linalg.LinAlgError:
            continue
    _LOG.info("propagated one labelling to %d/%d frames from frame %d",
              len(out), len(frames), reference)
    return out

