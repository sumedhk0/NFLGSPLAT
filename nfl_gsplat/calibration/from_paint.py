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

HOW GOOD IS IT, measured over 35 sideline views. A paint-only camera places
players a median 3.24 m from tracking where the tracking-fitted camera places
them 0.34 m, with the best play at 0.60 m and 13 of 35 inside 2 m. So paint
works, is far looser, and is not yet a replacement -- it is an initialiser, and
the honest state of the tracking-free path.

Free evidence that the LABELLING is right even when the pose is loose: the
recovered X shift lands within 0.15 of a whole 5-yard step on 20 of 35 views.
Nothing forces that, so a near-integer shift means the yard lines were
identified correctly and the error is in the fit, not the assignment.

THE ENDZONE VIEW IS NOT SOLVED. Its residuals run ~199 px against the sideline's
13 px and most views fail outright, usually with too few frames yielding a
focal. That camera looks ALONG the field, so its yard lines crowd toward the
vanishing point and little of the field's width is in frame.

A SHARED-CENTRE SOLVE ON PAINT WAS TRIED AND DOES NOT WORK YET. Pooling one
camera centre across the play is what took the helmet-fitted cameras from
6.17 m to 0.13 m, so the same was attempted here: fit each frame's homography,
sample points from it, hand them to joint_solve. It failed on all six views
tried -- the multi-start refit found no frames consistent with any candidate
camera, because per-frame paint poses scatter more than the audit tolerates.
The idea is still the most promising route to closing the gap; it needs better
per-frame homographies first, not a looser gate.
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
MIN_YARD_LINE_PX: float = 300.0

# Two rows within this many pixels at the image centre are the same line
# detected twice.
ROW_MERGE_PX: float = 25.0

GAP_CANDIDATES: tuple[int, ...] = (1, 2, 3)

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
        rows.append((ln / float(np.hypot(ln[0], ln[1])), int(inl.sum())))
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
    candidates = list(fit_rows(blobs))
    candidates += [(seg_line(s), 0) for s in features.sidelines]
    out = []
    for line, n in sorted(candidates, key=lambda z: -z[1]):
        v = line_at_x(line, width / 2.0)
        if v is None or not (-200 < v < height + 200):
            continue
        if all(abs(v - line_at_x(o[0], width / 2.0)) > ROW_MERGE_PX for o in out):
            out.append((line, n, v))
    out.sort(key=lambda z: z[2])          # top of the image first
    return out


def yard_ladder(features, *, min_len_px: float = MIN_YARD_LINE_PX):
    """Detected yard lines, filtered and ordered across the image."""
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
                     transposed: bool = False):
    """``(H, world_y, residual_px)`` for one frame, X ORIGIN ARBITRARY.

    The assignment of detected long lines to the field's known Y values is
    searched over, order-preserving, and scored by how well the resulting
    homography reproduces every line.
    """
    from itertools import combinations

    if transposed:
        features = _Transposed(features)
        width, height = height, width

    rows = long_lines(features, width, height)
    ladder = yard_ladder(features)
    if len(rows) < 2 or len(ladder) < 3:
        return None

    steps = ladder_indices(ladder)

    best = None
    for pick in combinations(range(len(CROSS_Y_M)), len(rows)):
        world_y = [CROSS_Y_M[i] for i in pick]
        world_lines, image_lines = [], []
        for k, seg in zip(steps, ladder):
            world_lines.append([1.0, 0.0, -k * gap * YARD_LINE_SPACING_M])
            image_lines.append(seg_line(seg))
        for (line, _n, _v), y in zip(rows, world_y):
            world_lines.append([0.0, 1.0, -y])
            image_lines.append(line)
        H = homography_from_lines(world_lines, image_lines)
        if H is None:
            continue
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
        if best is None or res < best[2]:
            best = (H, world_y, res)
    if best is not None and transposed:
        best = (_SWAP_UV @ best[0], best[1], best[2])
    return best


def best_frame_homography(features, width: int, height: int, *, gap: int = 1):
    """``(H, world_y, residual, transposed)`` -- tries both view orientations."""
    out = []
    for flag in (False, True):
        got = frame_homography(features, width, height, gap=gap, transposed=flag)
        if got is not None:
            out.append((got[2], got[0], got[1], flag))
    if not out:
        return None
    out.sort(key=lambda z: z[0])
    res, H, world_y, flag = out[0]
    return H, world_y, res, flag


def cameras_from_paint(features_by_frame, width: int, height: int, *,
                       gap: int = 1, transposed=None):
    """``({frame: (K, R, t)}, focal, residual)`` in the GROUND frame.

    The focal is pooled over frames and then held fixed, for the same reason as
    in ``from_helmets``: one frame constrains it far more weakly than the play
    as a whole.
    """
    per_frame, focals, residuals = {}, [], []
    for frame, feats in features_by_frame.items():
        if transposed is None:
            got = best_frame_homography(feats, width, height, gap=gap)
        else:
            got = frame_homography(feats, width, height, gap=gap,
                                   transposed=transposed)
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
