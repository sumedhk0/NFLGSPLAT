"""The endzone camera from its own paint: registered yard lines, hash-mark columns, the goal line.

WHY. The endzone camera was solved from the players' feet against the sideline's
placement (calibration.from_players): the mount CENTRE a grid prior held at
(60, 0, 20), the rotation per frame from 8-20 box bottoms. Measured on play 1
(2026-09-09): its projected grid sits 40-85 px off the painted lines, the two
cameras' rays through the same keypoint miss by 0.21 m at the median and 0.45 m
while the camera pans (frames 255-290), triangulated hips come out at 1.7 m and
ankles at 0.3-0.8 m -- every two-view pose, pairing and triangulation defect
traced back here. A paint refinement of rotation and focal alone
(calibration.refine_paint, centre held) halves the grid error and makes the rays
miss five times worse: a pencil of parallel yard lines does not see the lateral
axis, and the wrong centre cannot be fixed by rotating.

WHAT. Per frame the endzone view shows the far goal line (the red end zone
meeting the green field), the 5-yard lines running across the image, and the
two hash-mark columns (short dashes at y = +-3.124 m, one per yard). Together
they fix the camera completely: yard lines give pitch, roll and the depth
scale; the hash columns give yaw and the lateral position; the goal line
registers WHICH line is which, and consecutive counting registers the rest
(the nearest-line assignment of refine_paint slips a line at the far end where
lines are 60 px apart and the starting camera is 60 px off).

The mount centre is solved ONCE over a sample of frames (a tripod does not
move); rotation and focal are then solved per frame, each frame starting from
its neighbour's camera so the registration carries when the goal line leaves
the view, and the per-frame deltas are smoothed along the track.

RULERS. The paint itself (grid px) is what the fit minimises and proves only
consistency. The independent rulers are the players seen by both cameras: the
sideline's and the endzone's rays through the same keypoint must meet (metres
of miss), triangulated ankles must sit on the turf and hips at about a metre.
The caller (scripts/08l) refuses to apply a refinement that the players do not
confirm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nfl_gsplat.calibration import field_detect as fd
from nfl_gsplat.calibration.field_landmarks import GOAL_LINE_X_M, HALF_WIDTH_M, HASH_OFFSET_M, YARD_LINE_SPACING_M
from nfl_gsplat.calibration.grid_fit import detect_segments_any, projected_lines
from nfl_gsplat.calibration.refine_paint import gate_by_orientation
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

LINE_ORIENT_TOL_DEG: float = 15.0
LINE_CLUSTER_PX: float = 10.0      # segments whose row at the image centre is within this are one line
GOAL_ROW_TOL_PX: float = 30.0      # the goal line is the first detected line within this much below the red/green boundary
DASH_THRESH: int = 150             # the hash dashes are thin and blurred: peak grey 180-210 on play 1
DASH_W_PX: tuple = (10, 90)
DASH_H_MAX_PX: int = 9
DASH_ROW_GATE_PX: float = 250.0    # a dash counts for the hash row it is nearest to, within this
DASH_LINE_CLEAR_PX: int = 7        # dashes this close to a yard-line segment are line fragments
SOFT_L1_PX: float = 4.0
PRIOR_PX_PER_DEG: float = 20.0
PRIOR_PX_PER_10PCT_FOCAL: float = 20.0
GOAL_WEIGHT: float = 0.0           # the red/green boundary registers the lines; it is not the goal line itself
MAX_ROT_DEG: float = 10.0
MAX_FOCAL_CHANGE: float = 0.6
CENTRE_BOUNDS_M = ((-40.0, 80.0), (-25.0, 25.0), (-15.0, 30.0))   # around the starting centre
SMOOTH_FRAMES: int = 9
CENTRE_FRAME_MAX_PX: float = 6.0    # a centre frame the joint fit leaves further off the paint is dropped
GOAL_EDGE_CLEAR_PX: float = 8.0     # the red end zone's own white edge sits at the boundary; the goal line is below it
ACCEPT_LINE_PX: float = 6.0         # a frame's fit is kept only this close to its lines ...
ACCEPT_DASH_PX: float = 6.0         # ... and its dashes ...
ACCEPT_STEP_DEG: float = 3.0        # ... and this close to the camera it started from


@dataclass
class PaintFrame:
    """What one endzone frame shows of the paint, registered."""
    seg_p0: np.ndarray          # [S, 3] homogeneous endpoints
    seg_p1: np.ndarray          # [S, 3]
    seg_k: np.ndarray           # [S] yard-line index from the far goal line (0), int
    dashes: np.ndarray          # [D, 3] homogeneous dash centroids
    dash_row: np.ndarray        # [D] 0 = the y < 0 hash row, 1 = the y > 0 row
    goal_row: float | None      # image row of the far goal line at the centre column, if seen
    n_lines: int = 0


@dataclass
class FrameFit:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    before_px: float
    after_px: float
    dash_px: float
    goal_px: float
    applied: bool
    rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    log_f: float = 0.0


# ----------------------------------------------------------------------------- geometry

def hash_row_lines(K, R, t):
    """Homogeneous image lines [2, 3] of the hash rows y = -h, +h (unit normal), or None."""
    out = []
    for y in (-HASH_OFFSET_M, HASH_OFFSET_M):
        a = K @ (R @ np.array([-GOAL_LINE_X_M, y, 0.0]) + t)
        b = K @ (R @ np.array([GOAL_LINE_X_M, y, 0.0]) + t)
        if a[2] <= 1e-9 or b[2] <= 1e-9:
            return None
        L = np.cross(a / a[2], b / b[2])
        n = np.linalg.norm(L[:2])
        if n < 1e-12:
            return None
        out.append(L / n)
    return np.stack(out)


def goal_line_row(K, R, t, u: float):
    """Image row where the far goal line (x = -GOAL_LINE_X_M) crosses column ``u``."""
    a = K @ (R @ np.array([-GOAL_LINE_X_M, -HALF_WIDTH_M, 0.0]) + t)
    b = K @ (R @ np.array([-GOAL_LINE_X_M, HALF_WIDTH_M, 0.0]) + t)
    if a[2] <= 1e-9 or b[2] <= 1e-9:
        return float("nan")
    a, b = a / a[2], b / b[2]
    if abs(b[0] - a[0]) < 1e-9:
        return float(a[1])
    return float(a[1] + (b[1] - a[1]) * (u - a[0]) / (b[0] - a[0]))


def line_rows(K, R, t, u: float):
    """Image row of every 5-yard line (k = 0 at the far goal line) at column ``u``; NaN behind the camera."""
    n = int(round(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M))
    rows = np.full(n + 1, np.nan)
    for k in range(n + 1):
        x = -GOAL_LINE_X_M + YARD_LINE_SPACING_M * k
        a = K @ (R @ np.array([x, -HALF_WIDTH_M, 0.0]) + t)
        b = K @ (R @ np.array([x, HALF_WIDTH_M, 0.0]) + t)
        if a[2] <= 1e-9 or b[2] <= 1e-9:
            continue
        a, b = a / a[2], b / b[2]
        rows[k] = a[1] if abs(b[0] - a[0]) < 1e-9 else a[1] + (b[1] - a[1]) * (u - a[0]) / (b[0] - a[0])
    return rows


def camera_from(params, K0, R0, centre):
    """Rotation ``exp(w) R0``, focal ``f0 exp(s)``, the given centre."""
    w, s = np.asarray(params[:3], float), float(params[3])
    R = Rotation.from_rotvec(w).as_matrix() @ np.asarray(R0, float)
    K = np.asarray(K0, float).copy()
    K[0, 0] = K0[0, 0] * np.exp(s)
    K[1, 1] = K0[1, 1] * np.exp(s)
    t = -R @ np.asarray(centre, float)
    return K, R, t


# ----------------------------------------------------------------------------- detection

def red_green_boundary(img, boxes=None, *, cols=(200, 1700), max_row=450):
    """Row where the far end zone's red meets the field's green, from the per-row median of
    (R - G) over the turf pixels (white paint, letters and player boxes left out): the first
    row from the top after which the turf stays green. None when the top is not red -- the
    goal line is NOT this row (play 1: the red stops 25 px short of the goal line), it is
    the first yard line below it (register_lines)."""
    im = np.asarray(img).astype(float)
    m = np.ones(im.shape[:2], bool)
    for x1, y1, x2, y2 in boxes or []:
        m[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = False
    m &= im.min(axis=2) < 140                                  # not white paint or letters
    rg = np.where(m, im[:, :, 2] - im[:, :, 1], np.nan)[:max_row, cols[0]:cols[1]]
    row = np.nanmedian(rg, axis=1)
    row = np.nan_to_num(row, nan=0.0)
    red = row > 25
    green = row < -8
    if red[:60].sum() < 10:                                    # the top is not red
        return None
    for r in range(20, len(row) - 20):
        if red[:r].sum() >= 10 and green[r:r + 20].all():
            return float(r)
    return None


def _column_inliers(pts, *, tol_px: float = 4.0, iters: int = 200, seed: int = 0):
    """RANSAC line through 2-D points; the inlier mask of the best line (a straight hash
    column is a line in the image; letters and stray blobs are not on it)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 6:
        return np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        nrm = np.hypot(*d)
        if nrm < 30:
            continue
        normal = np.array([-d[1], d[0]]) / nrm
        dist = np.abs((pts - pts[i]) @ normal)
        inl = dist <= tol_px
        if inl.sum() > best.sum():
            best = inl
    if best.sum() >= 6:                                       # refit on the inliers, one more pass
        c = pts[best].mean(axis=0)
        u, _s, vt = np.linalg.svd(pts[best] - c)
        normal = np.array([-vt[0, 1], vt[0, 0]])
        best = np.abs((pts - c) @ normal) <= tol_px
    return best


def detect_lines(img, boxes, K, R, t, *, u: float = 960.0):
    """Yard-line segments (gated to the camera's line direction) clustered into distinct lines
    by their row at column ``u``. Returns ``[(row, p0 [S,3], p1 [S,3]), ...]`` sorted by row."""
    segs = gate_by_orientation(detect_segments_any(img, boxes), projected_lines(K, R, t), LINE_ORIENT_TOL_DEG)
    if not segs:
        return []
    p0 = np.asarray([[s.p0[0], s.p0[1], 1.0] for s in segs])
    p1 = np.asarray([[s.p1[0], s.p1[1], 1.0] for s in segs])
    dx = p1[:, 0] - p0[:, 0]
    dx = np.where(np.abs(dx) < 1e-6, 1e-6, dx)
    row_u = p0[:, 1] + (u - p0[:, 0]) * (p1[:, 1] - p0[:, 1]) / dx
    order = np.argsort(row_u)
    lines, cur = [], [order[0]]
    for i in order[1:]:
        if row_u[i] - row_u[cur[-1]] <= LINE_CLUSTER_PX:
            cur.append(i)
        else:
            lines.append(cur)
            cur = [i]
    lines.append(cur)
    out = []
    for idx in lines:
        idx = np.asarray(idx)
        out.append((float(np.median(row_u[idx])), p0[idx], p1[idx]))
    return out


def register_lines(lines, K, R, t, goal_row, *, u: float = 960.0):
    """Yard-line index k (0 = far goal line) per detected line. With a goal row the line
    nearest it is k = 0 and the rest count on by the camera's projected spacing; without
    one every line takes the nearest projected line under the camera (tracking mode).
    Returns ``[k or None, ...]``."""
    if not lines:
        return []
    pred = line_rows(K, R, t, u)
    ok = np.isfinite(pred)
    rows = np.asarray([ln[0] for ln in lines])
    ks: list = [None] * len(lines)
    if goal_row is not None:
        # the goal line is the first yard line BELOW the red: the red paint stops short of
        # it (25 px on play 1), its own white edge sits AT the boundary, and the end-zone
        # letters make lines above it
        below = np.flatnonzero((rows > goal_row + GOAL_EDGE_CLEAR_PX) & (rows < goal_row + GOAL_ROW_TOL_PX + 30))
        anchor = int(below[0]) if len(below) else -1
        if anchor >= 0:
            ks[anchor] = 0
            k = 0
            for i in range(anchor + 1, len(lines)):
                gap = rows[i] - rows[i - 1]
                # the projected spacing at this row, from the nearest projected lines
                pk = np.flatnonzero(ok)
                near = pk[np.argmin(np.abs(pred[pk] - rows[i - 1]))]
                if near + 1 < len(pred) and np.isfinite(pred[near + 1]):
                    spacing = abs(pred[near + 1] - pred[near])
                elif near - 1 >= 0 and np.isfinite(pred[near - 1]):
                    spacing = abs(pred[near] - pred[near - 1])
                else:
                    spacing = gap
                ratio = gap / max(spacing, 1e-6)
                if ratio < 0.5:                 # a second cluster of the same line (play 1 f456: 788 and 800)
                    ks[i] = k if k < len(pred) else None
                    continue
                k += max(1, int(round(ratio)))
                ks[i] = k if k < len(pred) else None
            return ks
    for i, r in enumerate(rows):
        pk = np.flatnonzero(ok)
        if len(pk) == 0:
            continue
        near = pk[np.argmin(np.abs(pred[pk] - r))]
        ks[i] = int(near) if abs(pred[near] - r) < 0.45 * _spacing_at(pred, near) else None
    return ks


def _spacing_at(pred, k):
    cands = []
    if k + 1 < len(pred) and np.isfinite(pred[k + 1]):
        cands.append(abs(pred[k + 1] - pred[k]))
    if k - 1 >= 0 and np.isfinite(pred[k - 1]):
        cands.append(abs(pred[k] - pred[k - 1]))
    return min(cands) if cands else 1e9


def detect_dashes(img, boxes, K, R, t, line_segments=None):
    """Centroids of the hash-mark dashes: short, thin, bright blobs off the yard lines,
    each assigned to the hash row it is nearest to under the camera. Returns ``(dashes [D,3], row [D])``."""
    import cv2

    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DASH_THRESH, 255, cv2.THRESH_BINARY)
    mask = fd._zero_boxes(mask, boxes)
    if line_segments:
        for p0, p1 in line_segments:
            for a, b in zip(p0, p1):
                cv2.line(mask, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 0, 2 * DASH_LINE_CLEAR_PX + 1)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if DASH_W_PX[0] <= w <= DASH_W_PX[1] and h <= DASH_H_MAX_PX and w >= 2.5 * max(h, 1) \
                and area >= 0.35 * w * h:
            keep.append(cent[i])
    if not keep:
        return np.zeros((0, 3)), np.zeros(0, int)
    H = hash_row_lines(K, R, t)
    if H is None:
        return np.zeros((0, 3)), np.zeros(0, int)
    d = np.c_[np.asarray(keep, float), np.ones(len(keep))]
    dist = np.abs(d @ H.T)
    row = np.argmin(dist, axis=1)
    near = dist[np.arange(len(d)), row] <= DASH_ROW_GATE_PX
    d, row = d[near], row[near]
    # each column is a straight line in the image: keep what lies on it
    keep_mask = np.zeros(len(d), bool)
    for r in (0, 1):
        idx = np.flatnonzero(row == r)
        if len(idx):
            keep_mask[idx] = _column_inliers(d[idx, :2])
    return d[keep_mask], row[keep_mask]


def measure_frame(img, boxes, K, R, t, *, u: float = 960.0, use_goal: bool = True) -> PaintFrame:
    """Detect and register everything one frame shows of the paint, under camera (K, R, t).
    ``use_goal=False`` registers by the camera alone (tracking mode) even when the red
    end zone is in view."""
    goal = red_green_boundary(img, boxes)
    lines = detect_lines(img, boxes, K, R, t, u=u)
    ks = register_lines(lines, K, R, t, goal if use_goal else None, u=u)
    p0s, p1s, kk, segs_for_mask = [], [], [], []
    for (row, p0, p1), k in zip(lines, ks):
        segs_for_mask.append((p0, p1))
        if k is None:
            continue
        p0s.append(p0)
        p1s.append(p1)
        kk.append(np.full(len(p0), int(k)))
    dashes, drow = detect_dashes(img, boxes, K, R, t, segs_for_mask)
    if p0s:
        P0, P1, KK = np.concatenate(p0s), np.concatenate(p1s), np.concatenate(kk)
    else:
        P0, P1, KK = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
    return PaintFrame(P0, P1, KK, dashes, drow, goal, n_lines=len(lines))


# ----------------------------------------------------------------------------- fitting

def frame_residuals(K, R, t, pf: PaintFrame, *, u: float = 960.0):
    """Paint residuals in pixels: segment endpoints to their registered line, dashes to their
    hash row, the goal line to the red/green boundary (weighted)."""
    parts = []
    if len(pf.seg_k):
        L = projected_lines(K, R, t)
        if len(L) < pf.seg_k.max() + 1:
            return None
        Ls = L[pf.seg_k]
        parts.append(np.einsum("ij,ij->i", pf.seg_p0, Ls))
        parts.append(np.einsum("ij,ij->i", pf.seg_p1, Ls))
    if len(pf.dashes):
        H = hash_row_lines(K, R, t)
        if H is None:
            return None
        parts.append(np.einsum("ij,ij->i", pf.dashes, H[pf.dash_row]))
    # the red/green boundary registers the lines (register_lines) and is NOT a residual:
    # the red paint stops 25 px short of the goal line on play 1, and the goal line
    # itself is among the segments as k = 0
    return np.concatenate(parts) if parts else np.zeros(0)


def _n_resid(pf: PaintFrame):
    return 2 * len(pf.seg_k) + len(pf.dashes)


def _priors(params):
    return np.concatenate([PRIOR_PX_PER_DEG * np.degrees(np.asarray(params[:3], float)),
                           [PRIOR_PX_PER_10PCT_FOCAL * float(params[3]) / np.log(1.1)]])


def frame_score(K, R, t, pf: PaintFrame):
    """``(line px, dash px, goal px)`` medians under a camera (NaN where nothing was seen)."""
    line_px = dash_px = goal_px = float("nan")
    if len(pf.seg_k):
        L = projected_lines(K, R, t)
        if len(L) >= pf.seg_k.max() + 1:
            Ls = L[pf.seg_k]
            d = 0.5 * (np.abs(np.einsum("ij,ij->i", pf.seg_p0, Ls)) + np.abs(np.einsum("ij,ij->i", pf.seg_p1, Ls)))
            line_px = float(np.median(d))
    if len(pf.dashes):
        H = hash_row_lines(K, R, t)
        if H is not None:
            dash_px = float(np.median(np.abs(np.einsum("ij,ij->i", pf.dashes, H[pf.dash_row]))))
    if pf.goal_row is not None:
        goal_px = float(goal_line_row(K, R, t, 960.0) - pf.goal_row)      # signed; ~+25 px on play 1
    return line_px, dash_px, goal_px


def fit_frame(pf: PaintFrame, K0, R0, centre, *, max_nfev: int = 300) -> FrameFit:
    """Rotation and focal of one frame with the centre held, from its registered paint."""
    K0 = np.asarray(K0, float)
    R0 = np.asarray(R0, float)
    t0 = -R0 @ np.asarray(centre, float)
    before = frame_score(K0, R0, t0, pf)
    n = _n_resid(pf)
    if n < 12:
        return FrameFit(K0, R0, t0, before[0], before[0], before[1], before[2], False)

    def resid(x):
        K, R, t = camera_from(x, K0, R0, centre)
        r = frame_residuals(K, R, t, pf)
        if r is None:
            return np.full(n + 4, 1e3)
        return np.concatenate([r, _priors(x)])

    lo = np.array([-np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 - MAX_FOCAL_CHANGE)])
    hi = np.array([np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 + MAX_FOCAL_CHANGE)])
    sol = least_squares(resid, np.zeros(4), loss="soft_l1", f_scale=SOFT_L1_PX, bounds=(lo, hi), max_nfev=max_nfev)
    K, R, t = camera_from(sol.x, K0, R0, centre)
    after = frame_score(K, R, t, pf)
    ok = np.isfinite(after[0]) and (not np.isfinite(before[0]) or after[0] <= before[0] + 1e-6)
    if not ok:
        return FrameFit(K0, R0, t0, before[0], before[0], before[1], before[2], False)
    return FrameFit(K, R, t, before[0], after[0], after[1], after[2], True, sol.x[:3].copy(), float(sol.x[3]))


def fit_centre(frames_pf: list, K0s, R0s, centre0, *, max_nfev: int = 60):
    """The mount centre shared by ``frames_pf`` (each with its own rotation and focal), from
    their registered paint. Returns ``(centre, per-frame params [F, 4])``."""
    F = len(frames_pf)
    ns = [_n_resid(pf) for pf in frames_pf]

    def unpack(x):
        return x[:3], x[3:].reshape(F, 4)

    def resid(x):
        dc, per = unpack(x)
        centre = np.asarray(centre0, float) + dc
        parts = []
        for i, pf in enumerate(frames_pf):
            K, R, t = camera_from(per[i], K0s[i], R0s[i], centre)
            r = frame_residuals(K, R, t, pf)
            parts.append(np.full(ns[i], 1e3) if r is None else r)
            parts.append(_priors(per[i]))
        parts.append(0.2 * dc)                                    # a weak hold on the centre, px per metre
        return np.concatenate(parts)

    lo = np.array([b[0] for b in CENTRE_BOUNDS_M] + [-np.radians(MAX_ROT_DEG)] * 3 * F)
    hi = np.array([b[1] for b in CENTRE_BOUNDS_M] + [np.radians(MAX_ROT_DEG)] * 3 * F)
    # interleave the focal bounds per frame
    lo = np.concatenate([lo[:3]] + [np.array([-np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 - MAX_FOCAL_CHANGE)])] * F)
    hi = np.concatenate([hi[:3]] + [np.array([np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 + MAX_FOCAL_CHANGE)])] * F)
    x0 = np.zeros(3 + 4 * F)
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=SOFT_L1_PX, bounds=(lo, hi), max_nfev=max_nfev)
    dc, per = unpack(sol.x)
    return np.asarray(centre0, float) + dc, per


def smooth_deltas(rotvecs, log_f, *, window: int = SMOOTH_FRAMES):
    """Zero-phase moving average of per-frame deltas; NaN rows interpolated first."""
    rv = np.asarray(rotvecs, float).copy()
    lf = np.asarray(log_f, float).copy()
    ok = np.isfinite(lf)
    if ok.sum() == 0:
        return np.zeros_like(rv), np.zeros_like(lf)
    idx = np.arange(len(lf))
    for d in range(3):
        rv[~ok, d] = np.interp(idx[~ok], idx[ok], rv[ok, d])
    lf[~ok] = np.interp(idx[~ok], idx[ok], lf[ok])
    k = max(1, min(window, len(lf) if len(lf) % 2 else len(lf) - 1))
    if k < 3:
        return rv, lf
    pad = k // 2
    kern = np.ones(k) / k

    def sm(v):
        vp = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
        return np.convolve(vp, kern, mode="valid")

    return np.stack([sm(rv[:, d]) for d in range(3)], 1), sm(lf)


def _rays(K, R, t, uv):
    C = -R.T @ t
    d = (R.T @ np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T)).T
    return C, d / np.linalg.norm(d, axis=1, keepdims=True)


def _closest_points(C1, d1, C2, d2):
    w = C1 - C2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = d1 @ w
    e = d2 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    tt = (a * e - b * d) / den
    p1 = C1 + s[:, None] * d1
    p2 = C2 + tt[:, None] * d2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1)


def player_rulers(track_a, track_b, keypoints, *, offset: int = 0, min_conf: float = 0.5,
                  cam_a: str = "sideline", cam_b: str = "endzone", stride: int = 1):
    """The players as the ruler of two cameras: per frame, the rays of both cameras through
    the same confident keypoint (camera b at ``frame + offset``) -- their miss distance in
    metres and the heights the closest points give the ankles (COCO 15, 16) and hips (11, 12).
    Returns ``{"frames", "miss_p50", "miss_p90", "ankle_z", "hip_z"}`` (arrays per frame)."""
    k = keypoints[keypoints["conf"] >= min_conf]
    A = k[k["cam"] == cam_a].set_index(["frame", "global_player_id", "joint"]).sort_index()
    B = k[k["cam"] == cam_b].set_index(["frame", "global_player_id", "joint"]).sort_index()
    fa = set(A.index.get_level_values(0))
    fb = set(B.index.get_level_values(0))
    out = {"frames": [], "miss_p50": [], "miss_p90": [], "ankle_z": [], "hip_z": []}
    for f in range(0, min(len(track_a.conf), len(track_b.conf) - offset), stride):
        if track_a.conf[f] <= 0 or track_b.conf[f + offset] <= 0 or f not in fa or (f + offset) not in fb:
            continue
        a = A.loc[f]
        b = B.loc[f + offset]
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 6:
            continue
        ua = a.loc[common][["x", "y"]].to_numpy(float)
        ub = b.loc[common][["x", "y"]].to_numpy(float)
        joints = np.array([j for (_, j) in common])
        C1, d1 = _rays(track_a.K[f], track_a.R[f], track_a.t[f], ua)
        C2, d2 = _rays(track_b.K[f + offset], track_b.R[f + offset], track_b.t[f + offset], ub)
        X, miss = _closest_points(C1, d1, C2, d2)
        ank = np.isin(joints, (15, 16))
        hip = np.isin(joints, (11, 12))
        out["frames"].append(f)
        out["miss_p50"].append(float(np.median(miss)))
        out["miss_p90"].append(float(np.percentile(miss, 90)))
        out["ankle_z"].append(float(np.median(X[ank, 2])) if ank.any() else np.nan)
        out["hip_z"].append(float(np.median(X[hip, 2])) if hip.any() else np.nan)
    return {kk: np.asarray(v) for kk, v in out.items()}


def refine_track(video_path, track, boxes_by_frame, *, frames=None, centre_frames: int = 12,
                 centre=None, log_every: int = 100):
    """The endzone camera track refined to its paint: the centre solved once over
    ``centre_frames`` frames that show the goal line (or ``centre`` given), then every frame's
    rotation and focal, each frame starting from its neighbour's refined camera and the
    deltas smoothed along the track. Returns ``(new track, stats dict)``."""
    import cv2

    from nfl_gsplat.calibration.cameras_io import CameraTrack

    cap = cv2.VideoCapture(str(video_path))
    all_frames = [int(f) for f in np.flatnonzero(track.conf > 0)] if frames is None else list(frames)
    imgs: dict = {}

    def image(f):
        if f not in imgs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            imgs[f] = im if ok else None
            if len(imgs) > 64:
                imgs.pop(next(iter(imgs)))
        return imgs[f]

    # --- the centre, from frames that show the goal line; frames the joint fit cannot
    # bring onto the paint (a mis-registered pan frame) are dropped and it is refitted
    seeds: dict = {}                                   # frame -> (K, R) solved with the centre
    if centre is None:
        cands = [int(x) for x in np.linspace(all_frames[0], all_frames[-1], centre_frames * 3)]
        pfs, K0s, R0s, used = [], [], [], []
        for f in cands:
            im = image(f)
            if im is None:
                continue
            pf = measure_frame(im, boxes_by_frame.get(f), track.K[f], track.R[f], track.t[f])
            if pf.goal_row is None or len(pf.seg_k) < 10 or len(pf.dashes) < 8:
                continue
            pfs.append(pf)
            K0s.append(track.K[f])
            R0s.append(track.R[f])
            used.append(f)
            if len(pfs) >= centre_frames:
                break
        if len(pfs) < 3:
            raise ValueError(f"only {len(pfs)} frames show the goal line, yard lines and hash marks together; "
                             "the centre cannot be solved")
        c0 = -track.R[all_frames[0]].T @ track.t[all_frames[0]]
        for _round in range(2):
            centre, per = fit_centre(pfs, K0s, R0s, c0)
            keep = []
            for i, pf in enumerate(pfs):
                K, R, t = camera_from(per[i], K0s[i], R0s[i], centre)
                sc = frame_score(K, R, t, pf)
                keep.append(sc[0] <= CENTRE_FRAME_MAX_PX and (not np.isfinite(sc[1]) or sc[1] <= CENTRE_FRAME_MAX_PX))
            _LOG.info("endzone mount centre %s -> %s from %d frames (%d on the paint)", np.round(c0, 1),
                      np.round(centre, 1), len(pfs), sum(keep))
            if all(keep) or sum(keep) < 3:
                break
            pfs = [p for p, k in zip(pfs, keep) if k]
            K0s = [p for p, k in zip(K0s, keep) if k]
            R0s = [p for p, k in zip(R0s, keep) if k]
            used = [p for p, k in zip(used, keep) if k]
        for i, f in enumerate(used):
            if keep[i]:
                K, R, _t = camera_from(per[i], K0s[i], R0s[i], centre)
                seeds[f] = (K, R)
    centre = np.asarray(centre, float)

    # --- per frame, outward from the seeded frames so the registration never travels far:
    # each frame starts from the nearest already-solved camera
    T = len(track.conf)
    Kn, Rn, tn = track.K.copy(), track.R.copy(), track.t.copy()
    rot = np.full((T, 3), np.nan)
    logf = np.full(T, np.nan)
    stats = {"frames": [], "before": [], "after": [], "dash": [], "goal": [], "n_seg": [], "n_dash": []}
    seed_frames = sorted(seeds) or [all_frames[len(all_frames) // 2]]
    order = sorted(all_frames, key=lambda f: (min(abs(f - s) for s in seed_frames), f))
    prev_cam: dict = dict(seeds)
    for i, f in enumerate(order):
        im = image(f)
        if im is None:
            continue
        near = [g for d in range(1, 8) for g in (f - d, f + d) if g in prev_cam]
        if f in seeds:
            K0, R0 = seeds[f]
        elif near:
            K0, R0 = prev_cam[near[0]]
        else:
            K0, R0 = track.K[f], track.R[f]
        t0 = -R0 @ centre
        # two registrations -- by the red end zone's edge and by the camera alone -- and the
        # one the paint agrees with wins (a mis-read boundary during the pan registered
        # every line five lines off: 500 px on play 1 frames 270-294)
        best = None
        for use_goal in (True, False):
            pf_c = measure_frame(im, boxes_by_frame.get(f), K0, R0, t0, use_goal=use_goal)
            res_c = fit_frame(pf_c, K0, R0, centre)
            if best is None or (np.nan_to_num(res_c.after_px, nan=1e9) < np.nan_to_num(best[1].after_px, nan=1e9)):
                best = (pf_c, res_c)
            if pf_c.goal_row is None:
                break                                  # the two registrations are the same
        pf, res = best
        step_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(res.R @ R0.T).as_rotvec()))
        good = (np.isfinite(res.after_px) and res.after_px <= ACCEPT_LINE_PX
                and (not np.isfinite(res.dash_px) or res.dash_px <= ACCEPT_DASH_PX)
                and step_deg <= ACCEPT_STEP_DEG and len(pf.seg_k) >= 8)
        if good:
            prev_cam[f] = (res.K, res.R)
            # the delta is measured against the EXPORTED camera so the smoothing is along the track
            rot[f] = Rotation.from_matrix(res.R @ track.R[f].T).as_rotvec()
            logf[f] = np.log(res.K[0, 0] / track.K[f, 0, 0])
        else:
            res = FrameFit(K0, R0, t0, res.before_px, res.before_px, res.dash_px, res.goal_px, False)
        stats["frames"].append(f)
        stats["before"].append(res.before_px)
        stats["after"].append(res.after_px)
        stats["dash"].append(res.dash_px)
        stats["goal"].append(res.goal_px)
        stats["n_seg"].append(len(pf.seg_k))
        stats["n_dash"].append(len(pf.dashes))
        if log_every and (i + 1) % log_every == 0:
            _LOG.info("endzone paint: %d/%d frames, line px %.1f -> %.1f", i + 1, len(order),
                      np.nanmedian(stats["before"]), np.nanmedian(stats["after"]))
    ok = np.isfinite(logf)
    if ok.sum() == 0:
        raise ValueError("no frame refined")
    rv_s, lf_s = smooth_deltas(rot[all_frames], logf[all_frames])
    for j, f in enumerate(all_frames):
        R = Rotation.from_rotvec(rv_s[j]).as_matrix() @ track.R[f]
        K = track.K[f].copy()
        K[0, 0] = track.K[f, 0, 0] * np.exp(lf_s[j])
        K[1, 1] = track.K[f, 1, 1] * np.exp(lf_s[j])
        Kn[f], Rn[f], tn[f] = K, R, -R @ centre
    stats["centre"] = centre
    stats["n_refined"] = int(ok.sum())
    new = CameraTrack(K=Kn, R=Rn, t=tn, conf=track.conf.copy(), width=track.width, height=track.height)
    return new, stats
