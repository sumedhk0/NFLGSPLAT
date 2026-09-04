"""Refine every frame's camera against the painted yard lines.

WHY. The paint solve gives good cameras on the frames it solves and the
export interpolates between them; on both plays the projected grid drifts to
70-160 px from the paint in the last quarter of the clip as the camera pans
out (calibration.grid_fit). Every metre a player is placed goes through
that camera, so the drift is jitter and slide in the render.

WHAT. The camera is on a tripod: its centre does not move. Per frame, the
rotation (3 parameters) and the focal length (1) are refined so the detected
white yard-line segments lie on the projected 5-yard lines, starting from
the exported camera, with a soft-L1 loss so a stray segment does not pull.
Each segment is assigned to its nearest projected line at the start of a
round and the assignment is redone once after convergence. A frame with too
few segments keeps its camera; a refinement that does not lower the
distance is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nfl_gsplat.calibration.grid_fit import (MIN_SEGMENTS, projected_lines,
                                             segment_distances_px)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

SOFT_L1_PX: float = 4.0
ROUNDS: int = 2
# The nearest projected line is the wrong line when the starting camera has
# drifted by about half a line spacing (play 1's tail, frame 655: the six
# segments went to lines 0,1,2,3,5,6 and the continuous fit chased that into
# 12-degree rotations at 64 px). The assignment is searched over these
# whole-line offsets and the lowest final residual wins.
ASSIGN_OFFSETS = (-1, 0, 1)
# A pencil of near-parallel lines is fit equally well by a rotation about
# their direction traded against the focal length: frame 630 of play 1
# reached 2 px by rotating 7.5 degrees and zooming 40%. A refinement that
# moves the camera more than this is that degeneracy, not a correction.
# Measured degenerate fits: 4.5-7.9 degrees, 12-40% zoom; a real correction
# on play 1 was under 1 degree and 5%.
MAX_APPLIED_ROT_DEG: float = 3.0
MAX_APPLIED_FOCAL: float = 0.20
# Priors that make the fit take the SMALLEST camera motion along that valley:
# one degree of rotation and ten percent of focal length each cost as much as
# these many pixels of line residual. Play 2 without them: every frame
# "wanted" 4.5-7 degrees and up to 35% zoom to go from 10 px to 3.
PRIOR_PX_PER_DEG: float = 20.0
PRIOR_PX_PER_10PCT_FOCAL: float = 20.0
MAX_ROT_DEG: float = 6.0        # a frame's camera is never rotated more than this
MAX_FOCAL_CHANGE: float = 0.40  # the lens may change this much: the camera ZOOMS OUT at
                                # the end of a play (play 1 frames 600-655 needed -15% and
                                # hit a 15% bound at 44 px); rotation was never the limit
# The camera pans smoothly; refined frame by frame the rotation about the
# yard lines' own direction wanders (play 1: frame-to-frame rotation p95
# 0.03 -> 2.3 degrees). The per-frame deltas are smoothed along the track.
SMOOTH_FRAMES: int = 31


@dataclass(frozen=True)
class RefineResult:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    before_px: float
    after_px: float
    n_segments: int
    applied: bool


def _camera_from(params, K0, R0, centre):
    """Rotation ``exp(w) R0`` and focal ``f0 exp(s)``; the centre stays."""
    w, s = params[:3], params[3]
    R = Rotation.from_rotvec(w).as_matrix() @ R0
    K = K0.copy()
    K[0, 0] = K0[0, 0] * np.exp(s)
    K[1, 1] = K0[1, 1] * np.exp(s)
    t = -R @ centre
    return K, R, t


def _fit_assignment(p0, p1, assign, K0, R0, centre, n_lines, *, focal):
    """Continuous fit for one fixed segment-to-line assignment."""
    n = p0.shape[0]

    prior_rot = PRIOR_PX_PER_DEG * np.degrees(1.0)          # px per radian
    prior_f = PRIOR_PX_PER_10PCT_FOCAL / np.log(1.1)         # px per unit log-focal

    def resid(x):
        Kx, Rx, tx = _camera_from(x, K0, R0, centre)
        L = projected_lines(Kx, Rx, tx)
        if len(L) != n_lines:
            return np.full(2 * n + 4, 1e3)
        Ls = L[assign]
        return np.concatenate([np.einsum("ij,ij->i", p0, Ls), np.einsum("ij,ij->i", p1, Ls),
                               prior_rot * x[:3], [prior_f * x[3]]])

    lo = np.array([-np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 - MAX_FOCAL_CHANGE)])
    hi = np.array([np.radians(MAX_ROT_DEG)] * 3 + [np.log(1 + MAX_FOCAL_CHANGE)])
    if not focal:
        lo[3], hi[3] = -1e-9, 1e-9
    sol = least_squares(resid, np.zeros(4), loss="soft_l1", f_scale=SOFT_L1_PX, bounds=(lo, hi),
                        max_nfev=200)
    return sol.x


def refine_frame(segments, K, R, t, *, focal: bool = True) -> RefineResult:
    """Refine one camera to ``segments`` (YardLineSeg list) -- see module doc."""
    K0 = np.asarray(K, float)
    R0 = np.asarray(R, float)
    t0 = np.asarray(t, float).reshape(3)
    centre = -R0.T @ t0
    n = len(segments)
    before = float(np.median(segment_distances_px(segments, projected_lines(K0, R0, t0)))) \
        if n else float("nan")
    if n < MIN_SEGMENTS:
        return RefineResult(K0, R0, t0, before, before, n, False)

    p0 = np.asarray([[s.p0[0], s.p0[1], 1.0] for s in segments])
    p1 = np.asarray([[s.p1[0], s.p1[1], 1.0] for s in segments])
    mids = 0.5 * (p0 + p1)
    mids[:, 2] = 1.0
    lines0 = projected_lines(K0, R0, t0)
    if len(lines0) == 0:
        return RefineResult(K0, R0, t0, before, before, n, False)
    nearest = np.argmin(np.abs(mids @ lines0.T), axis=1)
    best = (before, K0, R0, t0)
    for offset in ASSIGN_OFFSETS:
        assign = nearest + offset
        if assign.min() < 0 or assign.max() >= len(lines0):
            continue
        params = np.zeros(4)
        for _round in range(ROUNDS):
            params = _fit_assignment(p0, p1, assign, K0, R0, centre, len(lines0), focal=focal)
            Kc, Rc, tc = _camera_from(params, K0, R0, centre)
            Lc = projected_lines(Kc, Rc, tc)
            if len(Lc) != len(lines0):
                break
            assign = np.argmin(np.abs(mids @ Lc.T), axis=1)     # re-assign, then refit once
        Kr, Rr, tr = _camera_from(params, K0, R0, centre)
        after = float(np.median(segment_distances_px(segments, projected_lines(Kr, Rr, tr))))
        if np.isfinite(after) and after < best[0]:
            best = (after, Kr, Rr, tr)
    after, Kr, Rr, tr = best
    if after >= before:
        return RefineResult(K0, R0, t0, before, before, n, False)
    moved = np.degrees(np.linalg.norm(Rotation.from_matrix(Rr @ R0.T).as_rotvec()))
    zoomed = abs(np.log(Kr[0, 0] / K0[0, 0]))
    if moved > MAX_APPLIED_ROT_DEG or zoomed > np.log(1 + MAX_APPLIED_FOCAL):
        return RefineResult(K0, R0, t0, before, before, n, False)      # degenerate
    return RefineResult(Kr, Rr, tr, before, after, n, True)


def _smooth_deltas(rotvecs, log_f, *, window: int):
    """Zero-phase moving average of per-frame deltas; NaN rows (frames that
    were not refined) are interpolated from their neighbours first."""
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


def refine_track(video_path, track, *, cfg=None, player_boxes_by_frame=None, log_every: int = 100,
                 smooth_frames: int = SMOOTH_FRAMES):
    """Every frame of a CameraTrack refined to the paint, the per-frame
    rotation and focal deltas smoothed along the track (the camera pans
    smoothly; the lines do not pin every rotation). Frames with too few
    segments take the smoothed delta of their neighbours. Returns
    ``(CameraTrack, per-frame after_px)``."""
    import cv2

    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines
    from nfl_gsplat.calibration.grid_fit import GRID_VERTICAL_DEG

    cfg = cfg or FieldDetectConfig(vertical_deg=GRID_VERTICAL_DEG)
    K = track.K.copy()
    R = track.R.copy()
    t = track.t.copy()
    after = np.full(len(K), np.nan)
    cap = cv2.VideoCapture(str(video_path))
    frames = [int(f) for f in np.flatnonzero(track.conf > 0)]
    deltas = np.full((len(frames), 3), np.nan)
    log_f = np.full(len(frames), np.nan)
    segs_by_frame = {}
    n_applied = 0
    for i, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        boxes = None if player_boxes_by_frame is None else player_boxes_by_frame.get(f)
        segs = detect_lines(img, cfg, boxes)
        segs_by_frame[f] = segs
        res = refine_frame(segs, K[f], R[f], t[f])
        if res.applied:
            deltas[i] = Rotation.from_matrix(res.R @ R[f].T).as_rotvec()
            log_f[i] = np.log(res.K[0, 0] / K[f][0, 0])
            n_applied += 1
        if log_every and i % log_every == 0:
            _LOG.info("refine: frame %d  %.1f -> %.1f px over %d segments", f, res.before_px,
                      res.after_px, res.n_segments)
    cap.release()
    sm_rv, sm_lf = _smooth_deltas(deltas, log_f, window=smooth_frames)
    for i, f in enumerate(frames):
        centre = -R[f].T @ t[f]
        R[f] = Rotation.from_rotvec(sm_rv[i]).as_matrix() @ R[f]
        K[f][0, 0] *= np.exp(sm_lf[i])
        K[f][1, 1] *= np.exp(sm_lf[i])
        t[f] = -R[f] @ centre
        segs = segs_by_frame.get(f, [])
        if len(segs) >= MIN_SEGMENTS:
            after[f] = float(np.median(segment_distances_px(segs, projected_lines(K[f], R[f], t[f]))))
    _LOG.info("refine: %d of %d frames refined; smoothed over %d frames; median after %.1f px",
              n_applied, len(frames), smooth_frames, float(np.nanmedian(after)))
    return CameraTrack(K=K, R=R, t=t, conf=track.conf.copy(), width=track.width,
                       height=track.height), after
