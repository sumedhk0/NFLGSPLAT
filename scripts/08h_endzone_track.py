#!/usr/bin/env python
"""The endzone camera track from the footage's own motion.

WHY. The endzone camera is solved on a handful of frames (14 on play 2) and
08b interpolates between them. Measured 2026-09-05 on play 2: a lineman who
is pixel-static in the endzone view before the snap (1 px) has a ground
point that wanders 1.7-2.6 m, and a fixed endzone pixel's ground point
swings 4 m in one second while the sideline's holds to 0.1 m. The endzone
track was inventing motion between noisy anchors (their focals jump 5 %
between neighbours), so the two cameras never agreed on where a player
stood, which is why per-frame pairing was a coin flip and 53 % of the ids
seen by both cameras wore different kits in the two views.

WHAT. A tripod camera pans, tilts and zooms but never translates, so its
frames are related EXACTLY by homographies (calibration.endzone_mosaic,
measured 0.2-0.6 px on real footage). Every frame is registered into one
reference frame with the players masked; every EXISTING endzone camera
frame then implies a reference camera (K_ref R_ref = H_t K_t R_t), and the
robust average over all of them is the absolute pose (their per-frame noise
averages down), while the motion between frames is the footage's. The
principal point is held (endzone_mosaic.propagate's lesson); the centre is
the existing track's median.

RULERS (printed, and the second one must improve or the run refuses to
write): (1) the old track's per-frame deviation from the new one (rotation
degrees, focal ratio) -- expected at the anchors' own scatter; (2) the
players: over frames where both cameras are solved, the median distance
from each sideline player's ground point to the nearest endzone one
(kit-consistent), before and after. Writes cameras.npz in place with the
old track kept as cameras_endzone_interp.npz.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track, write_camera_track
from nfl_gsplat.errors import CalibrationError

PAD_FRAMES = 30            # propagate this far beyond the old track's solved span
DECODE_SCALE = 0.5         # register at half resolution (features are plentiful; memory is not)
ASPECT_RANGE = (0.95, 1.05)
FOCAL_BAND = (0.5, 2.0)    # a propagated focal must sit within this of the reference's


def _implied_reference(H_t, K_t, R_t, cx, cy):
    """(focal, R) of the reference camera implied by frame t's camera and its
    homography INTO the reference: K_ref R_ref = H_t K_t R_t, principal point held."""
    M = H_t @ (K_t @ R_t)
    K_pp = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    n = np.linalg.solve(K_pp, M)
    n = n / np.linalg.norm(n[2])
    fx, fy = np.linalg.norm(n[0]), np.linalg.norm(n[1])
    R = np.stack([n[0] / fx, n[1] / fy, n[2]])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return float(0.5 * (fx + fy)), R, fx / fy


def _chordal_mean(Rs):
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _decode(video, frames, scale):
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out[int(f)] = img
    cap.release()
    return out


def _boxes_by_frame(tracks_parquet, cam, scale):
    import pandas as pd

    if tracks_parquet is None or not Path(tracks_parquet).exists():
        return {}
    df = pd.read_parquet(tracks_parquet)
    df = df[df["cam"] == cam]
    out = {}
    for f, g in df.groupby("frame"):
        out[int(f)] = (g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float) * scale).tolist()
    return out


def player_ruler(cams, tracks_parquet, *, kit_margin=0.3):
    """Median distance from each sideline player's ground point to the nearest
    endzone one (same kit where both known), over frames both cameras solve."""
    import pandas as pd

    from nfl_gsplat.calibration.from_players import feet_of
    from nfl_gsplat.calibration.joint_views import ground_points

    df = pd.read_parquet(tracks_parquet)
    B = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    if "kit_margin" in df:
        m = df["kit_margin"].to_numpy(float)
        df["kit"] = np.where(np.isfinite(m) & (np.abs(m) >= kit_margin), (m > 0).astype(int), -1)
    else:
        df["kit"] = -1
    ts, te = cams["sideline"], cams["endzone"]
    nn = []
    for f, g in df.groupby("frame"):
        f = int(f)
        if f >= len(ts.conf) or ts.conf[f] <= 0 or te.conf[f] <= 0:
            continue
        gs, ge = g[g["cam"] == "sideline"], g[g["cam"] == "endzone"]
        if len(gs) < 3 or len(ge) < 3:
            continue
        ps = ground_points((ts.K[f], ts.R[f], ts.t[f]), feet_of(gs[B].to_numpy()))
        pe = ground_points((te.K[f], te.R[f], te.t[f]), feet_of(ge[B].to_numpy()))
        ks, ke = gs["kit"].to_numpy(), ge["kit"].to_numpy()
        oks = np.isfinite(ps).all(1) & (np.abs(ps[:, 0]) <= 56) & (np.abs(ps[:, 1]) <= 26)
        oke = np.isfinite(pe).all(1) & (np.abs(pe[:, 0]) <= 56) & (np.abs(pe[:, 1]) <= 26)
        ps, ks, pe, ke = ps[oks], ks[oks], pe[oke], ke[oke]
        if len(ps) == 0 or len(pe) == 0:
            continue
        d = np.linalg.norm(ps[:, None] - pe[None], axis=2)
        bad = (ks[:, None] >= 0) & (ke[None, :] >= 0) & (ks[:, None] != ke[None, :])
        d = np.where(bad, np.inf, d)
        nn.extend(d.min(1)[np.isfinite(d.min(1))].tolist())
    nn = np.asarray(nn)
    return (float(np.median(nn)), float(np.mean(nn < 1.0)), len(nn)) if len(nn) else (float("nan"), 0.0, 0)


def rot_focal_from_homography(H, f_ref, R_ref, cx, cy):
    """The frame camera (R_t, f_t) whose rays agree with the reference's
    through ``H`` (frame pixels INTO the reference), principal point held.

    A pixel grid is mapped through H into the reference, lifted to world
    directions by the reference camera, and (rotation, log focal) are fitted
    so those directions project back onto the grid. Measured 2026-09-06: the
    matrix route (K_t R_t = H^-1 K_ref R_ref, RQ-style split with an SVD
    polish) turned a 0.4 px homography step into 0.18 deg of rotation --
    0.9 m on the ground at 300 m -- because the polish, not the translation,
    set the rotation; the ray fit holds a pixel-static ground point to 1 cm
    over the same frames. Returns ``(R_t, f_t, rms_px)``."""
    import cv2
    from scipy.optimize import least_squares

    gu, gv = np.meshgrid(np.linspace(100, 2 * cx - 100, 7), np.linspace(100, 2 * cy - 100, 5))
    pts = np.column_stack([gu.ravel(), gv.ravel(), np.ones(gu.size)])
    q = (H @ pts.T).T
    q = q[:, :2] / q[:, 2:3]
    rays = np.column_stack([(q[:, 0] - cx) / f_ref, (q[:, 1] - cy) / f_ref, np.ones(len(q))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    world = (R_ref.T @ rays.T).T

    def resid(x):
        Rt = cv2.Rodrigues(x[:3])[0] @ R_ref
        ft = f_ref * np.exp(x[3])
        cam = (Rt @ world.T).T
        proj = np.column_stack([cam[:, 0] / cam[:, 2] * ft + cx, cam[:, 1] / cam[:, 2] * ft + cy])
        return (proj - pts[:, :2]).ravel()

    r = least_squares(resid, np.zeros(4), method="lm")
    Rt = cv2.Rodrigues(r.x[:3])[0] @ R_ref
    return Rt, float(f_ref * np.exp(r.x[3])), float(np.sqrt(np.mean(r.fun ** 2)))


def _propagate(H_full, f_ref, R_ref, C, cx, cy, n, *, band=FOCAL_BAND, max_rms_px: float = 3.0):
    """Per-frame (K, R, t, conf) from a reference camera and the homographies
    INTO the reference, by ``rot_focal_from_homography`` per frame; a frame
    whose fit is worse than ``max_rms_px`` or whose focal leaves ``band`` is
    dropped (conf 0)."""
    K = np.zeros((n, 3, 3))
    R = np.zeros((n, 3, 3))
    t = np.zeros((n, 3))
    conf = np.zeros(n)
    dropped = 0
    for f, H in H_full.items():
        if not (0 <= f < n):
            continue
        Rf, ff, rms = rot_focal_from_homography(H, f_ref, R_ref, cx, cy)
        if not np.isfinite(ff) or rms > max_rms_px or not (band[0] <= ff / f_ref <= band[1]):
            dropped += 1
            continue
        K[f] = np.array([[ff, 0.0, cx], [0.0, ff, cy], [0.0, 0.0, 1.0]])
        R[f] = Rf
        t[f] = -Rf @ C
        conf[f] = 1.0
    return K, R, t, conf, dropped


def anchored_chain(feats, ref, lo, hi, f_ref, R_ref, cx, cy, *, scale: float, every: int = 10,
                   max_rms_px: float = 2.0, min_inliers: int = 12):
    """``{frame: H}`` into the reference: a DIRECT link every ``every`` frames
    where its ray fit is within ``max_rms_px`` (an anchor), consecutive steps
    composed outward from the nearest anchor elsewhere. Measured on play 1
    (2026-09-07): single steps fit at 0.1-1 px everywhere, but 200 composed
    steps through the second half drifted to 7-90 px; direct links hold at
    0.2-2 px there except across a 60-frame span (475-535) where the
    periodic yard lines alias them (7-21 px) -- which is where the chain
    carries. Returns ``(H_by, anchors, inliers)`` at the feature scale."""
    from nfl_gsplat.calibration.endzone_mosaic import _homography

    D = np.diag([scale, scale, 1.0])
    Di = np.linalg.inv(D)
    anchors = {ref: np.eye(3)}
    for f in range(lo, hi + 1, every):
        if f == ref or f not in feats:
            continue
        hom, n_inl = _homography(feats[ref], feats[f], min_inliers)
        if hom is None:
            continue
        _R, _f, rms = rot_focal_from_homography(Di @ hom @ D, f_ref, R_ref, cx, cy)
        if rms <= max_rms_px:
            anchors[f] = hom
    H_by = dict(anchors)
    inliers = {}
    order = sorted(anchors)

    def chain(start, stop, step):
        """{frame: H into the reference} by consecutive steps from an anchor."""
        out = {}
        prev = start
        for f in range(start + step, stop, step):
            if f not in feats:
                continue
            hom, n_inl = _homography(feats[prev], feats[f], min_inliers)
            if hom is None:
                break
            out[f] = (H_by[prev] if prev in H_by and prev == start else out[prev]) @ hom
            inliers[f] = max(inliers.get(f, 0), n_inl)
            prev = f
        return out

    for i, a in enumerate(order):
        nxt = order[i + 1] if i + 1 < len(order) else None
        fwd = chain(a, nxt if nxt is not None else hi + 1, +1)
        if nxt is None:
            H_by.update(fwd)
            continue
        bwd = chain(nxt, a, -1)
        for f in range(a + 1, nxt):
            if f in fwd and f in bwd:
                w = (f - a) / (nxt - a)
                H_by[f] = _blend(fwd[f], bwd[f], w)
            elif f in fwd:
                H_by[f] = fwd[f]
            elif f in bwd:
                H_by[f] = bwd[f]
    H_by.update(chain(order[0], lo - 1, -1))
    return H_by, sorted(anchors), inliers


def _blend(Ha, Hb, w):
    """Homography between two estimates of the same frame's link, ``w`` the
    weight of the second (0 at the first anchor, 1 at the next): both are
    normalised so the blend is well posed near the identity."""
    Ha = Ha / Ha[2, 2]
    Hb = Hb / Hb[2, 2]
    return (1.0 - w) * Ha + w * Hb


def chain_homographies(imgs, feats, ref, lo, hi, *, min_inliers: int = 12):
    """``{frame: H}`` mapping each frame INTO the reference by composing
    CONSECUTIVE-frame homographies outward from ``ref`` (one step is a few
    pixels: no aliasing on the periodic yard lines, which broke the direct
    registration at a 100-frame gap on play 2). Returns ``(H_by, inliers)``."""
    from nfl_gsplat.calibration.endzone_mosaic import _homography

    H_by = {ref: np.eye(3)}
    inl = {}
    for direction in (+1, -1):
        prev = ref
        f = ref + direction
        while lo <= f <= hi:
            if f not in feats:
                f += direction
                continue
            hom, n_inl = _homography(feats[prev], feats[f], min_inliers)
            if hom is None:
                break
            H_by[f] = H_by[prev] @ hom
            inl[f] = n_inl
            prev = f
            f += direction
    return H_by, inl


def _player_frames(tracks_parquet, side, *, kit_margin=0.3):
    """Per frame: the sideline players' on-field ground points (fixed) and the
    endzone boxes' foot pixels with kits -- the fit's data."""
    import pandas as pd

    from nfl_gsplat.calibration.from_players import feet_of
    from nfl_gsplat.calibration.joint_views import ground_points

    df = pd.read_parquet(tracks_parquet)
    B = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    if "kit_margin" in df:
        m = df["kit_margin"].to_numpy(float)
        df["kit"] = np.where(np.isfinite(m) & (np.abs(m) >= kit_margin), (m > 0).astype(int), -1)
    else:
        df["kit"] = -1
    out = {}
    for f, g in df.groupby("frame"):
        f = int(f)
        if f >= len(side.conf) or side.conf[f] <= 0:
            continue
        gs, ge = g[g["cam"] == "sideline"], g[g["cam"] == "endzone"]
        if len(gs) < 3 or len(ge) < 3:
            continue
        ps = ground_points((side.K[f], side.R[f], side.t[f]), feet_of(gs[B].to_numpy()))
        ok = np.isfinite(ps).all(1) & (np.abs(ps[:, 0]) <= 56) & (np.abs(ps[:, 1]) <= 26)
        if ok.sum() < 3:
            continue
        out[f] = (ps[ok], gs["kit"].to_numpy()[ok], feet_of(ge[B].to_numpy()), ge["kit"].to_numpy())
    return out


def _ruler(frames, K, R, t, conf):
    """Median over sideline players of the kit-consistent nearest endzone ground point."""
    from nfl_gsplat.calibration.joint_views import ground_points

    nn = []
    for f, (ps, ks, fe, ke) in frames.items():
        if f >= len(conf) or conf[f] <= 0:
            continue
        pe = ground_points((K[f], R[f], t[f]), fe)
        ok = np.isfinite(pe).all(1) & (np.abs(pe[:, 0]) <= 56) & (np.abs(pe[:, 1]) <= 26)
        pe_, ke_ = pe[ok], ke[ok]
        if len(pe_) == 0:
            nn.extend([5.0] * len(ps))          # nothing on the field: every player unpaired
            continue
        d = np.linalg.norm(ps[:, None] - pe_[None], axis=2)
        bad = (ks[:, None] >= 0) & (ke_[None, :] >= 0) & (ks[:, None] != ke_[None, :])
        d = np.where(bad, np.inf, d).min(1)
        nn.extend(np.minimum(d, 5.0).tolist())    # capped: an unpaired player costs 5 m, no more
    nn = np.asarray(nn)
    return (float(np.median(nn)), float(np.mean(nn < 1.0)), len(nn)) if len(nn) else (float("inf"), 0.0, 0)


def fit_reference_on_players(frames, H_full, f0, R0, C, cx, cy, n):
    """The reference (focal, rotation) that brings the endzone's players onto
    the sideline's, with the mosaic fixed: four parameters (log focal scale
    and a rotation vector applied to R0), Nelder-Mead on the capped median
    nearest distance. The sideline is the trustworthy camera (refined to the
    paint every frame; a static pixel's ground point holds to 0.1 m), the
    endzone anchors are not (transported to one frame they scatter 3.7 deg)."""
    import cv2
    from scipy.optimize import minimize

    C = np.asarray(C, float)
    across = np.array([0.0, 1.0, 0.0]) if abs(C[0]) >= abs(C[1]) else np.array([1.0, 0.0, 0.0])

    def unpack(x):
        f = f0 * float(np.exp(x[0]))
        Rd = cv2.Rodrigues(np.asarray(x[1:4], np.float64))[0]
        return f, Rd @ R0, C + float(x[4]) * across

    def cost(x):
        f, R, Cx = unpack(x)
        K, Rr, t, conf, _ = _propagate(H_full, f, R, Cx, cx, cy, n)
        return _ruler(frames, K, Rr, t, conf)[0]

    # Restarts: one Nelder-Mead from the anchors' mean converged to a worse
    # place on play 1's fresh calibration (1.68 -> 1.77 m) after finding
    # 1.19 -> 1.03 m on the accumulated one; the objective has basins. The
    # mount's across-field offset is a prior (y = 0 or 4 on this game), so
    # it is the fifth parameter.
    x0 = np.zeros(5)
    c0 = cost(x0)
    rng = np.random.default_rng(0)
    starts = [x0] + [np.array([rng.normal(0, 0.06), *np.radians(rng.normal(0, 1.0, 3)), rng.normal(0, 3.0)])
                     for _ in range(8)]
    best = None
    nfev = 0
    for xs in starts:
        res = minimize(cost, xs, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3, "maxfev": 300})
        nfev += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    f, R, Cb = unpack(best.x)
    return f, R, c0, float(best.fun), nfev, Cb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--cam", default="endzone")
    ap.add_argument("--video", type=Path, default=None, help="default <play-dir>/<cam>.mp4")
    ap.add_argument("--tracks", type=Path, default=None, help="boxes to mask (default <play-dir>/tracks.parquet)")
    ap.add_argument("--scale", type=float, default=DECODE_SCALE)
    ap.add_argument("--pad", type=int, default=PAD_FRAMES)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--no-write", action="store_true", help="measure only")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a cameras.npz whose endzone camera at the reference frame seeds the fit (focal, "
                         "rotation, centre) instead of the old track's transported mean: the fit is "
                         "start-dependent (play 1 fresh: 1.82 m from the anchors' mean, 1.03 m earlier)")
    args = ap.parse_args()
    play = args.play_dir
    video = args.video or play / f"{args.cam}.mp4"
    tracks = args.tracks or play / "tracks.parquet"
    cams = load_camera_track(play / "cameras.npz")
    old = cams[args.cam]
    n = len(old.conf)
    solved = np.flatnonzero(old.conf > 0)
    if len(solved) < 3:
        raise CalibrationError(f"{args.cam}: only {len(solved)} solved frames in cameras.npz")
    lo, hi = max(0, int(solved[0]) - args.pad), min(n - 1, int(solved[-1]) + args.pad)
    frames = list(range(lo, hi + 1))
    print(f"{args.cam}: old track solves {len(solved)} of {n} frames ({solved[0]}..{solved[-1]}); "
          f"registering {len(frames)} frames ({lo}..{hi}) at scale {args.scale}")
    imgs = _decode(video, frames, args.scale)
    if len(imgs) < 0.9 * len(frames):
        raise CalibrationError(f"decoded only {len(imgs)} of {len(frames)} frames from {video}")
    boxes = _boxes_by_frame(tracks, args.cam, args.scale)
    ref = int(solved[len(solved) // 2])
    if ref not in imgs:
        ref = min(imgs, key=lambda f: abs(f - ref))
    from nfl_gsplat.calibration.endzone_mosaic import _features, keep_mask
    feats = {f: _features(img, keep_mask(img.shape, boxes.get(f))) for f, img in imgs.items()}
    cx0, cy0 = old.K[ref][0, 2], old.K[ref][1, 2]
    H_by, anchors, inliers = anchored_chain(feats, ref, lo, hi, old.K[ref][0, 0], old.R[ref], cx0, cy0,
                                            scale=args.scale, min_inliers=args.min_inliers)
    print(f"anchored chain: {len(anchors)} direct anchors to reference {ref} "
          f"(gaps up to {max(np.diff(anchors)) if len(anchors) > 1 else 0} frames), {len(H_by)} frames")
    # half-resolution homographies to full resolution: p_half = D p_full
    D = np.diag([args.scale, args.scale, 1.0])
    Dinv = np.linalg.inv(D)
    H_full = {f: Dinv @ H @ D for f, H in H_by.items()}
    inl = np.array([inliers[f] for f in H_full if f in inliers])
    print(f"registered {len(H_full)} of {len(imgs)} frames to reference {ref}; chain-step inliers p10/p50 "
          f"{np.percentile(inl, 10):.0f}/{np.median(inl):.0f}")
    cx, cy = old.K[ref][0, 2], old.K[ref][1, 2]
    # the reference camera implied by every existing solved frame
    implied = []
    for f in solved:
        f = int(f)
        if f not in H_full:
            continue
        focal, R, aspect = _implied_reference(H_full[f], old.K[f], old.R[f], cx, cy)
        if ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
            implied.append((f, focal, R))
    if len(implied) < 3:
        raise CalibrationError(f"only {len(implied)} solved frames imply a reference camera")
    focals = np.array([i[1] for i in implied])
    f_ref = float(np.median(focals))
    keep = np.abs(focals / f_ref - 1.0) <= 0.25
    R_ref = _chordal_mean([i[2] for i, k in zip(implied, keep) if k])
    dev = np.array([_angle_deg(R_ref, i[2]) for i in implied])
    centres = np.stack([-old.R[int(f)].T @ old.t[int(f)] for f in solved])
    C = np.median(centres, axis=0)
    if args.init_from is not None:
        seed = load_camera_track(args.init_from)[args.cam]
        sf = ref if seed.conf[ref] > 0 else int(np.flatnonzero(seed.conf > 0)[np.argmin(np.abs(np.flatnonzero(seed.conf > 0) - ref))])
        f_ref, R_ref = float(seed.K[sf][0, 0]), seed.R[sf].copy()
        C = -seed.R[sf].T @ seed.t[sf]
        print(f"fit seeded from {args.init_from.name} at frame {sf}: focal {f_ref:.0f}, centre {np.round(C, 1)}")
    print(f"reference camera from {int(keep.sum())} of {len(implied)} frames: focal {f_ref:.0f} px "
          f"(implied focals spread p10/p90 {np.percentile(focals / f_ref, 10):.3f}/{np.percentile(focals / f_ref, 90):.3f}), "
          f"rotation scatter p50/p90 {np.median(dev):.2f}/{np.percentile(dev, 90):.2f} deg; centre {np.round(C, 1)} "
          f"(old track's centre spread {np.linalg.norm(centres - C, axis=1).max():.2f} m)")
    # The absolute pose from the players: fit (focal, rotation) of the reference
    # camera on the sideline's ground points with the mosaic fixed.
    frames_pl = _player_frames(tracks, cams["sideline"])
    print(f"fit data: {len(frames_pl)} frames with sideline players and endzone boxes")
    f_fit, R_fit, c_mean, c_fit, nfev, C_fit = fit_reference_on_players(frames_pl, H_full, f_ref, R_ref, C, cx, cy, n)
    print(f"reference fit on the players: capped median nearest distance {c_mean:.2f} m (anchors' mean) -> "
          f"{c_fit:.2f} m in {nfev} evaluations over 9 starts; focal {f_ref:.0f} -> {f_fit:.0f} px, rotation moved "
          f"{_angle_deg(R_ref, R_fit):.2f} deg, centre moved {np.linalg.norm(C_fit - C):.1f} m across")
    C = C_fit
    K, R, t, conf, dropped = _propagate(H_full, f_fit, R_fit, C, cx, cy, n)
    new = CameraTrack(K=K, R=R, t=t, conf=conf, width=old.width, height=old.height)
    print(f"propagated {int(conf.sum())} frames ({dropped} dropped by the aspect/focal gates)")
    # ruler 1: the old track against the new one where both exist
    both = [int(f) for f in solved if conf[int(f)] > 0]
    ang = np.array([_angle_deg(old.R[f], R[f]) for f in both])
    fr = np.array([old.K[f][0, 0] / K[f][0, 0] for f in both])
    print(f"old track vs new over {len(both)} frames: rotation p50/p90 {np.median(ang):.2f}/{np.percentile(ang, 90):.2f} deg, "
          f"focal ratio p10/p50/p90 {np.percentile(fr, 10):.3f}/{np.median(fr):.3f}/{np.percentile(fr, 90):.3f}")
    # ruler 2: the players, uncapped, old track vs new (the fit's own objective is capped)
    before = player_ruler(cams, tracks)
    after = player_ruler({**cams, args.cam: new}, tracks)
    print(f"players: sideline->{args.cam} nearest ground distance median {before[0]:.2f} m -> {after[0]:.2f} m; "
          f"share within 1 m {before[1]:.2f} -> {after[1]:.2f} (n {before[2]} -> {after[2]})")
    if not (after[0] < before[0]):
        raise CalibrationError("the footage-driven track does not bring the cameras' players closer; not written")
    if args.no_write:
        return
    backup = play / f"cameras_{args.cam}_interp.npz"
    if not backup.exists():
        shutil.copy2(play / "cameras.npz", backup)
    out = dict(cams)
    out[args.cam] = new
    fps = float(np.load(play / "cameras.npz")["fps"]) if "fps" in np.load(play / "cameras.npz") else 59.94
    write_camera_track(play / "cameras.npz", out, fps=fps)
    print(f"wrote {play / 'cameras.npz'} ({args.cam} from the footage; the old track is {backup.name})")


if __name__ == "__main__":
    main()
