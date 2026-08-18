"""Register endzone frames into one reference frame and accumulate static paint.

The endzone camera is a fixed tripod: it pans/tilts/zooms but never translates.
With no translational parallax the frames are related EXACTLY by homographies
(measured 0.2-0.6 px on real footage), which is what makes this far more precise
than player correspondences (87 px at the true camera).

Registration is DIRECT frame->reference wherever possible. Direct links were
measured to hold across a whole play (36-47 inliers at a 420-frame gap), and
they cannot accumulate drift the way sequential chaining does (chaining drifted
6 px -> 282 px on real footage). A local-chain fallback covers frames whose
direct link is too weak.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

_RATIO = 0.78          # Lowe ratio for descriptor matching
_RANSAC_PX = 2.5


def keep_mask(shape, boxes, pad: int = 10) -> np.ndarray:
    """255 where features may be taken (field), 0 over players.

    Moving players would contribute non-rigid matches and break the
    pure-rotation model, so they are excluded before feature detection."""
    m = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in boxes or []:
        a = max(0, int(x1) - pad)
        b = max(0, int(y1) - pad)
        c = min(shape[1], int(x2) + pad)
        d = min(shape[0], int(y2) + pad)
        m[b:d, a:c] = 0
    return m


def _detector():
    return cv2.SIFT_create(nfeatures=4000)


def _features(img_bgr, mask=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return _detector().detectAndCompute(gray, mask)


def _homography(fa, fb, min_inliers):
    """Homography mapping image A's pixels into image B. Returns (H, inliers)."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < _RATIO * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, msk = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_PX, maxIters=8000)
    if H is None or msk is None:
        return None, 0
    return H, int(msk.sum())


def register_to_reference(frames, *, ref_idx, min_inliers: int = 25,
                          boxes_by_frame=None,
                          max_unregistered_frac: float = 0.25):
    """{frame: H mapping that frame's pixels INTO the reference}, {frame: inliers}.

    Direct link first; if it is too weak, fall back to composing through the
    nearest DIRECTLY-registered frame — exactly one hop, so drift stays
    bounded by construction rather than by how many pending frames happen to
    register first.

    ``boxes_by_frame``, if given, excludes players from feature detection
    (via :func:`keep_mask`) so moving players never contribute non-rigid
    matches that would break the pure-rotation/zoom model.

    Frames that still cannot be registered after both passes are common at
    the TAIL of a play (the broadcast camera swings away, wipes to a
    graphic, or loses the field once the play ends) — that is normal
    footage, not a defect, and it contributes nothing to the mosaic either
    way. Up to ``max_unregistered_frac`` of the sampled frames are therefore
    dropped (no entry in the returned dict) with a loud warning rather than
    aborting the whole calibration; only when MORE than that fraction fails
    does this raise, since at that point too little of the sample is usable
    to trust the result."""
    if ref_idx not in frames:
        raise CalibrationError(
            f"endzone mosaic: reference frame {ref_idx} not among the sampled "
            "frames — pick a reference that was actually decoded.")
    boxes_by_frame = boxes_by_frame or {}
    feats = {i: _features(img, keep_mask(img.shape, boxes_by_frame.get(i)))
             for i, img in frames.items()}
    H_by = {ref_idx: np.eye(3)}
    inl_by = {ref_idx: len(feats[ref_idx][0])}
    pending = []
    for i in sorted(frames):
        if i == ref_idx:
            continue
        H, n = _homography(feats[i], feats[ref_idx], min_inliers)
        if H is not None and n >= min_inliers:
            H_by[i], inl_by[i] = H, n
        else:
            pending.append(i)

    # Fallback: compose through the nearest DIRECTLY-registered frame. The
    # candidate set is frozen before the pass, so a fallback is always exactly
    # one hop from a direct registration and can never chain through another
    # fallback — chaining is what drifts (6px -> 282px measured), so it is
    # bounded here by construction rather than by hope.
    direct_done = sorted(H_by)
    for i in list(pending):
        if not direct_done:
            break
        j = min(direct_done, key=lambda d: abs(d - i))
        H, n = _homography(feats[i], feats[j], min_inliers)
        if H is not None and n >= min_inliers:
            H_by[i] = H_by[j] @ H
            inl_by[i] = n
            pending.remove(i)
    if pending:
        frac = len(pending) / len(frames)
        if frac > max_unregistered_frac:
            raise CalibrationError(
                f"endzone mosaic: {len(pending)} frames could not be registered "
                f"(e.g. {pending[:5]}) — too little static field visible; sample "
                f"different frames or lower the frame stride. That is "
                f"{frac:.0%} of {len(frames)} sampled frames, above the "
                f"max_unregistered_frac={max_unregistered_frac:.0%} tolerance.")
        # Below tolerance: drop them rather than abort. They simply get no
        # entry in H_by_frame -- accumulate_field_paint skips any frame whose
        # H_by_frame.get(i) is None, and propagate leaves out[t] = None for
        # it, which assemble_track_from_results handles as an interpolated
        # or clamp-extrapolated gap. Never silent, though: a couple of
        # dropped frames looks identical to a healthy run unless logged.
        _LOG.warning(
            "endzone mosaic: dropping %d/%d unregistered frames (%.0f%%, "
            "e.g. %s) — too little static field visible in each; they get "
            "no entry in H_by_frame and are skipped downstream",
            len(pending), len(frames), frac * 100, pending[:5])
    return H_by, inl_by


def register_dense_to_reference(frame_iter, ref_img, *, ref_boxes=None,
                                min_inliers: int = 25,
                                boxes_by_frame=None):
    """Stream frames past the reference, returning {frame: H into reference}.

    Same job as :func:`register_to_reference`, but it never holds more than one
    frame at a time. That matters because the mosaic only needs a SPARSE sample
    to accumulate paint, while the output camera track wants one camera per
    frame to composite against -- and holding 1302 frames of 1920x1080 to get
    that would cost about 8 GB. Here the reference's features are computed once
    and each frame is matched and dropped.

    The one-hop fallback is kept, but the hop target is the most recently
    registered frame rather than the nearest of all of them: it is the closest
    in time to whatever is failing now, and it is the only one still in memory.
    A fallback still never chains through another fallback -- ``hop`` is only
    ever replaced by a DIRECT registration -- so drift stays bounded by
    construction, which is what the sparse path relies on too (chaining was
    measured at 6 px -> 282 px).

    Frames that register neither way are simply absent from the result, exactly
    as in the sparse path; propagate leaves them as gaps.
    """
    boxes_by_frame = boxes_by_frame or {}
    ref_feats = _features(ref_img, keep_mask(ref_img.shape, ref_boxes))
    h_by, hop = {}, None
    failed = 0
    for idx, img in frame_iter:
        feats = _features(img, keep_mask(img.shape, boxes_by_frame.get(idx)))
        h_mat, n = _homography(feats, ref_feats, min_inliers)
        if h_mat is not None and n >= min_inliers:
            h_by[idx] = h_mat
            hop = (feats, h_mat)          # only DIRECT results become hops
            continue
        if hop is not None:
            h_rel, n2 = _homography(feats, hop[0], min_inliers)
            if h_rel is not None and n2 >= min_inliers:
                h_by[idx] = hop[1] @ h_rel
                continue
        failed += 1
    if failed:
        _LOG.warning("endzone mosaic: dense registration could not place "
                     "%d frame(s); they become gaps in the camera track",
                     failed)
    return h_by


def register_dense_to_reference(frame_iter, ref_img, *, ref_boxes=None,
                                min_inliers: int = 25,
                                boxes_by_frame=None):
    """Stream frames past the reference, returning {frame: H into reference}.

    Same job as :func:`register_to_reference`, but it never holds more than one
    frame at a time. That matters because the mosaic only needs a SPARSE sample
    to accumulate paint, while the output camera track wants one camera per
    frame to composite against -- and holding 1302 frames of 1920x1080 to get
    that would cost about 8 GB. Here the reference's features are computed once
    and each frame is matched and dropped.

    The one-hop fallback is kept, but the hop target is the most recently
    registered frame rather than the nearest of all of them: it is the closest
    in time to whatever is failing now, and it is the only one still in memory.
    A fallback still never chains through another fallback -- ``hop`` is only
    ever replaced by a DIRECT registration -- so drift stays bounded by
    construction, which is what the sparse path relies on too (chaining was
    measured at 6 px -> 282 px).

    Frames that register neither way are simply absent from the result, exactly
    as in the sparse path; propagate leaves them as gaps.
    """
    boxes_by_frame = boxes_by_frame or {}
    ref_feats = _features(ref_img, keep_mask(ref_img.shape, ref_boxes))
    h_by, hop = {}, None
    failed = 0
    for idx, img in frame_iter:
        feats = _features(img, keep_mask(img.shape, boxes_by_frame.get(idx)))
        h_mat, n = _homography(feats, ref_feats, min_inliers)
        if h_mat is not None and n >= min_inliers:
            h_by[idx] = h_mat
            hop = (feats, h_mat)          # only DIRECT results become hops
            continue
        if hop is not None:
            h_rel, n2 = _homography(feats, hop[0], min_inliers)
            if h_rel is not None and n2 >= min_inliers:
                h_by[idx] = hop[1] @ h_rel
                continue
        failed += 1
    if failed:
        _LOG.warning("endzone mosaic: dense registration could not place "
                     "%d frame(s); they become gaps in the camera track",
                     failed)
    return h_by


def _white_mask(img_bgr, lo, hi) -> np.ndarray:
    """Painted lines: bright and low-saturation. Cables are DARK, so they are
    excluded here by construction; their only effect is occlusion, which
    accumulating over many frames heals."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def accumulate_field_paint(frames, H_by_frame, boxes_by_frame, *, ref_shape,
                           white_lo=(0, 0, 165), white_hi=(180, 70, 255)):
    """Votes-per-pixel image of STATIC paint in the reference frame (0..1).

    Each frame's player-masked white mask is warped into the reference and
    summed, then divided by how often each pixel was actually ELIGIBLE to be
    observed as paint -- i.e. warped-in coverage of the player-masked ``keep``
    region, not the whole frame. A pixel a player is standing on in some
    frames was never a candidate for paint there in those frames; counting it
    as "seen" anyway would dilute the vote of a line a player parks on for
    much of the play (e.g. a lineman on the line of scrimmage) below the
    survival threshold even though every frame that COULD see paint there
    agreed it was paint. Static paint reinforces; movers and per-frame junk
    wash out.

    A pixel eligible in very few frames is a separate failure mode, the
    mirror image of the one above: with only 1-2 observations, ``votes/seen``
    has no statistical weight behind it, so a single-frame bright non-paint
    artifact (glare, a towel, a jersey) at a pixel a player box happens to
    occlude everywhere else votes a confident 1.0 despite being paint in
    exactly one frame. Pixels with fewer eligible observations than
    ``max(2.0, 0.2 * n_contributing_frames)`` (``n_contributing_frames`` =
    frames that actually had a homography to accumulate through) are treated
    as unobserved (vote 0) rather than divided."""
    h, w = ref_shape[:2]
    votes = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)
    n_contributing = 0
    for i, img in frames.items():
        H = H_by_frame.get(i)
        if H is None:
            continue
        n_contributing += 1
        keep = keep_mask(img.shape, boxes_by_frame.get(i))
        paint = cv2.bitwise_and(_white_mask(img, white_lo, white_hi), keep)
        votes += cv2.warpPerspective(
            (paint > 0).astype(np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
        seen += cv2.warpPerspective(
            (keep > 0).astype(np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
    if not seen.any():
        raise CalibrationError(
            "endzone mosaic: no frames contributed coverage — check the "
            "homographies and that the videos decoded.")
    min_support = max(2.0, 0.2 * n_contributing)
    ratio = np.divide(votes, seen, out=np.zeros_like(votes), where=seen > 0)
    return np.where(seen >= min_support, ratio, 0.0).astype(np.float32)


from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M


def _refine_reference_camera(world, uv, K, R, t, line_constraints=None):
    """Least-squares refinement of (focal, rotation, centre) on the points.

    homography_to_krt recovers the focal by ENFORCING that the plane's two
    world axes come out orthonormal. At a narrow field of view that constraint
    is weakly determined, so noise in a fitted H swings the resulting camera
    far away from the fit H itself describes -- measured on SEA@AZ play_001
    (about 6 degrees horizontal FOV): the homography reprojects its own
    correspondences at 1.9 px while the camera decomposed from it reprojects
    them at 253 px, because the world-X column of H flipped sign under the
    orthonormality projection. Both were 'valid' cameras; only one fits.

    So the decomposition is an INITIALISATION and this is the estimate: it
    minimises reprojection under exactly the camera model the caller assumes
    (square pixels, principal point fixed where homography_to_krt put it),
    which is the quantity the rms gate below then judges. A soft_l1 loss keeps
    a handful of mis-snapped hash marks from dragging the fit."""
    from scipy.optimize import least_squares

    cx, cy = float(K[0, 2]), float(K[1, 2])
    WIDE, TALL = 2.0 * cx, 2.0 * cy
    centre = -R.T @ t
    rvec, _ = cv2.Rodrigues(R)
    p0 = np.concatenate([[float(K[0, 0])], rvec.ravel(), centre])

    # Every point correspondence sits on one of the two hash rows, a strip
    # only 2*HASH_OFFSET_M wide. That barely constrains rotation about the
    # field's long axis: measured on play_001, two rotations 0.29 deg apart
    # both fit the marks at ~2 px, and at f~19000 that is 96 px on the field.
    # The yard lines span the whole visible width and break the tie -- as LINE
    # constraints, since where a point falls ALONG a yard line is unknown (and
    # feeding their intersections as points instead imports their tilt error,
    # which is the least-determined thing in this view).
    spans = []
    if line_constraints:
        for world_x, line in line_constraints:
            # Sample only across the span the camera can actually SEE, not the
            # full field width: a zoomed endzone view covers ~14 m of it, so
            # sampling +-HALF_WIDTH_M puts most points far outside the frame
            # where the point-to-line residual is meaningless and swamps the
            # real ones (measured: it moved a correct solve 99 px off).
            ys = np.linspace(-3.0 * HASH_OFFSET_M, 3.0 * HASH_OFFSET_M, 25)
            pts = np.column_stack([np.full_like(ys, float(world_x)), ys,
                                   np.zeros_like(ys)])
            spans.append((pts, np.asarray(line, float)))

    def residual(p):
        rot, _ = cv2.Rodrigues(p[1:4])
        centre = p[4:7]
        cam = world @ rot.T + (-rot @ centre)
        z = np.where(np.abs(cam[:, 2]) < 1e-6, 1e-6, cam[:, 2])
        out = [p[0] * cam[:, 0] / z + cx - uv[:, 0],
               p[0] * cam[:, 1] / z + cy - uv[:, 1]]
        for pts, line in spans:
            c = pts @ rot.T + (-rot @ centre)
            zz = np.where(np.abs(c[:, 2]) < 1e-6, 1e-6, c[:, 2])
            u_ = p[0] * c[:, 0] / zz + cx
            v_ = p[0] * c[:, 1] / zz + cy
            # and drop any sample that still lands off-sensor
            usable = ((c[:, 2] > 1e-6) & (u_ > -0.25 * WIDE) & (u_ < 1.25 * WIDE)
                      & (v_ > -0.25 * TALL) & (v_ < 1.25 * TALL))
            d = line[0] * u_ + line[1] * v_ + line[2]
            out.append(np.where(usable, d, 0.0))
        return np.concatenate(out)

    # TWO stages. A robust loss alone cannot be used for the first: the yard
    # line constraints exist precisely to pull the fit out of a wrong basin,
    # and they start there at ~99 px -- which soft_l1 with a 3 px scale
    # dismisses as outliers, leaving the solve stuck exactly where it needs
    # moving from (measured: identical wrong camera with and without the line
    # constraints). So fit once with a plain quadratic loss to find the basin,
    # then refine robustly to shed genuinely mis-snapped marks.
    sol = least_squares(residual, p0, max_nfev=500)
    if np.isfinite(sol.x).all() and sol.x[0] > 0:
        p0 = sol.x
    sol = least_squares(residual, p0, loss="soft_l1", f_scale=3.0,
                        max_nfev=500)
    if not np.isfinite(sol.x).all() or sol.x[0] <= 0:
        return K, R, t          # refinement diverged; keep the decomposition
    rot, _ = cv2.Rodrigues(sol.x[1:4])
    k_new = np.array([[sol.x[0], 0.0, cx], [0.0, sol.x[0], cy], [0.0, 0.0, 1.0]])
    return k_new, rot, -rot @ sol.x[4:7]


def solve_reference_camera(world_xyz, ref_uv, image_size, prior,
                           line_constraints=None):
    """One camera for the reference frame from the labelled field lines.

    A single view of the z=0 field plane is a HOMOGRAPHY problem, not a
    multi-frame shared-centre problem. An earlier draft routed this through
    solve_fixed_center, which gates on >= 10 mutually consistent FRAMES and so
    could never have succeeded on one view — it would have failed every real
    run. decompose_homography.homography_to_krt is purpose-built for this: it
    recovers the focal from the orthonormality of the plane axes."""
    from nfl_gsplat.calibration.decompose_homography import homography_to_krt
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    world = np.asarray(world_xyz, np.float64).reshape(-1, 3)
    uv = np.asarray(ref_uv, np.float64).reshape(-1, 2)
    if len(world) < 4:
        raise CalibrationError(
            f"endzone mosaic: only {len(world)} field correspondences in the "
            "reference frame (need >= 4 for a homography) — accumulated paint "
            "too sparse; lower vote_thresh or sample more frames.")

    H, mask = cv2.findHomography(world[:, :2].astype(np.float64),
                                 uv.astype(np.float64), cv2.RANSAC, 3.0)
    if H is None:
        raise CalibrationError(
            "endzone mosaic: field->reference homography did not fit — the "
            "yard-line labeling is probably wrong (check the anchors).")

    # Gate on the evidence findHomography already computed instead of
    # discarding it. RANSAC quietly excludes disagreeing points from the fit,
    # so a HIGH inlier fraction does not by itself mean the labeling is right
    # -- it only means most points AGREE with each other. This gate requires
    # a clear majority (>=70%) of the labelled points to be self-consistent
    # under a single planar homography; below that, too much of the input
    # disagrees to trust any fit RANSAC happens to settle on.
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_frac = inliers / len(world)
    if inlier_frac < 0.7:
        raise CalibrationError(
            f"endzone mosaic: field->reference homography kept only "
            f"{inliers}/{len(world)} points as inliers ({inlier_frac:.0%}) — "
            "too many field correspondences disagree to trust this fit; "
            "check the yard-line labeling.")

    w, h = int(image_size[0]), int(image_size[1])
    try:
        K, R, t = homography_to_krt(H, width=w, height=h)
    except ValueError as e:
        raise CalibrationError(
            f"endzone mosaic: reference view is degenerate for focal recovery "
            f"({e}) — the accumulated lines are nearly all parallel in the "
            "image; sample frames with more pan/zoom spread.") from e
    K, R, t = _refine_reference_camera(world, uv, K, R, t,
                                       line_constraints=line_constraints)
    C = -R.T @ t

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = prior.center_bounds
    # NOTE: this box is structurally blind to a CONSTANT yard offset -- an
    # off-by-one/two/five mislabeling of every line shifts C along the field
    # axis but can still land inside the box with rms ~0 (it does catch
    # mirrors and scale errors, which move C off-axis or change its
    # magnitude). Catching a constant offset is Task 3's two-anchor
    # invariant, not this function's job.
    if not (xlo <= C[0] <= xhi and ylo <= C[1] <= yhi and zlo <= C[2] <= zhi):
        raise CalibrationError(
            f"endzone mosaic: solved camera centre {C.round(1)} is outside the "
            f"prior box {prior.center_bounds} — the yard-line labeling or the "
            "anchors are wrong.")
    flo, fhi = prior.focal_range
    if not (flo <= float(K[0, 0]) <= fhi):
        raise CalibrationError(
            f"endzone mosaic: solved focal {K[0, 0]:.0f} px is outside the "
            f"prior range {prior.focal_range}.")

    proj = project_points(world, K, R, t)
    ok = np.isfinite(proj).all(axis=1)
    rms = float(np.sqrt(np.mean(np.sum((proj[ok] - uv[ok]) ** 2, axis=1))))
    # This average is over ALL supplied points, not just RANSAC's inliers --
    # a mislabeled line RANSAC correctly excluded from the fit still shows up
    # here as a large residual (measured: one wrong yard line, kept at 9/10
    # inliers by RANSAC, still reprojected at 11.9 px rms). 3.0 px matches the
    # RANSAC inlier threshold used to fit H above: a genuinely single planar
    # camera should reproject every consistent point about that tightly.
    if rms > 3.0:
        raise CalibrationError(
            f"endzone mosaic: reference camera reprojection rms {rms:.2f} px "
            "exceeds 3.0 px — the field correspondences are inconsistent "
            "with a single planar camera; check the yard-line labeling.")
    return CalibrationResult(
        intrinsics=CameraIntrinsics(float(K[0, 0]), float(K[1, 1]),
                                    float(K[0, 2]), float(K[1, 2]), w, h),
        pose=CameraPose(R=R, t=t), rms_px=rms,
        num_correspondences=inliers, refined_with_ba=False)


def propagate(H_by_frame, ref_cam, n_frames: int, *,
             aspect_range=(0.95, 1.05), focal_range=None):
    """Per-frame cameras from the reference camera and each frame's homography.

    H_t maps frame t's pixels INTO the reference, so K_t R_t = H_t^-1 K_ref R_ref.
    The centre is shared (tripod), so t_t = -R_t C.

    Every decomposed frame is gated before being emitted: the RQ decomposition
    of an arbitrary 3x3 will always produce SOME (K, R), including camera
    models nothing else in this repo accepts (measured: fx=1300, fy=5200, a
    4:1 anisotropic K, from a frame resolved via the one-hop registration
    fallback or matched on weak texture). A frame whose fx/fy falls outside
    ``aspect_range`` (the pinhole/unit-aspect model used everywhere else) is
    dropped (``out[t] = None``) rather than emitted with unearned confidence.

    ``focal_range``, if given (pass the operator's own
    ``endzone_prior.focal_range``), additionally drops a frame whose fx falls
    outside it. An earlier version instead compared fx to +-20% of the
    REFERENCE camera's own focal; measured too narrow for a real play's zoom
    span (acceptance band s in [0.85, 1.25], barely 1.5x total zoom) and only
    tolerable because the reference happened to sit mid-range. When
    ``focal_range`` is None (the default), no focal gate applies -- the
    aspect gate alone still catches gross anisotropy (e.g. the measured
    4:1 case).

    Accepted frames carry rms_px=0.0/num_correspondences=0 -- NOT the
    reference camera's, since propagate never re-measures reprojection error
    and copying the reference's numbers would let a consumer read a
    propagated frame's confidence as if it had been independently verified.

    Drops are counted and logged (never silent: a couple of dropped frames
    get interpolated invisibly by assemble_track_from_results, and a longer
    dropped tail becomes a clamp-extrapolated conf=0 gap -- both
    indistinguishable from a healthy run unless the count is surfaced). If
    more than half the sampled frames are dropped, raises CalibrationError
    naming the count."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    C = np.asarray(ref_cam.pose.center_world(), np.float64)
    M_ref = ref_cam.intrinsics.K() @ ref_cam.pose.R
    w, h = ref_cam.intrinsics.width, ref_cam.intrinsics.height
    cx, cy = float(ref_cam.intrinsics.cx), float(ref_cam.intrinsics.cy)
    K_pp = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    cx, cy = float(ref_cam.intrinsics.cx), float(ref_cam.intrinsics.cy)
    K_pp = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    alo, ahi = aspect_range
    out: list = [None] * n_frames
    n_sampled = len(H_by_frame)
    dropped = 0
    for t, H in H_by_frame.items():
        if not (0 <= t < n_frames):
            continue
        M = np.linalg.inv(H) @ M_ref                 # = K_t R_t
        # A tripod camera's principal point CANNOT move -- it is fixed by the
        # sensor. Decomposing M with a general RQ lets the principal point
        # float and absorb registration noise instead, which is not a harmless
        # reparametrisation: measured on SEA@AZ play_001 it wandered to
        # (-286, -945) on a 1920x1080 frame, and even the REFERENCE frame,
        # where H is the identity and the answer must come back unchanged,
        # came out shifted 168 px with its yard-line spacing still correct --
        # the signature of a moved principal point.
        #
        # So the principal point is HELD at the reference's, and only the
        # focal and the rotation are recovered:
        #     K_pp^-1 M = diag(fx, fy, 1) R
        # whose third row is R's third row (unit norm, which fixes the
        # homogeneous scale) and whose first two rows are the focals times R's
        # first two. That is the same pinhole model the reference camera was
        # solved under, so the propagated frames stay comparable to it.
        n_mat = np.linalg.solve(K_pp, M)
        scale = float(np.linalg.norm(n_mat[2]))
        if not np.isfinite(scale) or scale < 1e-12:
            dropped += 1
            continue
        n_mat = n_mat / scale
        fx = float(np.linalg.norm(n_mat[0]))
        fy = float(np.linalg.norm(n_mat[1]))
        if not (np.isfinite(fx) and np.isfinite(fy)) or fx <= 0 or fy <= 0:
            dropped += 1
            continue
        # Sanity gate -- see the docstring. Anything outside the unit-aspect
        # pinhole model used everywhere else in this repo, or (if the
        # operator supplied a focal_range) outside their own declared focal
        # range, is dropped rather than emitted.
        if not (alo <= fx / fy <= ahi):
            dropped += 1
            continue
        if focal_range is not None:
            flo, fhi = focal_range
            if not (flo <= fx <= fhi):
                dropped += 1
                continue
        focal = 0.5 * (fx + fy)
        approx = np.vstack([n_mat[0] / focal, n_mat[1] / focal, n_mat[2]])
        # M is defined only up to a nonzero homogeneous scale, so the sign is
        # still free; exactly one choice is a rotation rather than a
        # reflection. Negating all three rows flips det for a 3x3, so this
        # cannot fail to find it.
        if np.linalg.det(approx) < 0:
            approx = -approx
        u_mat, _sv, vt = np.linalg.svd(approx)
        R = u_mat @ vt
        if np.linalg.det(R) < 0:
            raise CalibrationError(
                f"endzone mosaic: frame {t} decomposed to a reflection "
                "(det R < 0) - the homography or the reference camera is "
                "inconsistent; this must not be silently accepted.")
        out[t] = CalibrationResult(
            intrinsics=CameraIntrinsics(focal, focal, cx, cy, w, h),
            pose=CameraPose(R=R, t=-R @ C),
            rms_px=0.0, num_correspondences=0,
            refined_with_ba=False)
    if dropped:
        _LOG.warning(
            "endzone mosaic: propagate dropped %d/%d sampled frames "
            "(failed the aspect/focal sanity gate)", dropped, n_sampled)
    if n_sampled and dropped > 0.5 * n_sampled:
        raise CalibrationError(
            f"endzone mosaic: propagate dropped {dropped}/{n_sampled} sampled "
            "frames — more than half failed the aspect/focal sanity gate; "
            "check the mosaic's registration homographies and "
            "endzone_prior.focal_range.")
    return out


def _rq3(M):
    """RQ decomposition of a 3x3 matrix into (upper-triangular, rotation)."""
    P = np.flipud(np.eye(3))
    Q_, R_ = np.linalg.qr((P @ M).T)
    K = P @ R_.T @ P
    R = P @ Q_.T
    for i in range(3):                                # make K's diagonal positive
        if K[i, i] < 0:
            K[:, i] *= -1.0
            R[i, :] *= -1.0
    return K, R
