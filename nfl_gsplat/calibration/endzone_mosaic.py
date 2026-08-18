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
                          boxes_by_frame=None):
    """{frame: H mapping that frame's pixels INTO the reference}, {frame: inliers}.

    Direct link first; if it is too weak, fall back to composing through the
    nearest DIRECTLY-registered frame — exactly one hop, so drift stays
    bounded by construction rather than by how many pending frames happen to
    register first.

    ``boxes_by_frame``, if given, excludes players from feature detection
    (via :func:`keep_mask`) so moving players never contribute non-rigid
    matches that would break the pure-rotation/zoom model."""
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
        raise CalibrationError(
            f"endzone mosaic: {len(pending)} frames could not be registered "
            f"(e.g. {pending[:5]}) — too little static field visible; sample "
            "different frames or lower the frame stride.")
    return H_by, inl_by


def _white_mask(img_bgr, lo, hi) -> np.ndarray:
    """Painted lines: bright and low-saturation. Cables are DARK, so they are
    excluded here by construction; their only effect is occlusion, which
    accumulating over many frames heals."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def field_extent_score(img_bgr, boxes=None, *,
                       white_lo=(0, 0, 165), white_hi=(180, 70, 255)) -> int:
    """Cheap proxy for how much of the field is in view: count of
    player-masked white/paint pixels in one frame.

    Used to pick the mosaic's reference frame. The mosaic is clipped to the
    reference frame's FOV, so a zoomed-in reference silently clips yard lines
    at the image border (the ``+-HALF_WIDTH_M`` endpoint assumption then
    becomes wrong) -- preferring the frame with the most visible field paint
    over an arbitrary median-index pick avoids that."""
    mask = cv2.bitwise_and(_white_mask(img_bgr, white_lo, white_hi),
                           keep_mask(img_bgr.shape, boxes))
    return int(np.count_nonzero(mask))


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
    wash out."""
    h, w = ref_shape[:2]
    votes = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)
    for i, img in frames.items():
        H = H_by_frame.get(i)
        if H is None:
            continue
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
    return np.divide(votes, seen, out=np.zeros_like(votes), where=seen > 0)


def solve_reference_camera(world_xyz, ref_uv, image_size, prior):
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
             aspect_range=(0.95, 1.05), ref_focal_tol: float = 0.20):
    """Per-frame cameras from the reference camera and each frame's homography.

    H_t maps frame t's pixels INTO the reference, so K_t R_t = H_t^-1 K_ref R_ref.
    The centre is shared (tripod), so t_t = -R_t C.

    Every decomposed frame is gated before being emitted: the RQ decomposition
    of an arbitrary 3x3 will always produce SOME (K, R), including camera
    models nothing else in this repo accepts (measured: fx=1300, fy=5200, a
    4:1 anisotropic K, from a frame resolved via the one-hop registration
    fallback or matched on weak texture). A frame whose fx/fy falls outside
    ``aspect_range`` (the pinhole/unit-aspect model used everywhere else) or
    whose fx strays more than ``ref_focal_tol`` from the reference camera's
    focal is dropped (``out[t] = None``) rather than emitted with unearned
    confidence. Accepted frames carry rms_px=0.0/num_correspondences=0 -- NOT
    the reference camera's, since propagate never re-measures reprojection
    error and copying the reference's numbers would let a consumer read a
    propagated frame's confidence as if it had been independently verified."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    C = np.asarray(ref_cam.pose.center_world(), np.float64)
    M_ref = ref_cam.intrinsics.K() @ ref_cam.pose.R
    w, h = ref_cam.intrinsics.width, ref_cam.intrinsics.height
    ref_fx = float(ref_cam.intrinsics.fx)
    alo, ahi = aspect_range
    out: list = [None] * n_frames
    for t, H in H_by_frame.items():
        if not (0 <= t < n_frames):
            continue
        M = np.linalg.inv(H) @ M_ref                 # = K_t R_t
        # H (and so M) is only defined up to an arbitrary nonzero homogeneous
        # scale, so det(M) < 0 just means that scale came out negative here --
        # NOT that the frame camera is a reflection. Normalise BEFORE
        # decomposing: negating a 3x3 flips its determinant, so this picks the
        # positive-scale representative and _rq3 (which always returns a
        # positive-diagonal K) is then guaranteed det(R) > 0. multiplying K by
        # diag(-1,-1,1) AFTER decomposition cannot fix a wrong det(R) --
        # det(diag(-1,-1,1)) = +1, so it cannot flip sign(det(R)); it can only
        # corrupt K's focal signs, which is the bug this replaces.
        if np.linalg.det(M) < 0:
            M = -M
        # RQ-decompose M into upper-triangular K and rotation R.
        K, R = _rq3(M)
        if K[2, 2] == 0:
            continue
        K = K / K[2, 2]
        if np.linalg.det(R) < 0:
            raise CalibrationError(
                f"endzone mosaic: frame {t} decomposed to a reflection "
                "(det R < 0) — the homography or the reference camera is "
                "inconsistent; this must not be silently accepted.")
        fx, fy = float(K[0, 0]), float(K[1, 1])
        # Sanity gate -- see the docstring. Anything outside the unit-aspect
        # pinhole model used everywhere else in this repo, or too far from
        # the reference camera's own focal, is dropped rather than emitted.
        if fy == 0 or not (alo <= fx / fy <= ahi):
            continue
        if ref_fx == 0 or abs(fx - ref_fx) > ref_focal_tol * ref_fx:
            continue
        out[t] = CalibrationResult(
            intrinsics=CameraIntrinsics(fx, fy, float(K[0, 2]), float(K[1, 2]), w, h),
            pose=CameraPose(R=R, t=-R @ C),
            rms_px=0.0, num_correspondences=0,
            refined_with_ba=False)
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
