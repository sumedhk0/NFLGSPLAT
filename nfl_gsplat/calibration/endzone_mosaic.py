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


def register_to_reference(frames, *, ref_idx, min_inliers: int = 25):
    """{frame: H mapping that frame's pixels INTO the reference}, {frame: inliers}.

    Direct link first; if it is too weak, fall back to composing through the
    nearest DIRECTLY-registered frame — exactly one hop, so drift stays
    bounded by construction rather than by how many pending frames happen to
    register first."""
    if ref_idx not in frames:
        raise CalibrationError(
            f"endzone mosaic: reference frame {ref_idx} not among the sampled "
            "frames — pick a reference that was actually decoded.")
    feats = {i: _features(img) for i, img in frames.items()}
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


def accumulate_field_paint(frames, H_by_frame, boxes_by_frame, *, ref_shape,
                           white_lo=(0, 0, 165), white_hi=(180, 70, 255)):
    """Votes-per-pixel image of STATIC paint in the reference frame (0..1).

    Each frame's player-masked white mask is warped into the reference and
    summed, then divided by how often each pixel was actually observed. Static
    paint reinforces; movers and per-frame junk wash out."""
    h, w = ref_shape[:2]
    votes = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)
    for i, img in frames.items():
        H = H_by_frame.get(i)
        if H is None:
            continue
        paint = _white_mask(img, white_lo, white_hi)
        paint = cv2.bitwise_and(paint, keep_mask(img.shape, boxes_by_frame.get(i)))
        votes += cv2.warpPerspective(
            (paint > 0).astype(np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
        seen += cv2.warpPerspective(
            np.ones(img.shape[:2], np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
    if not seen.any():
        raise CalibrationError(
            "endzone mosaic: no frames contributed coverage — check the "
            "homographies and that the videos decoded.")
    return np.divide(votes, seen, out=np.zeros_like(votes), where=seen > 0)
