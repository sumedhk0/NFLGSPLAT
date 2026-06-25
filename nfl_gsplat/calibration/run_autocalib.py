"""Per-frame registration over a clip → smoothed CameraTrack → cameras.npz.

Detect+register each frame (env-gated seam: video read + cv2 line/hash detection), then smooth
the per-frame (K,R,t) and interpolate short gaps; fail loud on a long gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.field_identify import seed_state_from_hint
from nfl_gsplat.calibration.register_frame import register_frame
from nfl_gsplat.errors import CalibrationError


def _longest_gap_range(valid: np.ndarray) -> tuple[int, int, int]:
    """Return (longest_gap_len, start, end) over False runs in ``valid``."""
    best = (0, -1, -1)
    i, n = 0, len(valid)
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            if (j - i) > best[0]:
                best = (j - i, i, j - 1)
            i = j
        else:
            i += 1
    return best


def assemble_track_from_results(results, *, width, height, max_gap: int = 5) -> CameraTrack:
    """Stack per-frame CalibrationResults (None = gap) into a CameraTrack.

    Interior short gaps (<= max_gap consecutive) are linearly interpolated.
    Boundary gaps (leading/trailing None runs) are clamp-extrapolated from the
    nearest valid frame via np.interp, flagged by ``conf=0``.
    A longer interior gap raises CalibrationError naming the range (fail loud).
    After interpolation, K/R/t are smoothed with a 1€ filter to reduce
    frame-to-frame jitter; R is then re-orthonormalized via SVD."""
    T = len(results)
    valid = np.array([r is not None for r in results])
    if not valid.any():
        raise CalibrationError("no frame could be registered for this camera.")
    gap_len, gs, ge = _longest_gap_range(valid)
    if gap_len > max_gap:
        raise CalibrationError(
            f"field registration failed on frames {gs}-{ge} "
            f"({gap_len} consecutive). Footage too occluded/zoomed there; see SETUP.md §3."
        )
    K = np.zeros((T, 3, 3)); R = np.zeros((T, 3, 3)); t = np.zeros((T, 3))
    conf = valid.astype(float)
    idx = np.arange(T)
    vi = idx[valid]
    for i in vi:
        r = results[i]
        K[i] = r.intrinsics.K(); R[i] = r.pose.R; t[i] = r.pose.t
    def _interp(stack):
        flat = stack.reshape(T, -1)
        for c in range(flat.shape[1]):
            flat[:, c] = np.interp(idx, vi, flat[vi, c])
        return flat.reshape(stack.shape)
    K, R, t = _interp(K), _interp(R), _interp(t)
    from nfl_gsplat.pose.temporal_smooth import OneEuroConfig, smooth_param_sequence
    sm = OneEuroConfig()
    K = smooth_param_sequence(K.reshape(T, 9), sm).reshape(T, 3, 3)
    R = smooth_param_sequence(R.reshape(T, 9), sm).reshape(T, 3, 3)
    t = smooth_param_sequence(t, sm)
    for i in range(T):
        U, _, Vt = np.linalg.svd(R[i]); R[i] = U @ Vt
    return CameraTrack(K=K, R=R, t=t, conf=conf, width=width, height=height)


def _register_sequence(feats_by_frame, hint, image_size):
    """Seed identity at hint.ref_frame, propagate forward and backward, register
    each frame. Returns [CalibrationResult|None] aligned to feats_by_frame.

    register_frame returns (result, IdentityState); the returned state (labels for
    this frame's lines) becomes the next prior, so labels ride along through pans —
    even across frames whose PnP failed (result None but state still propagated)."""
    T = len(feats_by_frame)
    results = [None] * T
    if T == 0:
        return results
    ref = max(0, min(int(hint.ref_frame), T - 1))
    if feats_by_frame[ref] is None:
        raise CalibrationError(
            f"ref_frame {ref} has no detected features (frame unreadable or out of range)."
        )
    seed = seed_state_from_hint(feats_by_frame[ref], hint)

    res, state_ref = register_frame(feats_by_frame[ref], seed, image_size)
    results[ref] = res
    # Propagation signal is the recovered homography (carried on IdentityState);
    # it lets the next frame predict the anchor line's new position under a pan.
    base = state_ref if state_ref.homography is not None else seed

    prior = base
    for f in range(ref + 1, T):                      # forward
        if feats_by_frame[f] is None:
            results[f] = None
            continue
        res, st = register_frame(feats_by_frame[f], prior, image_size)
        results[f] = res
        if st.homography is not None:
            prior = st
    prior = base
    for f in range(ref - 1, -1, -1):                 # backward
        if feats_by_frame[f] is None:
            results[f] = None
            continue
        res, st = register_frame(feats_by_frame[f], prior, image_size)
        results[f] = res
        if st.homography is not None:
            prior = st
    return results


def build_autocalib_npz(*, play_dir, videos, fps, hints, cfg=None, masks_provider=None):
    """Detect+register every frame of each camera using its CalibHint → cameras.npz."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_field_features
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    cfg = cfg or FieldDetectConfig()
    tracks = {}
    for cam, video in videos.items():
        if cam not in hints:
            raise SetupError(
                f"no calib_hints for camera {cam!r} in meta.yaml — add a one-line "
                "yardage hint (ref_frame/ref_x/yard/side/increasing). See SETUP.md §3."
            )
        meta = ffprobe_meta(video)
        boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
        feats_by_frame = [None] * meta.num_frames
        for fidx, frame in iter_frames(video, start_frame=0):
            if 0 <= fidx < meta.num_frames:
                feats_by_frame[fidx] = detect_field_features(
                    frame, cfg=cfg, player_boxes=boxes_for(fidx))
        results = _register_sequence(feats_by_frame, hints[cam], (meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width, height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)
