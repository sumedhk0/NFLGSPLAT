"""Per-frame registration over a clip → smoothed CameraTrack → cameras.npz.

Detect+register each frame (env-gated seam: video read + cv2 line/hash detection), then smooth
the per-frame (K,R,t) and interpolate short gaps; fail loud on a long gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.field_identify import seed_state_from_hint
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame
from nfl_gsplat.calibration.register_frame import register_frame
from nfl_gsplat.errors import CalibrationError


def _longest_gap_range(valid: np.ndarray, *, interior_only: bool = False) -> tuple[int, int, int]:
    """Return (longest_gap_len, start, end) over False runs in ``valid``.

    With ``interior_only=True``, runs touching index 0 or len(valid)-1 (leading/
    trailing gaps) are skipped — those are clamp-extrapolated, not fail-loud."""
    best = (0, -1, -1)
    i, n = 0, len(valid)
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            is_boundary = i == 0 or j == n
            if not (interior_only and is_boundary) and (j - i) > best[0]:
                best = (j - i, i, j - 1)
            i = j
        else:
            i += 1
    return best


def _check_ckpt_classes(ckpt_classes, schema_names):
    """Fail loud if a checkpoint's landmark class list doesn't match the schema —
    otherwise model channels map to the wrong landmark names (silent wrong calib)."""
    from nfl_gsplat.errors import SetupError
    if list(ckpt_classes) != list(schema_names):
        raise SetupError(
            "model checkpoint landmark classes do not match the schema "
            f"(yard window). Checkpoint has {len(ckpt_classes)} classes, schema has "
            f"{len(schema_names)}. Re-run with the SAME --yard-min/--yard-max used at "
            "training time (see SETUP.md §3)."
        )


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
    gap_len, gs, ge = _longest_gap_range(valid, interior_only=True)
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


def _register_corrs(corrs, image_size, *, max_reproj_px=6.0, min_landmarks=6,
                     initial_intrinsics=None):
    """One frame's correspondences → CalibrationResult|None (gap)."""
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
    from nfl_gsplat.errors import CalibrationError
    if len(corrs) < min_landmarks:
        return None
    try:
        return solve_pnp_from_correspondences(
            corrs, image_size=image_size, max_reproj_px=max_reproj_px,
            min_landmarks=min_landmarks, initial_intrinsics=initial_intrinsics)
    except CalibrationError:
        return None


def _solve_sweep(corrs_by_frame, num_frames, image_size, *,
                  max_reproj_px=6.0, min_landmarks=6):
    """Cached per-frame correspondences -> [CalibrationResult|None], solved in two
    sweeps that carry an intrinsics prior instead of solving each frame blind.

    Blind-init PnP (``initial_intrinsics=None``) falls into bad local minima on
    many real frames while neighboring frames — same physically-fixed camera —
    solve fine. Forward sweep: seed each frame's solve with the last successful
    frame's intrinsics. Backward sweep: fill frames that precede the first
    forward success, using the nearest later success as the prior."""
    results = [None] * num_frames
    prior = None
    for fidx in range(num_frames):                    # forward
        corrs = corrs_by_frame.get(fidx)
        if not corrs:
            continue
        res = _register_corrs(corrs, image_size, max_reproj_px=max_reproj_px,
                              min_landmarks=min_landmarks, initial_intrinsics=prior)
        if res is not None:
            results[fidx] = res
            prior = getattr(res, "intrinsics", prior)
    prior = None
    for fidx in range(num_frames - 1, -1, -1):         # backward fill
        if results[fidx] is not None:
            prior = getattr(results[fidx], "intrinsics", prior)
            continue
        corrs = corrs_by_frame.get(fidx)
        if corrs and prior is not None:
            res = _register_corrs(corrs, image_size, max_reproj_px=max_reproj_px,
                                  min_landmarks=min_landmarks, initial_intrinsics=prior)
            if res is not None:
                results[fidx] = res
    return results


def _register_sequence_learned(frames, *, detector, image_size,
                               max_reproj_px=6.0, min_landmarks=6):
    """Per-frame: detector(frame)->[(name,(u,v))] → PnP. No hint/consensus needed
    (the learned detector outputs labeled, well-spread correspondences)."""
    results = []
    for fr in frames:
        if fr is None:
            results.append(None); continue
        results.append(_register_corrs(detector(fr), image_size,
                                       max_reproj_px=max_reproj_px,
                                       min_landmarks=min_landmarks))
    return results


def build_autocalib_npz_learned(*, play_dir, videos, fps, model_ckpt, yard_min,
                                yard_max, conf_thresh=0.5, in_hw=(540, 960), heat_stride=4):
    """Learned-mode calibration: a trained LandmarkNet drives per-frame PnP."""
    import torch

    from nfl_gsplat.landmarks.infer import detect_landmarks, landmarks_to_correspondences, run_model
    from nfl_gsplat.landmarks.model import LandmarkNet
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    # FIX 3: fail loud on missing checkpoint before trying torch.load
    from pathlib import Path as _P
    if not _P(model_ckpt).exists():
        from nfl_gsplat.errors import SetupError
        raise SetupError(
            f"model checkpoint not found: {model_ckpt} — train one first "
            "(sbatch scripts/train_landmarks.sbatch ...). See SETUP.md §3."
        )

    schema = LandmarkSchema(yard_min=yard_min, yard_max=yard_max)
    st = torch.load(model_ckpt, map_location="cpu")

    # FIX 1: validate checkpoint classes against schema before loading weights
    _check_ckpt_classes(st["classes"], schema.class_names())

    net = LandmarkNet(schema.num_classes)
    net.load_state_dict(st["net"])
    tracks = {}
    for cam, video in videos.items():
        meta = ffprobe_meta(video)

        def detector(bgr, _net=net, _meta=meta):
            hm = run_model(_net, bgr, in_hw=in_hw)
            dets = detect_landmarks(hm, schema, src_hw=(_meta.height, _meta.width),
                                    in_hw=in_hw, heat_stride=heat_stride, conf_thresh=conf_thresh)
            return landmarks_to_correspondences(dets, schema)

        # FIX 2: detect and discard each frame inline — no full-res frame buffer
        results = [None] * meta.num_frames
        for fidx, fr in iter_frames(video, start_frame=0):
            if 0 <= fidx < meta.num_frames:
                results[fidx] = _register_corrs(detector(fr), (meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width, height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)


def _register_sequence_pretrained(frames, *, kps_by_frame, territory, image_size,
                                  cfg=None, boxes_for=None):
    """Per frame: classical detect + cached model kps -> fuse -> correspondences,
    then solve with a two-sweep intrinsics prior (see :func:`_solve_sweep`).
    Frames without cached keypoints (or unreadable) are gaps (None)."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
    cfg = cfg or FieldDetectConfig()
    boxes_for = boxes_for or (lambda f: [])
    corrs_by_frame: dict[int, list] = {}
    for fidx, fr in enumerate(frames):
        kps = kps_by_frame.get(fidx, [])
        if fr is None or not kps:
            continue
        feats = detect_field_features(fr, cfg=cfg, player_boxes=boxes_for(fidx))
        corrs_by_frame[fidx] = fuse_frame(feats.yard_lines, feats.hashes, kps,
                                          territory=territory, image_size=image_size)
    return _solve_sweep(corrs_by_frame, len(frames), image_size)


def build_autocalib_npz_pretrained(*, play_dir, videos, fps, kps_json, territory,
                                   cfg=None, masks_provider=None):
    """Pretrained-hybrid calibration: cached Roboflow identity + repaired
    classical geometry -> per-frame PnP -> cameras.npz. No GPU, no training.

    Two phases, kept separate so decoding stays a single streaming pass while
    solving gets to see the whole clip:

    Phase 1 (streaming): decode each frame once via ``iter_frames``; per frame
    with cached model keypoints, run detect_field_features + fuse_frame and
    stash the resulting correspondences in ``corrs_by_frame`` (a few tuples per
    frame — cheap to hold in memory). No PnP solving happens here.

    Phase 2 (in-memory, two sweeps): solve every frame's correspondences with
    :func:`_solve_sweep`, which seeds each solve with the last successful
    frame's intrinsics as a prior. This rescues frames whose blind-init PnP
    lands in a bad local minimum — common on real footage where geometry is
    noisy/sparse but the camera is physically fixed, so neighboring frames'
    intrinsics are a strong prior.
    """
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
    from nfl_gsplat.calibration.roboflow_kps import load_kps_json
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    tracks = {}
    for cam, video in videos.items():
        meta = ffprobe_meta(video)
        kps_by_frame = load_kps_json(kps_json, expect_num_frames=meta.num_frames)
        boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
        _cfg = cfg or FieldDetectConfig()

        corrs_by_frame: dict[int, list] = {}
        for fidx, fr in iter_frames(video, start_frame=0):
            if not (0 <= fidx < meta.num_frames):
                continue
            kps = kps_by_frame.get(fidx, [])
            if not kps:
                continue
            feats = detect_field_features(fr, cfg=_cfg, player_boxes=boxes_for(fidx))
            corrs_by_frame[fidx] = fuse_frame(feats.yard_lines, feats.hashes, kps,
                                              territory=territory,
                                              image_size=(meta.width, meta.height))

        results = _solve_sweep(corrs_by_frame, meta.num_frames, (meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width,
                                                  height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)


def build_autocalib_npz(*, play_dir, videos, fps, hints, cfg=None, masks_provider=None):
    """Detect+register every frame of each camera using its CalibHint → cameras.npz."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
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
