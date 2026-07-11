"""Per-frame registration over a clip → smoothed CameraTrack → cameras.npz.

Detect+register each frame (env-gated seam: video read + cv2 line/hash detection), then smooth
the per-frame (K,R,t) and interpolate short gaps; fail loud on a long gap.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.field_identify import fit_hash_rows, seed_state_from_hint
from nfl_gsplat.calibration.fuse_pretrained import (
    correspondences_from_identities, filter_static_hashes, fuse_frame,
    predict_identities, static_hash_cells,
)
from nfl_gsplat.calibration.joint_solve import solve_fixed_center
from nfl_gsplat.calibration.register_frame import register_frame
from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_image, rotate_uv, rotated_wh,
)
from nfl_gsplat.errors import CalibrationError

_LOG = logging.getLogger(__name__)


def _camera_rotation(cam: str, rotations) -> int:
    """View rotation for a camera: explicit map wins; else endzone -> 90
    (broadcast endzone views need the sideline-shaped pipeline rotated),
    everything else 0."""
    if rotations and cam in rotations:
        return int(rotations[cam])
    return 90 if cam == "endzone" else 0


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


def _plane_H(res):
    """The Z=0 plane homography implied by a solved frame: world ``(X, Y, 1)``
    -> image ``(u, v, w)`` up to scale, via ``K @ [R[:,0], R[:,1], t]``. This is
    exactly the mapping used by :func:`~nfl_gsplat.calibration.fuse_pretrained.predict_identities`
    to identify a neighboring frame's classical lines."""
    K = res.intrinsics.K()
    Rt = np.column_stack([res.pose.R[:, 0], res.pose.R[:, 1], res.pose.t])
    return K @ Rt


def _sweep_direction(order, results, corrs_by_frame, feats_by_frame,
                     image_size, max_reproj_px, min_landmarks):
    """Walk ``order`` (a permutation of frame indices), solving/rescuing gaps
    in place on the shared ``results`` list and carrying an intrinsics prior
    (``prior``) plus the nearest solved frame's plane homography (``last_H``).

    ``last_H`` is the temporal-identity-propagation signal: when a frame's
    model-vote correspondences are missing or too sparse to solve, its
    classical lines are identified by mapping them through the nearest
    solved neighbor's plane homography (:func:`predict_identities`), chained
    frame-by-frame through gaps. ``feats_by_frame`` (``{fidx: (yard_lines,
    rows, hashes)}``) is only populated for frames with cached classical
    detections; ``None`` disables the rescue path entirely (keeps this
    sweep's behavior identical to the pre-propagation solver for callers/
    tests that don't supply it)."""
    last_H = None
    prior = None
    for fidx in order:
        if results[fidx] is not None:
            prior = getattr(results[fidx], "intrinsics", prior)
            if feats_by_frame is not None:
                try:
                    last_H = _plane_H(results[fidx])
                except AttributeError:
                    pass
            continue
        corrs = corrs_by_frame.get(fidx)
        res = None
        if corrs:
            res = _register_corrs(corrs, image_size, max_reproj_px=max_reproj_px,
                                  min_landmarks=min_landmarks, initial_intrinsics=prior)
        if res is None and last_H is not None and feats_by_frame is not None:
            feats = feats_by_frame.get(fidx)
            if feats is not None:
                yard_lines, rows, _hashes = feats
                ident = predict_identities(yard_lines, rows, last_H)
                corrs2 = correspondences_from_identities(ident, yard_lines, rows)
                if len(corrs2) >= min_landmarks:
                    res = _register_corrs(corrs2, image_size, max_reproj_px=max_reproj_px,
                                          min_landmarks=min_landmarks, initial_intrinsics=prior)
        if res is not None:
            results[fidx] = res
            prior = getattr(res, "intrinsics", prior)
            if feats_by_frame is not None:
                try:
                    last_H = _plane_H(res)
                except AttributeError:
                    pass


def _solve_sweep(corrs_by_frame, num_frames, image_size, *,
                  max_reproj_px=6.0, min_landmarks=6, feats_by_frame=None):
    """Cached per-frame correspondences -> [CalibrationResult|None], solved in two
    sweeps that carry an intrinsics prior instead of solving each frame blind.

    Blind-init PnP (``initial_intrinsics=None``) falls into bad local minima on
    many real frames while neighboring frames — same physically-fixed camera —
    solve fine. Forward sweep: seed each frame's solve with the last successful
    frame's intrinsics. Backward sweep: fill frames that precede the first
    forward success, using the nearest later success as the prior.

    ``feats_by_frame``, if given, additionally enables temporal identity
    propagation (see :func:`_sweep_direction`): a frame whose own
    correspondences are missing/insufficient is rescued via the plane
    homography of the nearest already-solved frame in the current sweep
    direction, chained across multi-frame gaps."""
    results = [None] * num_frames
    _sweep_direction(range(num_frames), results, corrs_by_frame, feats_by_frame,
                     image_size, max_reproj_px, min_landmarks)
    _sweep_direction(range(num_frames - 1, -1, -1), results, corrs_by_frame, feats_by_frame,
                     image_size, max_reproj_px, min_landmarks)
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
    Frames without cached keypoints (or unreadable) are gaps (None), unless a
    frame's own model votes are too sparse to solve but its classical lines
    can be identified via a solved neighbor's plane homography (temporal
    identity propagation — see :func:`_sweep_direction`).

    Broadcast graphics (watermark/score bug) form hash candidates that sit at
    the SAME pixel every frame — unlike real field ticks, which move under
    pan/zoom. A static-cell census over this frame list's detections (see
    :func:`~nfl_gsplat.calibration.fuse_pretrained.static_hash_cells`) flags
    those cells; candidates there are stripped before row fitting/fusion."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
    cfg = cfg or FieldDetectConfig()
    boxes_for = boxes_for or (lambda f: [])

    detected_by_frame: dict[int, tuple] = {}      # fidx -> (yard_lines, hashes)
    for fidx, fr in enumerate(frames):
        kps = kps_by_frame.get(fidx, [])
        if fr is None or not kps:
            continue
        feats = detect_field_features(fr, cfg=cfg, player_boxes=boxes_for(fidx))
        detected_by_frame[fidx] = (feats.yard_lines, feats.hashes)

    static_cells = static_hash_cells(
        {fidx: hashes for fidx, (_yard_lines, hashes) in detected_by_frame.items()})

    corrs_by_frame: dict[int, list] = {}
    feats_by_frame: dict[int, tuple] = {}
    for fidx, (yard_lines, hashes) in detected_by_frame.items():
        clean_hashes = filter_static_hashes(hashes, static_cells)
        rows = fit_hash_rows(clean_hashes, image_width=image_size[0])
        feats_by_frame[fidx] = (yard_lines, rows, clean_hashes)
        corrs_by_frame[fidx] = fuse_frame(yard_lines, clean_hashes, kps_by_frame.get(fidx, []),
                                          territory=territory, image_size=image_size)
    return _solve_sweep(corrs_by_frame, len(frames), image_size,
                        feats_by_frame=feats_by_frame)


def build_autocalib_npz_pretrained(*, play_dir, videos, fps, kps_json, territory,
                                   cfg=None, masks_provider=None, rotations=None):
    """Pretrained-hybrid calibration: cached Roboflow identity + repaired
    classical geometry -> per-frame PnP -> cameras.npz. No GPU, no training.

    Phase 1 is split into three steps so the whole play's hash detections can
    be censused for static broadcast graphics before anything is fused, while
    decoding still stays a single streaming pass and no frame images are
    buffered:

    Phase 1a (streaming): decode each frame once via ``iter_frames``; per
    frame with cached model keypoints, run detect_field_features and stash
    its raw ``(yard_lines, hashes)`` in ``detected_by_frame``. No fusing, no
    row fitting, no PnP solving happens here.

    Phase 1b (in-memory): :func:`~nfl_gsplat.calibration.fuse_pretrained.static_hash_cells`
    over every stored frame's hashes flags cells occupied in >25% of frames —
    broadcast graphics (watermark/score bug) sit at the same pixel every
    frame, unlike real field ticks, which move under pan/zoom.

    Phase 1c (in-memory): each frame is fused with
    :func:`~nfl_gsplat.calibration.fuse_pretrained.filter_static_hashes`-cleaned
    hashes, exactly as the raw hashes were passed before; ``feats_by_frame``
    (handed to :func:`_solve_sweep`) is rebuilt here so its hash rows are
    fitted from the CLEANED hashes too.

    Phase 2 (in-memory, two sweeps): solve every frame's correspondences with
    :func:`_solve_sweep`, which seeds each solve with the last successful
    frame's intrinsics as a prior. This rescues frames whose blind-init PnP
    lands in a bad local minimum — common on real footage where geometry is
    noisy/sparse but the camera is physically fixed, so neighboring frames'
    intrinsics are a strong prior. It also rescues frames whose model votes
    are too sparse to identify enough lines on their own, via temporal
    identity propagation: the classical lines are named by mapping them
    through the plane homography of the nearest already-solved frame,
    chained frame-by-frame across gaps (see :func:`_sweep_direction`).

    Phase 3 (in-memory, one joint solve): the sweep's per-frame results only
    initialize the fixed-center joint solve (:func:`solve_fixed_center`) --
    per-frame PnP is multimodal on planar telephoto views (see spec
    2026-07-06), so the assembled track comes from the joint solve, not the
    sweep.

    ``rotations`` maps camera name -> view rotation in degrees (0/90/180/270);
    a camera missing from the map defaults by name (``endzone`` -> 90, else
    0). Broadcast endzone views are the sideline pipeline rotated 90 deg:
    frames and cached keypoints are rotated INTO the working (rotated) frame
    before detection/fusion/solving, and the joint solve's results are
    de-rotated back to the camera's original pixel coordinates before
    assembly, so cameras.npz always ends up in original pixel space.
    """
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
    from nfl_gsplat.calibration.roboflow_kps import load_kps_json
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    # kps_json: one cache per camera. A dict maps camera name -> json path;
    # a single path is accepted only for a single-camera run (each camera's
    # cache is validated against ITS video's frame count — sharing one file
    # across cameras can only ever fit one of them).
    if not isinstance(kps_json, dict):
        if len(videos) != 1:
            raise SetupError(
                "multiple cameras need per-camera roboflow keypoint caches — "
                "pass kps_json as {camera: path} (CLI: put "
                "roboflow_kps_{camera}.json files in the play dir).")
        kps_json = {next(iter(videos)): kps_json}
    missing = [c for c in videos if c not in kps_json]
    if missing:
        raise SetupError(
            f"no roboflow keypoint cache for camera(s) {missing} — run "
            "scripts/03_roboflow_precompute.py on each camera's video.")

    tracks = {}
    for cam, video in videos.items():
        try:
            meta = ffprobe_meta(video)
            deg = _camera_rotation(cam, rotations)
            orig_wh = (meta.width, meta.height)
            work_w, work_h = rotated_wh(deg, orig_wh)

            kps_by_frame = load_kps_json(kps_json[cam], expect_num_frames=meta.num_frames)
            if deg != 0:
                kps_by_frame = {
                    fidx: [(n, *rotate_uv(u, v, deg, orig_wh), c) for (n, u, v, c) in kps]
                    for fidx, kps in kps_by_frame.items()
                }
            if masks_provider is not None and deg != 0:
                # Player boxes are in ORIGINAL pixel coordinates; the frame they
                # would mask is rotated. Masking with unrotated boxes silently
                # blanks the wrong region. Rotate the box corners before wiring
                # masks through a rotated camera.
                raise CalibrationError(
                    f"camera {cam!r}: masks_provider is not supported with a "
                    f"{deg}-degree view rotation (player boxes would be in "
                    "unrotated coordinates). Rotate the boxes first.")
            boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
            _cfg = cfg or FieldDetectConfig()

            # Phase 1a: stream frames once; detect classical features per frame
            # with cached model kps. No fusing yet -- frames are NOT buffered.
            detected_by_frame: dict[int, tuple] = {}   # fidx -> (yard_lines, hashes)
            for fidx, fr in iter_frames(video, start_frame=0):
                if not (0 <= fidx < meta.num_frames):
                    continue
                kps = kps_by_frame.get(fidx, [])
                if not kps:
                    continue
                fr = rotate_image(fr, deg)
                feats = detect_field_features(fr, cfg=_cfg, player_boxes=boxes_for(fidx))
                detected_by_frame[fidx] = (feats.yard_lines, feats.hashes)

            # Phase 1b: static-graphics census over the whole play's hashes.
            static_cells = static_hash_cells(
                {fidx: hashes for fidx, (_yard_lines, hashes) in detected_by_frame.items()})

            # Phase 1c: fuse each frame with cleaned hashes; feats_by_frame's rows
            # are rebuilt here from the CLEANED hashes for _solve_sweep.
            corrs_by_frame: dict[int, list] = {}
            feats_by_frame: dict[int, tuple] = {}
            for fidx, (yard_lines, hashes) in detected_by_frame.items():
                clean_hashes = filter_static_hashes(hashes, static_cells)
                rows = fit_hash_rows(clean_hashes, image_width=work_w)
                feats_by_frame[fidx] = (yard_lines, rows, clean_hashes)
                corrs_by_frame[fidx] = fuse_frame(yard_lines, clean_hashes, kps_by_frame.get(fidx, []),
                                                  territory=territory,
                                                  image_size=(work_w, work_h))

            results = _solve_sweep(corrs_by_frame, meta.num_frames, (work_w, work_h),
                                  feats_by_frame=feats_by_frame)

            # Phase 3: fixed-center joint solve — per-frame PnP is multimodal on
            # planar telephoto views (see spec 2026-07-06); the sweep output only
            # initializes the joint problem.
            joint, mirrored = solve_fixed_center(corrs_by_frame, (work_w, work_h),
                                                 init_results=results, view_deg=deg)
            if mirrored:
                _LOG.warning(
                    "camera %s: fused left/right hash convention was flipped "
                    "(mirrored labeling) for this camera side; joint solve results "
                    "are in the true world frame.", cam)
            # De-rotate each recovered frame back to the camera's original pixel
            # coordinates -- cameras.npz always stores original-frame geometry,
            # regardless of the working (rotated) pipeline used to solve it.
            joint = [derotate_result(r, deg, orig_wh) if r is not None else None for r in joint]
            # max_gap=30 (~1s of smooth pan): the joint solve audits per frame, so
            # interior gaps are frames whose fusion was rejected, not calibration
            # jumps; interpolation across them is safe and they carry conf=0.
            tracks[cam] = assemble_track_from_results(joint, width=meta.width,
                                                      height=meta.height, max_gap=30)
        except CalibrationError as e:
            raise CalibrationError(f"camera {cam!r}: {e}") from e
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
