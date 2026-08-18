"""Per-frame registration over a clip → smoothed CameraTrack → cameras.npz.

Detect+register each frame (env-gated seam: video read + cv2 line/hash detection), then smooth
the per-frame (K,R,t) and interpolate short gaps; fail loud on a long gap.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.cross_cam_calib import (
    endzone_feet_by_frame, sideline_field_by_frame, solve_endzone_cross_camera,
)
from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.field_identify import fit_hash_rows, seed_state_from_hint
from nfl_gsplat.calibration.fuse_pretrained import (
    correspondences_from_identities, filter_static_hashes, fuse_frame,
    predict_identities, static_hash_cells,
)
from nfl_gsplat.calibration.joint_solve import solve_fixed_center
from nfl_gsplat.calibration.register_frame import register_frame
from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_box, rotate_image, rotate_uv, rotated_wh,
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


def assemble_track_from_results(results, *, width, height, max_gap: int = 5,
                                smooth: bool = True) -> CameraTrack:
    """Stack per-frame CalibrationResults (None = gap) into a CameraTrack.

    Interior short gaps (<= max_gap consecutive) are linearly interpolated.
    Boundary gaps (leading/trailing None runs) are clamp-extrapolated from the
    nearest valid frame via np.interp, flagged by ``conf=0``.
    A longer interior gap raises CalibrationError naming the range (fail loud).
    After interpolation, K/R/t are smoothed with a 1€ filter to reduce
    frame-to-frame jitter; R is then re-orthonormalized via SVD.

    ``smooth=False`` skips that filter. The 1€ filter is CAUSAL -- it is built
    for real-time use -- so on an offline sequence it lags whatever it follows,
    by an amount that grows with how fast the camera moves. Callers whose
    per-frame cameras are already mutually consistent should turn it off:
    measured on the endzone mosaic path, whose frames are all derived from
    registration homographies against one reference, smoothing moved the
    REFERENCE frame's own camera -- which is exact by construction -- from
    2.72 px of yard-line reprojection error to 99.69 px, purely as pan lag."""
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
    if smooth:
        from nfl_gsplat.pose.temporal_smooth import (OneEuroConfig,
                                                     smooth_param_sequence)
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
            raw_boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
            if deg != 0:
                # Boxes are original-pixel; rotate them into the working frame so
                # they mask the rotated image the detector actually sees.
                def boxes_for(fidx, _bf=raw_boxes_for, _deg=deg, _owh=orig_wh):
                    return [rotate_box(b, _deg, _owh) for b in _bf(fidx)]
            else:
                boxes_for = raw_boxes_for
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


def build_endzone_from_sideline(*, play_dir, tracks_path, cameras_npz, endzone_video,
                                fps, init_C=(-111.0, -20.9, 63.8),
                                sideline_cam="sideline", endzone_cam="endzone",
                                rotations=None):
    """Calibrate the endzone camera from players shared with the calibrated
    sideline camera; merge endzone_* into cameras.npz (keep sideline)."""
    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta

    cams = load_camera_track(cameras_npz)
    sl = cams.get(sideline_cam)
    if sl is None or float(np.mean(sl.conf > 0)) < 0.2:
        raise SetupError(
            f"no calibrated {sideline_cam!r} camera in {cameras_npz} — run the "
            "sideline pretrained calibration first (02_autocalibrate --mode pretrained).")
    if not Path(tracks_path).exists():
        raise SetupError(
            f"player tracks not found: {tracks_path} — run scripts/03b_detect_players.py first.")
    df = pd.read_parquet(tracks_path)

    meta = ffprobe_meta(endzone_video)
    deg = _camera_rotation(endzone_cam, rotations)
    orig_wh = (meta.width, meta.height)
    from nfl_gsplat.calibration.view_rotation import rotated_wh
    work_wh = rotated_wh(deg, orig_wh)

    field_by = sideline_field_by_frame(df, sl, cam=sideline_cam)
    feet_by = endzone_feet_by_frame(df, cam=endzone_cam, deg=deg, orig_wh=orig_wh)
    results = solve_endzone_cross_camera(field_by, feet_by, work_wh,
                                         init_C=np.asarray(init_C, float), view_deg=deg)
    results = [derotate_result(r, deg, orig_wh) if r is not None else None for r in results]
    # pad/trim to the endzone frame count for assembly
    T = meta.num_frames
    results = (results + [None] * T)[:T]
    cams[endzone_cam] = assemble_track_from_results(results, width=meta.width,
                                                    height=meta.height, max_gap=30)
    return write_camera_track(Path(cameras_npz), cams, fps=fps)


def build_endzone_identity_from_plays(*, play_dirs, prior, sideline_cam="sideline",
                                      endzone_cam="endzone", fps, audit_drop_px=15.0):
    """Calibrate the endzone camera from jersey-identity correspondences across
    several first-half plays sharing one center; write endzone_* into each play's
    cameras.npz (sideline preserved)."""
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    from nfl_gsplat.calibration.endzone_multiplay import solve_endzone_identity
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta

    play_dirs = [Path(p) for p in play_dirs]
    corrs_by_play, metas, cams_by_play, sizes = [], [], [], []
    for pdir in play_dirs:
        cams = load_camera_track(pdir / "cameras.npz")
        sl = cams.get(sideline_cam)
        if sl is None:
            raise SetupError(f"no {sideline_cam!r} camera in {pdir/'cameras.npz'} — "
                             "run the sideline calibration first.")
        df = pd.read_parquet(pdir / "tracks.parquet")
        if "player_uid" not in df.columns:
            raise SetupError(f"{pdir/'tracks.parquet'} has no player_uid — run "
                             "scripts/03c_identity_tracks.py first.")
        meta = ffprobe_meta(str(pdir / f"{endzone_cam}.mp4"))
        sizes.append((meta.width, meta.height))
        corrs_by_play.append(identity_correspondences(
            df, sl, sideline_cam=sideline_cam, endzone_cam=endzone_cam))
        metas.append(meta); cams_by_play.append(cams)

    # All plays must share one endzone camera resolution -- the solve below
    # is a single joint problem over all plays' correspondences, so a
    # per-play resolution mismatch would silently solve every play against
    # whichever play's image_size happened to be kept (fail loud instead).
    if len(set(sizes)) > 1:
        raise SetupError(
            f"endzone videos differ in resolution across plays: {sizes} — "
            "all plays must share one endzone camera resolution.")
    image_size = sizes[0]

    per_play = solve_endzone_identity(corrs_by_play, image_size, prior,
                                      audit_drop_px=audit_drop_px)
    written = []
    for pdir, cams, meta, results in zip(play_dirs, cams_by_play, metas, per_play):
        results = (results + [None] * meta.num_frames)[:meta.num_frames]
        cams[endzone_cam] = assemble_track_from_results(
            results, width=meta.width, height=meta.height, max_gap=30)
        written.append(write_camera_track(pdir / "cameras.npz", cams, fps=fps))
    return written


def build_endzone_mosaic(*, play_dir, tracks_path, cameras_npz, endzone_video,
                         fps, prior, anchors=None, stride: int = 6,
                         ref_frame=None, sideline_cam: str = "sideline",
                         endzone_cam: str = "endzone",
                         max_gap: int | None = None,
                         vote_thresh: float = 0.5, min_len_frac: float = 0.25,
                         merge_tol_px: float = 12.0,
                         diag_dir: str | None = None):
    """Calibrate the endzone camera from an accumulated static-paint mosaic.

    Sampled frames are registered into one reference frame (homographies),
    their player-masked white paint is accumulated into a votes image, the
    accumulated yard lines are detected + labelled from two anchors, a single
    reference camera is solved from the labelled lines, and every sampled
    frame's camera is propagated from the reference through its homography.
    ``sideline`` is preserved; ``endzone_*`` is written/overwritten.

    ``vote_thresh``/``min_len_frac``/``merge_tol_px`` are
    :func:`~nfl_gsplat.calibration.field_model_fit.detect_accumulated_lines`
    knobs, exposed here so an operator can retune line detection without
    editing code. ``diag_dir`` defaults to ``play_dir`` (never a hardcoded
    path) for the first-run mosaic dump."""
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration import endzone_mosaic as em
    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M
    from nfl_gsplat.calibration.field_model_fit import (
        HASH_MARK_PITCH_M,
        detect_accumulated_lines,
        detect_hash_columns,
        label_yard_lines,
        prune_to_yard_ladder,
        yard_x_mapper,
    )
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    # The both-orientations mirror search inside solve_reference_camera /
    # propagate relies ENTIRELY on the mirror flipping C_z negative -- a
    # z_range spanning zero would silently accept a mirrored camera, since it
    # is the only bound distinguishing the true camera from its reflection.
    if prior.z_range[0] <= 0:
        raise SetupError(
            "endzone_prior.z_range must exclude negatives — it is the only "
            "thing distinguishing the true camera from its mirror image.")

    diag_dir = Path(diag_dir) if diag_dir is not None else Path(play_dir)

    cams = load_camera_track(cameras_npz)
    if cams.get(sideline_cam) is None:
        raise SetupError(
            f"no {sideline_cam!r} camera in {cameras_npz} — run the sideline "
            "calibration first (02_autocalibrate --mode pretrained).")
    if not Path(tracks_path).exists():
        raise SetupError(
            f"player tracks not found: {tracks_path} — run scripts/03b_detect_players.py first.")
    df = pd.read_parquet(tracks_path)

    meta = ffprobe_meta(str(endzone_video))
    image_size = (meta.width, meta.height)

    # sample frames (iter_frames yields RGB; OpenCV wants BGR)
    frames, boxes = {}, {}
    ez_boxes = df[df["cam"] == endzone_cam]
    if ez_boxes.empty:
        raise SetupError(
            f"no cam=={endzone_cam!r} rows in {tracks_path} — every player "
            "box would be empty, and white jerseys (bright, low-saturation) "
            "are then indistinguishable from paint to the white-mask "
            "threshold; run scripts/03b_detect_players.py on the endzone "
            "video first.")
    for idx, rgb in iter_frames(endzone_video, stride=stride):
        frames[idx] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        g = ez_boxes[ez_boxes["frame"] == idx]
        boxes[idx] = list(zip(g["bbox_x1"], g["bbox_y1"], g["bbox_x2"], g["bbox_y2"]))
    if not frames:
        raise SetupError(f"no frames decoded from {endzone_video}")
    # Median sampled index by default. A field-extent heuristic was tried and
    # measured to select the OPPOSITE of what's needed here: a digital zoom
    # about the principal point raises fx while painted-pixel COUNT stays
    # roughly invariant (lines thicken as fewer stay visible), so a
    # paint-pixel-count score is dominated by how much of the FRAME the field
    # fills, not how much of the FIELD is in frame -- it systematically
    # preferred more-zoomed-in frames, exactly the failure this mosaic is
    # trying to avoid (measured: picked the 2nd-most-zoomed of 6 candidate
    # frames while losing a third of the visible yard lines). ``--ref-frame``
    # is the durable, explicit recovery lever if the median clips a line; a
    # wrong heuristic default is worse than an obvious one.
    ref = ref_frame if ref_frame is not None else sorted(frames)[len(frames) // 2]

    H_by, _inl = em.register_to_reference(frames, ref_idx=ref, boxes_by_frame=boxes)
    votes = em.accumulate_field_paint(
        frames, H_by, boxes, ref_shape=(meta.height, meta.width))

    lines = detect_accumulated_lines(votes, vote_thresh=vote_thresh,
                                     min_len_frac=min_len_frac,
                                     merge_tol_px=merge_tol_px)
    if anchors is None:
        # No anchors yet: save the mosaic so the human can read one off it ONCE
        # per game, then fail loud. Guessing the offset is exactly the failure
        # this design exists to prevent. This must be reachable on a bare
        # tracks.parquet (03b_detect_players.py output, no player_uid) --
        # it is the very first thing an operator runs on a new game, before
        # 03c_identity_tracks.py has ever been run -- so nothing below this
        # point may depend on the identity pipeline.
        diag = Path(diag_dir) / f"{Path(play_dir).name}_mosaic.png"
        diag.parent.mkdir(parents=True, exist_ok=True)
        # Draw the DETECTED segments (+ midpoints) over the raw votes, not just
        # the raw votes field: an operator anchoring faint paint the detector
        # never found gets a confusing "anchor is N px from the nearest line"
        # on the next run instead of a match.
        diag_img = cv2.cvtColor((np.clip(votes, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for seg in lines:
            p0 = (round(seg.p0[0]), round(seg.p0[1]))
            p1 = (round(seg.p1[0]), round(seg.p1[1]))
            mid = (round((seg.p0[0] + seg.p1[0]) / 2), round((seg.p0[1] + seg.p1[1]) / 2))
            cv2.line(diag_img, p0, p1, (0, 0, 255), 1)
            cv2.circle(diag_img, mid, 4, (0, 255, 0), -1)
        cv2.imwrite(str(diag), diag_img)
        raise SetupError(
            f"no endzone_anchor in meta.yaml. Wrote the accumulated mosaic to "
            f"{diag} (detected lines in red, midpoints in green) — identify the "
            "OUTERMOST TWO yard lines in it and add:\n"
            "endzone_anchor:\n  lines:\n"
            "    - {point_px: [x1, y1], world_x_m: -18.288}\n"
            "    - {point_px: [x2, y2], world_x_m: 0.0}\n"
            "(once per game: the tripod shares one camera centre all half). "
            "Two anchors are required: one fixes only the offset, leaving the "
            "labeling direction free and a missing line undetectable.")

    # yard_range is a soft VALIDATOR only (label_yard_lines never uses it to
    # choose a labeling, just to reject an obviously-wrong one) -- it requires
    # player_uid, which only 03c_identity_tracks.py adds, not the bare
    # 03b_detect_players.py tracks.parquet this driver otherwise only needs.
    # Its absence (or any failure deriving it) must not block the mosaic path
    # it was designed to sidestep.
    yard_range = None
    if "player_uid" in df.columns:
        try:
            yard_range = _sideline_yard_range(df, cams[sideline_cam], cam=sideline_cam)
        except SetupError:
            yard_range = None

    # A real mosaic carries the odd spurious line, and one is enough to make
    # the whole set inconsistent, so prune before labelling (measured on
    # play_001: 8 lines fit at best 16.29 px, dropping the single stray at
    # 1.9 degrees fits at 0.862 px).
    lines, ladder_idx, dropped = prune_to_yard_ladder(lines)
    if dropped:
        _LOG.warning("endzone mosaic: dropped %d detected line(s) that fit no "
                    "yard ladder (positions %s)", len(dropped), dropped)
    xs = label_yard_lines(lines, anchors=anchors, yard_range_m=yard_range,
                          indices=ladder_idx)

    # The cross-field constraint comes from the HASH ROWS, not from the yard
    # lines' endpoints. An earlier version mapped each line's two endpoints to
    # world Y = +-HALF_WIDTH_M, assuming every line spanned sideline to
    # sideline; measured on play_001 that is false -- six of seven lines end at
    # x=1919, the IMAGE EDGE, and the camera sees only a ~14 m wide strip, so
    # those endpoints sit at |Y| ~ 4-10 m. Since yard lines are mutually
    # parallel, they leave 3 DOF of the homography unobservable and the
    # endpoints were the only thing supplying them -- wrongly.
    hash_rows = detect_hash_columns(votes, lines)
    to_x = yard_x_mapper(lines, ladder_idx, xs)

    def _world_uv(swap: bool):
        """Correspondences from the hash marks plus yard-line intersections.

        Each hash mark is a POINT of known world position: its row fixes Y, and
        its world X snaps to the 1-yard grid the marks are painted on. They also
        sample the full visible depth every 0.9144 m, which is what separates
        focal length from camera distance -- the seven yard lines alone leave
        that pair nearly degenerate at this field of view.

        Which row is +Y and which is -Y is not knowable here (it depends on the
        broadcast mount), so both assignments are tried, exactly as the caller
        already does for the fused-hash left/right ambiguity."""
        ys = (HASH_OFFSET_M, -HASH_OFFSET_M) if swap else (-HASH_OFFSET_M, HASH_OFFSET_M)
        w, u = [], []
        for row, world_y in zip(hash_rows, ys):
            pts = np.asarray(row.dashes, float).reshape(-1, 2)
            # Project each centroid onto its own row line: a mark broken by the
            # vote threshold leaves a centroid displaced along the mark, which
            # is pure noise across the row.
            homo = np.column_stack([pts, np.ones(len(pts))])
            pts = pts - (homo @ row.line)[:, None] * row.line[:2][None, :]
            # Snap each mark to the 1-yard grid it is painted on. Round about
            # ZERO, not about the set's own phase: the yard-line labels are
            # multiples of YARD_LINE_SPACING_M and so already sit on this grid,
            # and rounding about a fitted phase silently slid the whole dash set
            # one yard off them -- which fits the marks BETTER on their own
            # (rms 1.68 px against 2.06) while reprojecting every yard line
            # 99 px out. The ladder is tied to the operator's anchors and is the
            # authority for absolute position; the marks must share its grid.
            raw_x = to_x(pts)
            gx = np.round(raw_x / HASH_MARK_PITCH_M) * HASH_MARK_PITCH_M
            # Keep only marks whose YARD is unambiguous. Far down the field the
            # projected 1-yard pitch shrinks below the accumulation blur, so
            # neighbouring marks merge and a centroid can land halfway between
            # two grid positions -- assigning it either way is a coin flip that
            # injects a half-yard of world error. A residual past a quarter of
            # the pitch means the assignment is not trustworthy, so the mark is
            # dropped rather than guessed.
            trust = np.abs(raw_x - gx) <= 0.25 * HASH_MARK_PITCH_M
            for keep_it, (px, py), world_x in zip(trust, pts, gx):
                if not keep_it:
                    continue
                w.append([float(world_x), world_y, 0.0])
                u.append([float(px), float(py)])
        # Deliberately NOT adding yard-line x hash-row intersections. They look
        # like free correspondences but they inherit the yard lines' TILT, which
        # is the least-determined thing in this view: the lines are so nearly
        # parallel in the image that their vanishing point sits at x ~ 358,000,
        # so a sub-0.1-degree tilt error swings the intersection far along the
        # row. Measured on play_001: dashes alone solve to f=19006 and reproject
        # every yard line within 2.72 px, while adding the 14 intersections
        # pulls it to f=17864 and 97 px. The yard lines still enter through the
        # ladder that gives each dash its world X, and are then free to serve as
        # an independent check.
        return w, u

    # each labelled yard line, as a normalised image line, for the refinement
    yard_constraints = []
    for seg, world_x in zip(lines, xs):
        ln = np.cross([seg.p0[0], seg.p0[1], 1.0], [seg.p1[0], seg.p1[1], 1.0])
        n = float(np.hypot(ln[0], ln[1]))
        if n > 1e-9:
            yard_constraints.append((float(world_x), ln / n))

    def _yard_line_error(cam):
        """Mean distance from each labelled yard line to where it reprojects.

        The hash marks alone cannot choose between the two orientations -- they
        are symmetric about the field's centre line, so the mirror fits them
        just as well and lands inside a symmetric y_range prior. The yard lines
        are NOT symmetric in the frame, so they separate the two decisively
        (measured on play_001: 2.7 px for the true orientation against 99 px
        for its mirror)."""
        k = cam.intrinsics.K()
        rot, tvec = cam.pose.R, cam.pose.t
        errs = []
        for (world_x, line) in yard_constraints:
            ys = np.linspace(-3.0 * HASH_OFFSET_M, 3.0 * HASH_OFFSET_M, 25)
            pts = np.column_stack([np.full_like(ys, world_x), ys,
                                   np.zeros_like(ys)])
            cam_pts = (rot @ pts.T + tvec[:, None]).T
            ok = cam_pts[:, 2] > 1e-6
            if not ok.any():
                continue
            uvw = (k @ cam_pts[ok].T).T
            uv2 = uvw[:, :2] / uvw[:, 2:3]
            errs.append(np.abs(uv2 @ line[:2] + line[2]))
        return float(np.mean(np.concatenate(errs))) if errs else float("inf")

    # Solve BOTH orientations and keep the better one, rather than the first
    # that happens to satisfy the prior box: the box cannot tell them apart,
    # since mirroring the hash rows mirrors the camera in Y and a y_range
    # centred on the field admits both.
    candidates, failures = [], []
    for swap in (False, True):
        try:
            # yard_constraints are used to SCORE the two orientations below,
            # not to drive the fit. Folding them into the refinement was
            # measured to make things worse, not better: they contribute ~175
            # residuals against the marks' 72, so they dominate the objective
            # and dragged a solve that reprojects the yard lines at 2.7 px out
            # to 99 px. The marks alone fit well; the lines are the referee.
            cand = em.solve_reference_camera(*_world_uv(swap), image_size, prior)
        except CalibrationError as exc:
            failures.append(f"  rows {'mirrored' if swap else 'as detected'}: {exc}")
            continue
        candidates.append((_yard_line_error(cand), swap, cand))
    if not candidates:
        raise CalibrationError(
            "endzone mosaic: neither hash-row orientation solved.\n"
            + "\n".join(failures))
    candidates.sort(key=lambda c: c[0])
    best_err, best_swap, ref_cam = candidates[0]
    if len(candidates) == 2 and candidates[1][0] < 3.0 * best_err:
        raise CalibrationError(
            "endzone mosaic: the two hash-row orientations fit the yard lines "
            f"almost equally ({candidates[0][0]:.1f} px vs "
            f"{candidates[1][0]:.1f} px), so which sideline is +Y cannot be "
            "decided from this mosaic. Too few yard lines survived to break "
            "the symmetry.")
    _LOG.info("endzone mosaic: hash rows %s; yard-line reprojection %.2f px "
              "(mirror %.2f px)", "mirrored" if best_swap else "as detected",
              best_err, candidates[1][0] if len(candidates) == 2 else float("nan"))

    # focal_range is the OPERATOR's own declared prior, not an arbitrary
    # window derived from wherever the reference happened to land -- a
    # +-20%-of-reference window was measured too narrow for a real play's
    # zoom span (acceptance band s in [0.85, 1.25]).
    results = em.propagate(H_by, ref_cam, n_frames=meta.num_frames,
                           focal_range=prior.focal_range)
    # A fast pan can break the pure-rotation model outright -- during one on
    # play_001 the propagated frames failed the unit-aspect gate over ~95
    # consecutive frames, which is a real uncovered span, not a tuning
    # artefact (rolling shutter shears the frame while the camera slews). The
    # default stays 3x stride so that surfaces as an error; an operator who
    # has looked at the span can widen it deliberately.
    # smooth=False: these cameras all come from homographies registered to one
    # reference, so they are already mutually consistent, and the causal 1€
    # filter would only add pan lag (measured: it moved the reference frame's
    # own exact camera by 97 px).
    cams[endzone_cam] = assemble_track_from_results(
        results, width=meta.width, height=meta.height, smooth=False,
        max_gap=stride * 3 if max_gap is None else int(max_gap))
    return write_camera_track(Path(cameras_npz), cams, fps=fps)


def _sideline_yard_range(df, sideline_track, *, cam, pad_m: float = 8.0):
    """World-X window the sideline camera says is in play, padded."""
    import numpy as np

    from nfl_gsplat.calibration.endzone_identity import field_positions_by_uid
    from nfl_gsplat.errors import SetupError

    field = field_positions_by_uid(df, sideline_track, cam=cam, smooth_window=1)
    xs = [p[0] for fmap in field.values() for p in fmap.values()]
    if not xs:
        raise SetupError(
            "cannot derive a yard range: no sideline field positions — "
            "run scripts/03c_identity_tracks.py so tracks carry player_uid.")
    return (float(np.min(xs)) - pad_m, float(np.max(xs)) + pad_m)


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
