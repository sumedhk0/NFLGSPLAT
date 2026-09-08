"""SMPL-X pose refit to ONE camera's 2-D keypoints, feet on the turf.

WHY. Half the bodies in play 1's render are one-view (the sideline alone),
and their poses come from the monocular regressor on 130 px crops, which
regresses toward the mean pose: measured 2026-09-08 on v14, their joints
move 0.25 m/s in the body frame against 1.0 m/s for triangulated bodies
and 2-4 m/s for a running player's limbs -- mannequins gliding. The 2-D
keypoints (05m, YOLOv8-pose) exist for every tracked person and carry the
articulation the regressor lost.

HOW. Per frame, least squares over (body_pose 63, global_orient 3,
transl 3) on the same forward kinematics the renderer animates
(pose.forward_kinematics), with residuals:
  reprojection   (K R t of the camera) of the body joints onto the
                 keypoints, weighted by confidence, in pixels / px_scale;
  ground         the lower ankle at z = 0 (feet on the turf) -- the depth
                 along the camera ray is what one view cannot see, and the
                 turf fixes it;
  placement      the pelvis' (x, y) within a metre of the box-bottom ground
                 point -- the same anchor the timeline places the body on;
  pose prior     L2 on body_pose (as fuse_smplx) and a pull toward the
                 regressor's pose, small, so joints the keypoints do not see
                 (spine, collars, feet) keep a plausible bend;
  temporal       body_pose and global_orient toward the previous frame's.
Warm-started frame to frame; the first frame starts from the regressor's
pose and the rigid placement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params, _param_slices


@dataclass
class Mono2DConfig:
    min_conf: float = 0.3
    min_joints: int = 6
    px_scale: float = 10.0          # a pixel of reprojection error weighs 1/10 of a metre-unit residual
    ground_weight: float = 3.0      # metres of ankle height -> residual
    place_weight: float = 1.0       # metres of pelvis xy from the box-bottom ground point
    prior_weight: float = 0.02      # L2 on body_pose
    init_weight: float = 0.05       # pull toward the regressor's body_pose
    temporal_weight: float = 0.3    # toward the previous frame's body_pose and orient
    # One view trades lean against depth: without this the fits leaned 34 deg
    # (p50) where the triangulated bodies lean 16, 5 % past 60. Measured with
    # 05p --validate on play 1 (365 two-view frames): weight 3 -> 21 deg, 10 ->
    # 20 deg, |diff| 7 deg, none past 60, p90 joint error 0.31 -> 0.26 m,
    # reprojection 3.2 -> 3.4 px.
    tilt_weight: float = 10.0       # on the lean past tilt_free_deg, radians
    tilt_free_deg: float = 20.0
    up_axis: tuple = (0.0, 1.0, 0.0)  # the rest skeleton's up (SMPL-X is y-up)
    max_iter: int = 40
    loss: str = "soft_l1"


ANKLES = (7, 8)
PELVIS = 0


def tilt_rad(global_orient, up_axis=(0.0, 1.0, 0.0)):
    """The body's lean from upright: the rest skeleton's up axis (SMPL-X +y) against world +z."""
    from scipy.spatial.transform import Rotation

    up = Rotation.from_rotvec(np.asarray(global_orient, float)).apply(np.asarray(up_axis, float))
    return float(np.arccos(np.clip(up[2], -1.0, 1.0)))


def project(K, R, t, X):
    p = (K @ (R @ np.asarray(X, float).T + np.asarray(t, float)[:, None])).T
    z = p[:, 2:3]
    return p[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z), z[:, 0]


def fit_frame_2d(uv, conf, cam, init_params, forward, ground_xy, cfg: Mono2DConfig, base_cfg: SMPLXFitConfig,
                 init_body_pose=None, prev_params=None):
    """``(params, reproj_rms_px, n_used)`` for one frame; ``uv [22, 2]``, ``conf [22]``,
    ``cam = (K, R, t)`` world -> image, ``ground_xy`` the body's ground point."""
    K, R, t = cam
    use = np.asarray(conf, float) >= cfg.min_conf
    if int(use.sum()) < cfg.min_joints:
        raise ValueError(f"only {int(use.sum())} keypoints above {cfg.min_conf}")
    w = np.sqrt(np.asarray(conf, float)[use])
    target = np.asarray(uv, float)[use]
    bp_slice, go_slice, _ = _param_slices(base_cfg)
    bp_init = None if init_body_pose is None else np.asarray(init_body_pose, float).reshape(-1)
    gxy = np.asarray(ground_xy, float)

    def residuals(p):
        J = forward(p)
        pix, depth = project(K, R, t, J[use])
        rep = ((pix - target) * w[:, None] / cfg.px_scale).reshape(-1)
        behind = np.maximum(0.0, -depth)                      # nothing behind the camera
        ground = cfg.ground_weight * np.array([min(J[ANKLES[0], 2], J[ANKLES[1], 2])])
        place = cfg.place_weight * (J[PELVIS, :2] - gxy)
        prior = np.sqrt(cfg.prior_weight) * p[bp_slice]
        parts = [rep, behind, ground, place, prior]
        if cfg.tilt_weight > 0:
            parts.append(np.array([cfg.tilt_weight * max(0.0, tilt_rad(p[go_slice], cfg.up_axis)
                                                         - np.radians(cfg.tilt_free_deg))]))
        if bp_init is not None:
            parts.append(np.sqrt(cfg.init_weight) * (p[bp_slice] - bp_init))
        if prev_params is not None:
            parts.append(np.sqrt(cfg.temporal_weight) * (p[bp_slice] - prev_params[bp_slice]))
            parts.append(np.sqrt(cfg.temporal_weight) * (p[go_slice] - prev_params[go_slice]))
        return np.concatenate(parts)

    # max_nfev counts iterations here (one residual call each, the Jacobian
    # numerical: 70 more); at x10 a noisy frame ran 400 iterations = 5 s
    sol = least_squares(residuals, init_params, method="trf", loss=cfg.loss, max_nfev=cfg.max_iter,
                        x_scale="jac")
    pix, _ = project(K, R, t, forward(sol.x)[use])
    err = np.linalg.norm(pix - target, axis=1)
    return sol.x, float(np.sqrt(np.mean(err * err))), int(use.sum())


def rigid_start_2d(rest_joints, ground_xy, cam, uv, conf, forward, base_cfg, init_body_pose=None, *, min_conf=0.3,
                   init_orient=None, heading_span_deg: float = 40.0):
    """Initial params: the regressor's body pose (or zero), the pelvis over
    the ground point, and the heading. With ``init_orient`` (the regressor's
    world orientation) the headings tried are within ``heading_span_deg`` of
    it; without, a coarse search over 12 headings -- which a symmetric body
    cannot disambiguate front from back (measured: the mirrored heading
    wins on noise and drags the fit into the wrong arm basin)."""
    from scipy.spatial.transform import Rotation

    rest = np.asarray(rest_joints, float)
    bp = np.zeros(base_cfg.body_pose_dim) if init_body_pose is None else np.asarray(init_body_pose, float).reshape(-1)
    K, R, t = cam
    use = np.asarray(conf, float) >= min_conf
    best = None
    if init_orient is not None:
        base_rot = Rotation.from_rotvec(np.asarray(init_orient, float))
        headings = [Rotation.from_euler("z", np.radians(d)) * base_rot
                    for d in (-heading_span_deg, -heading_span_deg / 2, 0.0, heading_span_deg / 2, heading_span_deg)]
    else:
        headings = [Rotation.from_euler("z", yaw) for yaw in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    for rot in headings:
        go = rot.as_rotvec()
        p = _pack_params(bp, go, np.zeros(3))
        J = forward(p)
        # the pelvis over the ground point, the lower ankle on the turf (z = 0)
        p[-3:-1] += np.asarray(ground_xy, float) - J[PELVIS, :2]
        p[-1] -= min(J[ANKLES[0], 2], J[ANKLES[1], 2])
        J = forward(p)
        pix, depth = project(K, R, t, J[use])
        if (depth <= 0).any():
            continue
        e = float(np.sqrt(np.mean(np.sum((pix - np.asarray(uv, float)[use]) ** 2, axis=1))))
        if best is None or e < best[0]:
            best = (e, p)
    if best is None:
        raise ValueError("no heading puts the body in front of the camera")
    return best[1], best[0]


def fit_sequence_2d(uv_seq, conf_seq, cams, ground_seq, rest_joints, forward, *, cfg=None, base_cfg=None,
                    init_body_pose_seq=None, init_orient_seq=None, frames=None, max_gap=12, prev_seq=None):
    """``uv_seq [T, 22, 2]``, ``conf_seq [T, 22]``, ``cams`` a list of (K, R, t) per frame,
    ``ground_seq [T, 2]``. Returns ``(params [T, 69], valid [T], reproj_px [T])``.
    With ``frames``, a gap over ``max_gap`` frames restarts from the rigid start
    (the temporal pull would otherwise drag a pose across the gap). ``prev_seq``
    (a params vector or None per frame) replaces the warm start and the temporal
    anchor for that frame -- the neighbouring two-view fit where one exists, so
    a one-view frame between two fused blocks continues them instead of the
    pose from before the block (measured: 187 boundaries on play 1 with pelvis
    jumps of 0.43 m and 35 deg of orientation at the p50)."""
    cfg = cfg or Mono2DConfig()
    base_cfg = base_cfg or SMPLXFitConfig()
    T = len(uv_seq)
    params = np.zeros((T, base_cfg.body_pose_dim + base_cfg.global_orient_dim + base_cfg.transl_dim))
    valid = np.zeros(T, bool)
    rep = np.full(T, np.nan)
    prev = None
    last_frame = None
    for i in range(T):
        init_bp = None if init_body_pose_seq is None else init_body_pose_seq[i]
        init_go = None if init_orient_seq is None else init_orient_seq[i]
        if frames is not None and last_frame is not None and int(frames[i]) - last_frame > max_gap:
            prev = None
        if prev_seq is not None and prev_seq[i] is not None:
            prev = np.asarray(prev_seq[i], float).copy()
        try:
            if prev is None:
                start, _ = rigid_start_2d(rest_joints, ground_seq[i], cams[i], uv_seq[i], conf_seq[i], forward,
                                          base_cfg, init_bp, min_conf=cfg.min_conf, init_orient=init_go)
            else:
                start = prev.copy()
                # keep the pelvis over this frame's ground point
                start[-3:-1] += np.asarray(ground_seq[i], float) - forward(prev)[PELVIS, :2]
            p, e, n = fit_frame_2d(uv_seq[i], conf_seq[i], cams[i], start, forward, ground_seq[i], cfg, base_cfg,
                                   init_body_pose=init_bp, prev_params=prev)
        except ValueError:
            continue
        params[i], valid[i], rep[i] = p, True, e
        prev = p
        last_frame = None if frames is None else int(frames[i])
    return params, valid, rep


def body_frame_speeds(params_seq, frames, forward, *, fps: float, base_cfg=None):
    """Joint speeds in the BODY frame (pose only: orient and transl zeroed),
    metres per second between consecutive records: ``[T-1, 22]``. The ruler
    for gliding: play 1 v14 regressor bodies 0.25 m/s, triangulated 1.0."""
    base_cfg = base_cfg or SMPLXFitConfig()
    bp_slice, _, _ = _param_slices(base_cfg)
    J = []
    for p in params_seq:
        q = np.zeros(len(p))
        q[bp_slice] = np.asarray(p, float)[bp_slice]
        J.append(forward(q))
    J = np.stack(J)
    dt = np.diff(np.asarray(frames, float)) / float(fps)
    return np.linalg.norm(np.diff(J, axis=0), axis=2) / dt[:, None]


def merge_into_refit(blob, fits, betas_of, *, source: str = "mono2d"):
    """``blob`` (the 05f pose cache) with ``fits`` (pid -> (frames, params [T, 69], valid))
    added where the fused refit has no record for that frame and player; the
    fused record wins. Returns the new blob and the number of records added."""
    base = SMPLXFitConfig()
    bp_slice, go_slice, tr_slice = _param_slices(base)
    frames = {int(f): dict(recs) for f, recs in blob.get("frames", {}).items()}
    added = 0
    mono = {}
    for pid, (fs, params, valid) in fits.items():
        for f, p, ok in zip(fs, params, valid):
            f, pid = int(f), int(pid)
            if not ok or pid in frames.get(f, {}):
                continue
            frames.setdefault(f, {})[pid] = {
                "betas": np.asarray(betas_of[pid], np.float32),
                "body_pose": np.asarray(p[bp_slice], np.float32),
                "global_orient": np.asarray(p[go_slice], np.float32),
                "transl": np.asarray(p[tr_slice], np.float32),
            }
            added += 1
            mono[pid] = mono.get(pid, 0) + 1
    out = dict(blob)
    out["frames"] = frames
    out["mono"] = {"source": source, "records": mono}
    return out, added
