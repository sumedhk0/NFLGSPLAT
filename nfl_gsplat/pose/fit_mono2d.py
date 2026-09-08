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
    max_iter: int = 40
    loss: str = "soft_l1"


ANKLES = (7, 8)
PELVIS = 0


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
        if bp_init is not None:
            parts.append(np.sqrt(cfg.init_weight) * (p[bp_slice] - bp_init))
        if prev_params is not None:
            parts.append(np.sqrt(cfg.temporal_weight) * (p[bp_slice] - prev_params[bp_slice]))
            parts.append(np.sqrt(cfg.temporal_weight) * (p[go_slice] - prev_params[go_slice]))
        return np.concatenate(parts)

    sol = least_squares(residuals, init_params, method="trf", loss=cfg.loss, max_nfev=cfg.max_iter * 10,
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
        tr = np.array([ground_xy[0] - rest[PELVIS, 0], ground_xy[1] - rest[PELVIS, 1], 0.0])
        p = _pack_params(bp, go, tr)
        J = forward(p)
        # feet on the turf: shift so the lower ankle is at z = 0
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
                    init_body_pose_seq=None, init_orient_seq=None):
    """``uv_seq [T, 22, 2]``, ``conf_seq [T, 22]``, ``cams`` a list of (K, R, t) per frame,
    ``ground_seq [T, 2]``. Returns ``(params [T, 69], valid [T], reproj_px [T])``."""
    cfg = cfg or Mono2DConfig()
    base_cfg = base_cfg or SMPLXFitConfig()
    T = len(uv_seq)
    params = np.zeros((T, base_cfg.body_pose_dim + base_cfg.global_orient_dim + base_cfg.transl_dim))
    valid = np.zeros(T, bool)
    rep = np.full(T, np.nan)
    prev = None
    for i in range(T):
        init_bp = None if init_body_pose_seq is None else init_body_pose_seq[i]
        init_go = None if init_orient_seq is None else init_orient_seq[i]
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
    return params, valid, rep
