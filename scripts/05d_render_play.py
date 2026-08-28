"""Render a whole play from cached pose parameters: field + real bodies, per frame.

Reads the cache written by ``05c_pose_play.py`` and turns it into a frame
sequence from a fixed novel viewpoint. No GPU and no pose network -- everything
here is cheap, so the camera path and look can be re-run freely.

Three things happen per player before they are drawn:

* SHAPE comes from the ROSTER when the identity resolved, solved once per
  player rather than per frame. Body shape does not change during a play, and
  re-fitting each frame would let it jitter for no reason.
* POSE is SMOOTHED with a ZERO-PHASE filter. The 1-euro filter is causal and
  lags a recording; on a sprinting player that lag is metres of body position.
* PLACEMENT comes from the calibration, per frame, from the foot point.

Runs in the SMPL-X env (.venv-smplx).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

BODY_RGB = (0.72, 0.62, 0.55)
TEAM_RGB = {"ARI": (0.62, 0.11, 0.22), "SEA": (0.11, 0.22, 0.34)}

# A track must appear in at least this many sampled frames to be drawn. A
# fragment seen twice contributes flicker, not motion.
MIN_FRAMES_PER_TRACK = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--poses", required=True, type=Path)
    ap.add_argument("--identity", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--field-res-m", type=float, default=0.15)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--min-cutoff", type=float, default=2.0,
                    help="zero-phase filters twice, so this runs HIGHER than a "
                         "causal cutoff would")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import smplx
    import torch

    from nfl_gsplat.compositing.merge_ply import batch_from_arrays
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.compositing.preview_cpu import (intrinsics, look_at,
                                                    render_gaussians_cpu)
    from nfl_gsplat.field.procedural_field import (render_field_texture,
                                                   texture_to_gaussians)
    from nfl_gsplat.pose.fit_shape import fit_height_and_weight
    from nfl_gsplat.pose.place_on_field import place_mesh
    from nfl_gsplat.pose.temporal_smooth import (OneEuroConfig,
                                                 smooth_param_sequence_zero_phase)

    blob = pickle.load(open(args.poses, "rb"))
    cam_name, cache = blob["cam"], blob["frames"]
    frames = sorted(cache)
    if not frames:
        raise SetupError(f"{args.poses} holds no posed frames")
    _LOG.info("%d posed frames, %d..%d", len(frames), frames[0], frames[-1])

    track = load_camera_track(args.play_dir / "cameras.npz")[cam_name]
    merged = pickle.load(open(args.identity, "rb"))["merged"]
    player_of = {p.tracks[cam_name]: p for p in merged.values()
                 if cam_name in p.tracks}

    # ---- gather each track's sequence ------------------------------------
    tracks: dict[int, list[int]] = {}
    for f in frames:
        for tid in cache[f]:
            tracks.setdefault(tid, []).append(f)
    tracks = {t: fs for t, fs in tracks.items() if len(fs) >= MIN_FRAMES_PER_TRACK}
    _LOG.info("%d tracks with >= %d frames (%d identified)", len(tracks),
              MIN_FRAMES_PER_TRACK, sum(1 for t in tracks if t in player_of))

    model = smplx.create(str(args.body_models), model_type="smplx",
                         gender="neutral", use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)

    # ---- shape: once per player, not once per frame ----------------------
    betas_of: dict[int, np.ndarray] = {}
    for tid, fs in tracks.items():
        player = player_of.get(tid)
        if player is not None and player.weight_lb > 0:
            betas, _, _ = fit_height_and_weight(model, faces, player.height_m,
                                                player.weight_lb, n_betas=4)
        else:
            # No identity: average the network's own estimate over the track.
            # A per-frame beta would make the body breathe in and out.
            betas = np.mean([cache[f][tid]["betas"] for f in fs], axis=0)
        betas_of[tid] = np.asarray(betas, float)
    n_roster = sum(1 for t in tracks if t in player_of and player_of[t].weight_lb > 0)
    _LOG.info("%d roster bodies, %d from the network's own shape",
              n_roster, len(tracks) - n_roster)

    # ---- pose: zero-phase smoothing per track ----------------------------
    cfg = OneEuroConfig(min_cutoff=args.min_cutoff, beta=0.01, fps=59.94 / blob["stride"])
    smoothed: dict[int, dict[int, dict]] = {}
    for tid, fs in tracks.items():
        body = np.stack([cache[f][tid]["body_pose"].reshape(-1) for f in fs])
        orient = np.stack([cache[f][tid]["global_orient"].reshape(-1) for f in fs])
        body_s = smooth_param_sequence_zero_phase(body, cfg)
        orient_s = smooth_param_sequence_zero_phase(orient, cfg)
        for i, f in enumerate(fs):
            smoothed.setdefault(f, {})[tid] = {
                "body_pose": body_s[i], "global_orient": orient_s[i],
                "bbox": cache[f][tid]["bbox"]}

    # ---- field, built once -----------------------------------------------
    f_xyz, f_rot, f_scale, f_opac, f_dc = texture_to_gaussians(
        render_field_texture(res_m=args.field_res_m), args.field_res_m)
    field = batch_from_arrays(f_xyz, f_rot, f_scale, f_opac,
                              np.asarray(f_dc, np.float32)[:, :, None])

    # Aim at where the players actually are, over the WHOLE play, so the camera
    # does not stare at an empty patch of turf. Computed up front from foot
    # points -- which need no mesh, only the calibration -- rather than from a
    # running mean of bodies already drawn, which would drift the camera through
    # the sequence as later frames pulled the average around.
    from nfl_gsplat.pose.place_on_field import ground_point

    feet = []
    for f in frames:
        if f not in smoothed:
            continue
        intr, pose = track.at(f)
        k_cam, rot_cam, tvec_cam = intr.K(), pose.R, pose.t
        for rec in smoothed[f].values():
            x1, _y1, x2, y2 = rec["bbox"]
            try:
                feet.append(ground_point((0.5 * (x1 + x2), float(y2)),
                                         k_cam, rot_cam, tvec_cam))
            except Exception:      # ray parallel to the turf; skip this one
                continue
    if not feet:
        raise SetupError("no player foot point could be grounded; the camera "
                         "track and the pose cache disagree about frames")
    feet = np.asarray(feet)
    target = np.array([float(np.median(feet[:, 0])),
                       float(np.median(feet[:, 1])), 1.0])
    eye = target + np.array([2.0, -34.0, 13.0])
    rot_v, tvec_v = look_at(eye, target)
    _LOG.info("camera fixed on (%.1f, %.1f) m, from %d foot points",
              target[0], target[1], len(feet))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_mat = intrinsics(args.width, args.height, fov_deg=55.0)
    t0 = time.time()

    for n, f in enumerate(frames):
        if f not in smoothed:
            continue
        intr, pose = track.at(f)
        k_cam, rot_cam, tvec_cam = intr.K(), pose.R, pose.t
        batches = []
        for tid, rec in smoothed[f].items():
            with torch.no_grad():
                res = model(
                    betas=torch.tensor(betas_of[tid][None, :model.num_betas],
                                       dtype=torch.float32),
                    body_pose=torch.tensor(rec["body_pose"].reshape(1, -1),
                                           dtype=torch.float32),
                    global_orient=torch.tensor(rec["global_orient"].reshape(1, 3),
                                               dtype=torch.float32))
            verts = res.vertices[0].cpu().numpy().astype(np.float64)
            joints = res.joints[0].cpu().numpy().astype(np.float64)
            x1, y1, x2, y2 = rec["bbox"]
            placed = place_mesh(verts, joints, (0.5 * (x1 + x2), float(y2)),
                                k_cam, rot_cam, tvec_cam)
            player = player_of.get(tid)
            colour = (TEAM_RGB.get(player.team, BODY_RGB) if player is not None
                      else BODY_RGB)
            batches.append(mesh_to_gaussians(placed, faces, colour=colour))

        scene = merge([field] + batches)
        img = render_gaussians_cpu(scene, k_mat, rot_v, tvec_v,
                                   width=args.width, height=args.height)
        imageio.imwrite(args.out_dir / f"frame_{f:05d}.png", img)
        if n % 10 == 0:
            done = n + 1
            rate = done / max(1e-6, time.time() - t0)
            _LOG.info("rendered %d/%d frames (%.2f f/s)", done, len(frames), rate)

    written = sorted(args.out_dir.glob("frame_*.png"))
    _LOG.info("wrote %d frames to %s (%.0f s)", len(written), args.out_dir,
              time.time() - t0)
    if written:
        gif = args.out_dir / "play.gif"
        imageio.mimsave(gif, [imageio.imread(p) for p in written], fps=10, loop=0)
        _LOG.info("wrote %s", gif)


if __name__ == "__main__":
    main()
