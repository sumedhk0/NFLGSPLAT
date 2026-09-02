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
TEAM_RGB = {"ARI": (0.62, 0.11, 0.22), "SEA": (0.11, 0.22, 0.34),
            "KC": (0.89, 0.09, 0.22), "BAL": (0.14, 0.13, 0.44)}

# A track must survive this FRACTION of the posed frames to be drawn. Twenty-two
# players are on the field and a real one is tracked for most of the play; a
# fragment is not. The old floor of 4 frames was set to stop flicker and does not
# select players -- measured on play_001 it drew up to 34 bodies in a frame, and
# 26% of frames held more than 22 people.
#
#     floor   tracks   median/frame   max/frame   frames over 22
#         4       63             22          34              208
#        50       31             22          24              111
#       200       24             21          22                0
#       300       22             20          22                0
#
# A quarter of the play never exceeds 22 while keeping the most coverage, so the
# physical constraint is satisfied by evidence rather than by capping the count.
MIN_TRACK_FRACTION = 0.25

# Nobody outruns this. A foot point that moves faster has jumped, not run, and
# feeding that to the smoother drags the whole trajectory toward the outlier --
# measured on play_001, 1.6% of steps exceeded it and the worst reached 97 m/s.
MAX_SPEED_M_S = 11.0


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
    ap.add_argument("--min-facing", type=float, default=0.1,
                    help="cosine a vertex must face the camera by to be "
                         "sampled; higher erodes the silhouette, where grass "
                         "bleeds into the body's colour")
    ap.add_argument("--no-appearance", dest="appearance", action="store_false",
                    help="flat team colours instead of colours sampled from "
                         "the footage (compositing.appearance)")
    ap.add_argument("--min-cutoff", type=float, default=2.0,
                    help="zero-phase filters twice, so this runs HIGHER than a "
                         "causal cutoff would")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import smplx
    import torch

    from nfl_gsplat.compositing.appearance import (median_colours,
                                                   vertex_colours_from_view)
    from nfl_gsplat.compositing.merge_ply import batch_from_arrays
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.compositing.preview_cpu import (intrinsics, look_at,
                                                    render_gaussians_cpu)
    from nfl_gsplat.field.procedural_field import (render_field_texture,
                                                   texture_to_gaussians)
    from nfl_gsplat.pose.fit_shape import fit_height_and_weight
    from nfl_gsplat.pose.place_on_field import ground_point, place_mesh
    from nfl_gsplat.pose.temporal_smooth import (OneEuroConfig,
                                                 smooth_param_sequence_zero_phase)

    blob = pickle.load(open(args.poses, "rb"))
    cam_name, cache = blob["cam"], blob["frames"]
    frames = sorted(cache)
    if not frames:
        raise SetupError(f"{args.poses} holds no posed frames")
    _LOG.info("%d posed frames, %d..%d", len(frames), frames[0], frames[-1])

    track = load_camera_track(args.play_dir / "cameras.npz")[cam_name]
    ident_blob = pickle.load(open(args.identity, "rb"))
    merged = ident_blob["merged"]
    # Identities are keyed by STITCHED player id; this cache is keyed by the
    # tracker's raw fragment ids. Map every fragment of a player onto that
    # player, or nothing here matches and every body is drawn generic.
    stitched = (ident_blob.get("stitch") or {}).get(cam_name, {})
    by_player = {p.tracks[cam_name]: p for p in merged.values()
                 if cam_name in p.tracks}
    player_of = {frag: by_player[pid] for frag, pid in stitched.items()
                 if pid in by_player}
    for pid, p in by_player.items():          # unstitched ids map to themselves
        player_of.setdefault(pid, p)

    # ---- gather each track's sequence ------------------------------------
    tracks: dict[int, list[int]] = {}
    for f in frames:
        for tid in cache[f]:
            tracks.setdefault(tid, []).append(f)
    floor = max(4, int(MIN_TRACK_FRACTION * len(frames)))
    dropped = {t: len(fs) for t, fs in tracks.items() if len(fs) < floor}
    tracks = {t: fs for t, fs in tracks.items() if len(fs) >= floor}
    _LOG.info("%d tracks survive the %d-frame floor (%d identified); dropped %d "
              "shorter fragments", len(tracks), floor,
              sum(1 for t in tracks if t in player_of), len(dropped))

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

    # ---- pose AND placement: zero-phase smoothing per track --------------
    # Smoothing the pose alone left the visible jitter untouched, because a body
    # is put on the field by its FOOT POINT and that was passed through raw. The
    # foot point is the box's bottom edge, which the detector re-estimates every
    # frame; its noise becomes the player skating about.
    cfg = OneEuroConfig(min_cutoff=args.min_cutoff, beta=0.01, fps=59.94 / blob["stride"])
    smoothed: dict[int, dict[int, dict]] = {}
    n_outliers = 0
    for tid, fs in tracks.items():
        body = np.stack([cache[f][tid]["body_pose"].reshape(-1) for f in fs])
        orient = np.stack([cache[f][tid]["global_orient"].reshape(-1) for f in fs])
        foot = np.stack([[0.5 * (cache[f][tid]["bbox"][0] + cache[f][tid]["bbox"][2]),
                          float(cache[f][tid]["bbox"][3])] for f in fs])

        # Drop impossible jumps BEFORE smoothing. A zero-phase filter spreads an
        # outlier both forwards and backwards, so one bad frame smears over its
        # neighbours instead of being averaged away.
        #
        # Gated on GROUND SPEED, not on pixels. A first version compared the
        # foot point's pixel motion against the box height, which never fired
        # once across 816 frames -- a threshold that cannot trigger is not a
        # guard. The camera is available per frame, so the real quantity is
        # cheap: a player who covers more than MAX_SPEED_M_S has jumped.
        ground = np.full((len(fs), 2), np.nan)
        for i, f in enumerate(fs):
            intr_i, pose_i = track.at(f)
            try:
                ground[i] = ground_point(tuple(foot[i]), intr_i.K(),
                                         pose_i.R, pose_i.t)[:2]
            except Exception:      # ray parallel to the turf
                pass
        # Compare against the last ACCEPTED frame, with the elapsed time since
        # THAT frame. Measuring from the immediately preceding index instead
        # cascades: pinning one bad frame leaves the next genuine position two
        # frames of motion away but still divided by one frame of time, so it
        # reads as too fast and is rejected in turn. That version rejected 12%
        # of steps against the 1.3% that are actually implausible -- it would
        # have frozen every sprinting player.
        last = 0
        for i in range(1, len(fs)):
            dt = max((fs[i] - fs[last]) / (59.94 / blob["stride"]), 1e-6)
            step = float(np.linalg.norm(ground[i] - ground[last]))
            if np.isfinite(step) and step / dt > MAX_SPEED_M_S:
                foot[i] = foot[last]
                ground[i] = ground[last]
                n_outliers += 1
            else:
                last = i

        body_s = smooth_param_sequence_zero_phase(body, cfg)
        orient_s = smooth_param_sequence_zero_phase(orient, cfg)
        foot_s = smooth_param_sequence_zero_phase(foot, cfg)
        for i, f in enumerate(fs):
            smoothed.setdefault(f, {})[tid] = {
                "body_pose": body_s[i], "global_orient": orient_s[i],
                "bbox": cache[f][tid]["bbox"], "foot": foot_s[i]}
    _LOG.info("smoothed %d tracks; replaced %d impossible foot-point jumps",
              len(tracks), n_outliers)

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
    feet = []
    for f in frames:
        if f not in smoothed:
            continue
        intr, pose = track.at(f)
        k_cam, rot_cam, tvec_cam = intr.K(), pose.R, pose.t
        for rec in smoothed[f].values():
            try:
                feet.append(ground_point(tuple(rec["foot"]),
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
    # Appearance comes from the frame each body was posed in: project the
    # placed mesh through that frame's camera, read the pixels under the
    # vertices that face it, team colour where the frame did not see them.
    # One frame per body for now; the median over a track's frames is the
    # next step and appearance.median_colours is built for it.
    reader = (imageio.get_reader(str(args.play_dir / f"{cam_name}.mp4"))
              if args.appearance else None)

    def posed_mesh(tid, rec):
        with torch.no_grad():
            res = model(
                betas=torch.tensor(betas_of[tid][None, :model.num_betas],
                                   dtype=torch.float32),
                body_pose=torch.tensor(rec["body_pose"].reshape(1, -1),
                                       dtype=torch.float32),
                global_orient=torch.tensor(rec["global_orient"].reshape(1, 3),
                                           dtype=torch.float32))
        return (res.vertices[0].cpu().numpy().astype(np.float64),
                res.joints[0].cpu().numpy().astype(np.float64))

    # Pass 1: a body's colours from EVERY frame it was posed in, averaged per
    # vertex over the frames that saw that vertex. One frame carries motion
    # blur, whoever ran in front, and a single viewing side; over a track the
    # occluders move on and both sides get seen. Sums and counts per body,
    # so memory is two arrays per player, not one per frame.
    colour_sum: dict[int, np.ndarray] = {}
    colour_n: dict[int, np.ndarray] = {}
    if reader is not None:
        t_app = time.time()
        for f in frames:
            if f not in smoothed:
                continue
            try:
                frame_rgb = reader.get_data(int(f))
            except (IndexError, ValueError):
                continue
            intr, pose = track.at(f)
            k_cam, rot_cam, tvec_cam = intr.K(), pose.R, pose.t
            for tid, rec in smoothed[f].items():
                verts, joints = posed_mesh(tid, rec)
                placed = place_mesh(verts, joints, tuple(rec["foot"]),
                                    k_cam, rot_cam, tvec_cam)
                seen = vertex_colours_from_view(placed, faces, k_cam, rot_cam,
                                                tvec_cam, frame_rgb,
                                                min_facing=args.min_facing)
                ok = np.isfinite(seen).all(1)
                if tid not in colour_sum:
                    colour_sum[tid] = np.zeros_like(seen)
                    colour_n[tid] = np.zeros(len(seen))
                colour_sum[tid][ok] += seen[ok]
                colour_n[tid][ok] += 1
        _LOG.info("appearance: %d bodies coloured from the footage (%.0f s)",
                  len(colour_sum), time.time() - t_app)
    t0 = time.time()

    for n, f in enumerate(frames):
        if f not in smoothed:
            continue
        intr, pose = track.at(f)
        k_cam, rot_cam, tvec_cam = intr.K(), pose.R, pose.t
        batches = []
        for tid, rec in smoothed[f].items():
            verts, joints = posed_mesh(tid, rec)
            placed = place_mesh(verts, joints, tuple(rec["foot"]),
                                k_cam, rot_cam, tvec_cam)
            player = player_of.get(tid)
            colour = (TEAM_RGB.get(player.team, BODY_RGB) if player is not None
                      else BODY_RGB)
            if tid in colour_sum:
                n_seen = colour_n[tid]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = colour_sum[tid] / n_seen[:, None]
                mean[n_seen == 0] = np.nan
                colour, _unseen = median_colours(mean[None], fallback=colour)
            batches.append(mesh_to_gaussians(placed, faces, colour=colour))

        scene = merge([field] + batches)
        img = render_gaussians_cpu(scene, k_mat, rot_v, tvec_v,
                                   width=args.width, height=args.height)
        # render_gaussians_cpu returns BGR; imageio writes RGB. Every render
        # before this swapped red and blue, and the green field hid it until
        # the first footage-coloured bodies came out the wrong team colour.
        imageio.imwrite(args.out_dir / f"frame_{f:05d}.png", img[..., ::-1])
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
