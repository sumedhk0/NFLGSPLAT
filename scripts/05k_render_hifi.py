#!/usr/bin/env python
"""High-fidelity render of a play: every player, every frame, upright, 1080p.

Replaces the world-mode preview for the deliverable. Per frame:

  timeline   render.timeline: every detected player's ground position (both
             views' feet fused, smoothed, short gaps bridged), pose from the
             fused refit / the single-view pose / the play's median stance,
             SLERP-interpolated to every frame, tilt clamped to 35 degrees
  bodies     SMPL-X forward per player, feet on the turf under the pelvis
  look       fitted appearance (05i) where a player has one, else the team
             colour; the procedural field in the rule-book frame
  render     compositing.splat_torch on the GPU, the whole scene, 1920x1080,
             sparse front-to-back compositing (no CUDA toolkit needed)
  output     PNG per rendered frame and an mp4 at the source rate / stride

Runs in the smplx env (SMPL-X forward; torch with CUDA).

    python scripts/05k_render_hifi.py --play-dir <P> --out-dir <P>/render_hifi
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
BODY_RGB = (0.72, 0.62, 0.55)
TEAM_RGB = {"KC": (0.89, 0.09, 0.22), "BAL": (0.95, 0.95, 0.95), "ARI": (0.62, 0.11, 0.22),
            "SEA": (0.11, 0.22, 0.34)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--poses-refit", type=Path, default=None)
    ap.add_argument("--poses-sideline", type=Path, default=None)
    ap.add_argument("--identity", type=Path, default=None)
    ap.add_argument("--appearance", type=Path, default=None, help="05i directory")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--stride", type=int, default=2, help="render every n-th source frame")
    ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--field-res-m", type=float, default=0.12)
    ap.add_argument("--field-texture", type=Path, default=None,
                    help="05l field_texture.npz (the footage's field); default procedural")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--follow", action="store_true",
                    help="the virtual camera dollies with the play's smoothed centroid (render.camera_path)")
    ap.add_argument("--helmets", action="store_true",
                    help="head vertices wear the team's helmet colour, inflated 2 cm (render.helmet)")
    ap.add_argument("--stitch", action="store_true",
                    help="join the linker's fragments into players (tracking.stitch) so a "
                         "player keeps one id and one texture across breaks")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-render frames whose PNG already exists (default: skip them)")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import smplx
    import torch

    from nfl_gsplat.compositing import splat_torch as st
    from nfl_gsplat.compositing.merge_ply import batch_from_arrays
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
    from nfl_gsplat.field.procedural_field import render_field_texture, texture_to_gaussians
    from nfl_gsplat.render import helmet as hm
    from nfl_gsplat.render.play_timeline import load_play_timeline, placed_vertices

    P = args.play_dir
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)

    team_of: dict[int, str] = {}
    ident_path = args.identity or (P / "identity_resolved.pkl")
    if ident_path.exists():
        merged = pickle.load(open(ident_path, "rb")).get("merged", {})
        team_of = {int(pid): p.team for pid, p in merged.items() if getattr(p, "team", None)}
    fitted: dict[int, dict] = {}
    app_dir = args.appearance or (P / "appearance")
    if app_dir.exists():
        for path in app_dir.glob("appearance_*.npz"):
            z = np.load(path)
            fitted[int(path.stem.split("_")[1])] = {k: np.asarray(z[k], np.float32)
                                                    for k in ("colour", "log_scale_mult", "opacity_logit")}

    tl, tracks, df, frames_all, poses = load_play_timeline(
        P, model, poses_refit=args.poses_refit, poses_sideline=args.poses_sideline,
        stitch_ids=args.stitch)
    frames = frames_all[:: max(1, args.stride)]
    if args.limit:
        frames = frames[: args.limit]
    print(f"timeline: {len(frames_all)} frames, {len(poses)} posed players, "
          f"{tl.n_default} default-posed, {tl.n_clamped} tilt-clamped states; "
          f"appearance for {len(fitted)} players, teams for {len(team_of)}")

    # Field once; virtual camera on the players' centroid over the play.
    if args.field_texture is not None:
        from nfl_gsplat.field.footage_texture import load_texture

        field_tex, field_res = load_texture(args.field_texture)
        print(f"field from footage: {args.field_texture} "
              f"({field_tex.shape[1]}x{field_tex.shape[0]} at {field_res} m)")
    else:
        field_tex, field_res = render_field_texture(res_m=args.field_res_m), args.field_res_m
    f_xyz, f_rot, f_scale, f_opac, f_dc = texture_to_gaussians(field_tex, field_res)
    field = batch_from_arrays(f_xyz, f_rot, f_scale, f_opac, np.asarray(f_dc, np.float32)[:, :, None])
    centre = np.median(np.concatenate([np.stack([s.xy for s in tl.states[f]])
                                       for f in frames if f in tl.states]), axis=0)
    target = np.array([centre[0], centre[1], 1.0])
    eye = target + np.array([2.0, -34.0, 13.0])
    R_v, t_v = look_at(eye, target)
    K_v = intrinsics(args.width, args.height, fov_deg=55.0)
    print(f"camera on ({centre[0]:.1f}, {centre[1]:.1f}) m")
    path = None
    if args.follow:
        from nfl_gsplat.render.camera_path import follow_path

        path = follow_path(frames_all, {f: np.mean([s.xy for s in tl.states[f]], axis=0)
                                        for f in frames_all if tl.states.get(f)})
        print("camera follows the play (smoothed centroid dolly)")

    head = hm.head_mask(model.v_template.detach().cpu().numpy(),
                        (model.J_regressor @ model.v_template).detach().cpu().numpy())

    def body_batch(s):
        verts = placed_vertices(s, model)
        # Fitted COLOUR only: the fit's scale and opacity were tuned to blurry
        # 140-px crops and read as translucent bodies at this distance; the
        # colours were the acceptance-tested part.
        # A stitched player wears the texture fitted to any of its member ids.
        owner = next((m for m in tl.members.get(s.pid, [s.pid]) if m in fitted),
                     s.pid if s.pid in fitted else None)
        team = next((team_of[m] for m in tl.members.get(s.pid, [s.pid]) if m in team_of),
                    team_of.get(s.pid, ""))
        colour = fitted[owner]["colour"] if owner is not None else TEAM_RGB.get(team, BODY_RGB)
        if args.helmets:
            shell = hm.HELMET_RGB.get(team, hm.DEFAULT_HELMET_RGB)
            verts, colour = hm.wear_helmet(verts, colour, head, shell)
        return mesh_to_gaussians(verts, faces, colour=colour)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = []
    for i, f in enumerate(frames):
        out = args.out_dir / f"frame_{f:05d}.png"
        if args.resume and out.exists():
            written.append(out)                      # a stalled or interrupted run resumes here
            continue
        states = tl.states.get(f, [])
        scene = merge([field] + [body_batch(s) for s in states])
        sp = st.SceneParams.from_batch(scene, device=args.device)
        if path is not None and f in path:
            R_v, t_v = look_at(*path[f])
        with torch.no_grad():
            img = st.render(sp, K_v, R_v, t_v, crop=(0, 0, args.width, args.height),
                            background=(0.06, 0.06, 0.08))
        imageio.imwrite(out, (255 * img.clamp(0, 1).cpu().numpy()).astype(np.uint8))
        written.append(out)
        if i % 20 == 0:
            _LOG.info("rendered %d/%d (%d bodies, %.2f s/frame)", i + 1, len(frames), len(states),
                      (time.time() - t0) / (i + 1))
    mp4 = args.out_dir / "play.mp4"
    with imageio.get_writer(mp4, fps=args.fps / max(1, args.stride), codec="libx264",
                            quality=8, macro_block_size=None) as w:
        for p in written:
            w.append_data(imageio.imread(p))
    print(f"wrote {len(written)} frames and {mp4} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
