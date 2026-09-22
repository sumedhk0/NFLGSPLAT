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
import collections
import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
BODY_RGB = (0.72, 0.62, 0.55)
TEAM_RGB = {"KC": (0.89, 0.09, 0.22), "BAL": (0.95, 0.95, 0.95), "ARI": (0.62, 0.11, 0.22),
            "SEA": (0.11, 0.22, 0.34)}


def play_end_frame(play_dir):
    """The last frame to draw from ``<play-dir>/play_end.json`` (08x: ``{"end": f, "tail": n}``), or None."""
    import json

    f = Path(play_dir) / "play_end.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return int(d["end"]) + int(d.get("tail", 0))


def play_dead_frame(play_dir):
    """The dead-ball frame itself from ``<play-dir>/play_end.json`` (``"end"``), or None."""
    import json

    f = Path(play_dir) / "play_end.json"
    if not f.exists():
        return None
    return int(json.loads(f.read_text())["end"])


def play_start_frame(play_dir):
    """The first frame to draw from ``<play-dir>/play_end.json`` (``"start"``, 3 s before the snap), or None."""
    import json

    f = Path(play_dir) / "play_end.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return int(d["start"]) if d.get("start") is not None else None


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
    ap.add_argument("--gait", action="store_true",
                    help="synthesise a running gait for the legs of moving bodies (render.gait); the fit keeps the torso")
    ap.add_argument("--gait-arms", action="store_true",
                    help="with --gait: the arms pump opposite the legs too (render.gait ARMS)")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="first timeline frame to draw (default: play_end.json's start, else the first)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="last timeline frame to draw; default: <play-dir>/play_end.json (08x) end + tail, else all")
    ap.add_argument("--eye-offset", type=float, nargs=3, default=(2.0, -34.0, 13.0),
                    metavar=("DX", "DY", "DZ"),
                    help="virtual camera eye relative to its target, metres "
                         "(x along the field, y across, z up)")
    ap.add_argument("--export-joints", type=Path, default=None,
                    help="write every drawn body's 22 world joints per frame (JSON) for the interactive viewer")
    ap.add_argument("--no-render", action="store_true",
                    help="build the bodies (poses, gait, throw, catch, carry, fall) but do not rasterise: for --export-joints")
    ap.add_argument("--fov", type=float, default=55.0, help="horizontal field of view, degrees")
    ap.add_argument("--splat-sigma", type=float, default=1.0,
                    help="multiplier on every splat's in-plane extent (bodies and field)")
    ap.add_argument("--splat-opacity", type=float, default=None,
                    help="per-splat opacity for bodies and field (default: each builder's own, ~0.99); "
                         "lower with a larger --splat-sigma so neighbouring splats blend per pixel")
    ap.add_argument("--post-blur", type=float, default=0.0,
                    help="Gaussian blur of the rendered frame, sigma in px (0 = none)")
    ap.add_argument("--follow", action="store_true",
                    help="the virtual camera dollies with the play's smoothed centroid (render.camera_path)")
    ap.add_argument("--view-camera", default=None, choices=["sideline", "endzone"],
                    help="render from that broadcast camera's own solved pose per frame (cameras.npz), K scaled to "
                         "--width/--height: the render then overlays the footage frame for frame")
    ap.add_argument("--uniforms", action="store_true",
                    help="every body wears its team's synthetic kit by region (render.uniform); the "
                         "fitted textures are ignored")
    ap.add_argument("--team-by-colour", action="store_true",
                    help="teams from <play-dir>/team_by_colour.json (08f) instead of identity; "
                         "numbers only where identity's team agrees")
    ap.add_argument("--numbers", action="store_true",
                    help="with --uniforms: jersey numbers on ids whose identity is sure")
    ap.add_argument("--pads", action="store_true",
                    help="shoulder vertices pushed out and up (render.helmet.wear_pads)")
    ap.add_argument("--helmets", action="store_true",
                    help="head vertices wear the team's helmet colour, inflated 2 cm, with the team's facemask "
                         "on the lower face (render.helmet)")
    ap.add_argument("--stitch", action="store_true",
                    help="join the linker's fragments into players (tracking.stitch) so a "
                         "player keeps one id and one texture across breaks")
    ap.add_argument("--ball", action="store_true",
                    help="draw the football from <play-dir>/ball.json (08y_ball_path.py); the holder's arms carry it "
                         "and the passer throws it (render.carry)")
    ap.add_argument("--throw-hand", default="R", choices=["R", "L"], help="the passer's throwing hand")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-render frames whose PNG already exists (default: skip them)")
    args = ap.parse_args()
    if args.export_joints is not None and args.no_render and args.resume:
        # the export is filled inside the frame loop; resume would skip every frame whose PNG exists and
        # write an empty file over the last good one (2026-09-22, the v95 joints)
        args.resume = False

    import imageio.v2 as imageio
    import smplx
    import torch

    from nfl_gsplat.compositing import splat_torch as st
    from nfl_gsplat.compositing.merge_ply import batch_from_arrays
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
    from nfl_gsplat.field.procedural_field import render_field_texture, texture_to_gaussians
    from nfl_gsplat.render import helmet as hm
    from nfl_gsplat.render import uniform as un
    from nfl_gsplat.render import timeline as tlm
    from nfl_gsplat.render.carry import ball_at_hand, ball_between_hands, carry_body_pose, catch_body_pose, throw_body_pose
    from nfl_gsplat.render.play_timeline import load_play_timeline, placed_body
    from nfl_gsplat.render.stance import stance_body_pose, stance_weights

    P = args.play_dir
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)

    team_of: dict[int, str] = {}
    merged: dict = {}
    ident_path = args.identity or (P / "identity_resolved.pkl")
    if ident_path.exists():
        merged = pickle.load(open(ident_path, "rb")).get("merged", {})
        team_of = {int(pid): p.team for pid, p in merged.items() if getattr(p, "team", None)}
    if args.team_by_colour:
        import json

        tbc_path = P / "team_by_colour.json"
        if tbc_path.exists():
            tbc = json.loads(tbc_path.read_text())
            colour_team = {int(k): r["team"] for k, r in tbc["teams"].items()}
            changed = sum(1 for pid, t in colour_team.items() if team_of.get(pid) not in (None, t))
            team_of.update(colour_team)
            print(f"teams by colour: {len(colour_team)} ids (threshold {tbc['threshold']:.0f}), "
                  f"{changed} differ from identity")
        else:
            # 08f refuses when the colours are not bimodal; identity's teams stand.
            print(f"teams by colour: {tbc_path.name} absent (08f refused or not run); identity teams kept")
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
    if args.gait:
        from nfl_gsplat.render.gait import gait_timeline

        if args.gait_arms:
            import nfl_gsplat.render.gait as _gait

            _gait.ARMS = True
        reps = gait_timeline(tl)
        n_on = sum(r["on"] for r in reps.values())
        print(f"gait: legs synthesised on {n_on} body-frames of {sum(1 for r in reps.values() if r['on'])} ids")
    frames = frames_all[:: max(1, args.stride)]
    end = args.end_frame if args.end_frame is not None else play_end_frame(P)
    if end is not None:
        n0 = len(frames)
        frames = [f for f in frames if f <= end]
        print(f"clip ends when the play is dead: frame {end} ({n0 - len(frames)} of {n0} rendered frames after it left out)")
        # the aftermath: a body drawn up to the dead ball keeps its last state through the tail (timeline.hold_to_end);
        # the renderer's business, so the ball path and the play end see the tracks as they are
        dead = play_dead_frame(P)
        if dead is not None:
            # the ball's carrier is held from wherever the detector lost him (under the tackle) to the end:
            # the ball sits in his hands through the dead ball and must not float where a man was
            from nfl_gsplat.render.carry import load_holders

            holders_ = load_holders(P)
            carrier = holders_[max(holders_)] if holders_ else None
            n_end = tlm.hold_to_end(tl, dead, end, teams=team_of or None, always={carrier} if carrier is not None else None)
            if n_end:
                print(f"aftermath: {n_end} body-frames held through the tail after the dead ball at {dead}"
                      + (f" (the carrier {carrier} from his last frame)" if carrier is not None else ""))
    start = args.start_frame if args.start_frame is not None else play_start_frame(P)
    if start is not None:
        n0 = len(frames)
        frames = [f for f in frames if f >= start]
        print(f"clip starts at frame {start} ({n0 - len(frames)} of {n0} rendered frames before it left out)")
    if args.limit:
        frames = frames[: args.limit]
    # An id without a fit (a short fragment) wears its team's mean fitted
    # texture, not a flat colour: at a track break the body then keeps its
    # shading and skin instead of flipping to a paint-bucket player.
    team_texture: dict[str, np.ndarray] = {}
    for team in {t for t in team_of.values() if t}:
        members_ = [fitted[p]["colour"] for p in fitted if team_of.get(p) == team]
        if len(members_) >= 3:
            team_texture[team] = np.mean(np.stack(members_), axis=0).astype(np.float32)
    if team_texture:
        print("team mean textures for " + ", ".join(
            f"{t} ({sum(1 for p in fitted if team_of.get(p) == t)} fits)" for t in sorted(team_texture)))
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

    def tune(batch):
        """--splat-sigma / --splat-opacity applied to a batch in place."""
        if args.splat_sigma != 1.0:
            batch.scale[:, :2] += np.log(args.splat_sigma)
        if args.splat_opacity is not None:
            a = float(np.clip(args.splat_opacity, 1e-4, 1 - 1e-4))
            batch.opacity[:] = np.log(a / (1 - a))
        return batch

    field = tune(field)
    centre = np.median(np.concatenate([np.stack([s.xy for s in tl.states[f]])
                                       for f in frames if f in tl.states]), axis=0)
    target = np.array([centre[0], centre[1], 1.0])
    eye = target + np.array(args.eye_offset, float)
    R_v, t_v = look_at(eye, target)
    K_v = intrinsics(args.width, args.height, fov_deg=args.fov)
    print(f"camera on ({centre[0]:.1f}, {centre[1]:.1f}) m")
    view_track = None
    if args.view_camera:
        from nfl_gsplat.calibration.cameras_io import load_camera_track

        view_track = load_camera_track(P / "cameras.npz")[args.view_camera]
        print(f"camera: the {args.view_camera} broadcast camera's own pose per frame "
              f"({int(view_track.width)}x{int(view_track.height)} scaled to {args.width}x{args.height})")

    def broadcast_view(f):
        """``(K, R, t)`` of the broadcast camera at frame ``f`` (the last solved frame at or before it)."""
        g = min(int(f), len(view_track.conf) - 1)
        while g > 0 and view_track.conf[g] <= 0:
            g -= 1
        K = np.asarray(view_track.K[g], float).copy()
        sx, sy = args.width / float(view_track.width), args.height / float(view_track.height)
        K[0, :] *= sx
        K[1, :] *= sy
        return K, np.asarray(view_track.R[g], float), np.asarray(view_track.t[g], float)

    path = None
    if args.follow:
        from nfl_gsplat.render.camera_path import follow_path

        path = follow_path(frames_all, {f: np.mean([s.xy for s in tl.states[f]], axis=0)
                                        for f in frames_all if tl.states.get(f)},
                           eye_offset=tuple(args.eye_offset))
        print("camera follows the play (smoothed centroid dolly)")

    v_t = model.v_template.detach().cpu().numpy()
    j_t = (model.J_regressor @ model.v_template).detach().cpu().numpy()
    head = hm.head_mask(v_t, j_t)
    face = hm.facemask_mask(v_t, j_t)
    pads = hm.pads_mask(v_t, j_t)
    kit_colours: dict[str, np.ndarray] = {}
    numbered: dict[int, int] = {}
    decal_of: dict[int, tuple] = {}              # pid -> (Decal, number colour)
    if args.uniforms:
        masks = un.regions(v_t, j_t)
        kit_colours = {team: un.dress(masks, kit) for team, kit in un.KITS.items()}
        kit_colours[""] = un.dress(masks, un.DEFAULT_KIT)
        # Numbers only where identity is sure: a roster name AND a (team,
        # number) no other id of the play claims (the OCR names several ids
        # after one player; those stay blank rather than wrong).
        named = {int(pid): p for pid, p in merged.items()
                 if getattr(p, "player", None) and not str(p.player).startswith("P")
                 and getattr(p, "team", None) in un.KITS
                 and team_of.get(int(pid), getattr(p, "team", None)) == getattr(p, "team", None)}
        claims = collections.Counter((p.team, p.jersey) for p in named.values())
        decal_cache: dict[int, object] = {}
        for pid, p in named.items():
            if not args.numbers:
                break
            if claims[(p.team, p.jersey)] == 1 and p.jersey is not None and 0 <= int(p.jersey) <= 99:
                num = int(p.jersey)
                if num not in decal_cache:
                    decal_cache[num] = un.number_decal(v_t, j_t, faces, masks, num)
                decal_of[pid] = (decal_cache[num], un.KITS[p.team].number)
                numbered[pid] = pid                                   # counted below
        print("uniforms: " + ", ".join(f"{t or 'unknown'} kit" for t in kit_colours)
              + f"; numbers on {len(numbered)} sure ids of {len(named)} named")

    hands_at: dict = {}          # frame -> the holder's ball position (render.carry), filled as bodies are built
    stance_w: dict = {}          # (pid, frame) -> weight of the under-centre stance on the quarterback's held frames
    held_by_pid: dict = {}
    for hp, hf in getattr(tl, "held", set()):
        held_by_pid.setdefault(int(hp), []).append(int(hf))
    for hp, hfs in held_by_pid.items():
        for hf, sw_ in stance_weights(hfs).items():
            stance_w[(hp, hf)] = sw_
    if stance_w:
        print(f"stance: the quarterback under centre on {len(stance_w)} held frames of {len(held_by_pid)} ids")

    export: dict | None = {} if args.export_joints else None
    export_ids: dict = {}

    def body_batch(s, f):
        sw = stance_w.get((int(s.pid), f))
        if sw:                                    # the quarterback held under centre: his stance, not a copied crouch
            s = dataclasses.replace(s, body_pose=stance_body_pose(s.body_pose, sw))
        th = throw_w.get((s.pid, f))
        ca = catch_w.get((s.pid, f))
        w = hold_w.get((s.pid, f), 0.0)
        if th is not None:                        # the passer around the release: the throw over the carry
            s = dataclasses.replace(s, body_pose=throw_body_pose(s.body_pose, th[0], th[1], args.throw_hand))
        elif ca is not None:                      # the receiver around the catch: the reach over the carry
            s = dataclasses.replace(s, body_pose=catch_body_pose(s.body_pose, ca[0], ca[1]))
        elif w > 0:
            s = dataclasses.replace(s, body_pose=carry_body_pose(s.body_pose, w))
        fp = fall_w.get((int(s.pid), f))
        if fp:                                    # the tackle: pitched to the turf, legs curled, the carrier's arms on the ball
            s = dataclasses.replace(s, body_pose=tackled_body_pose(s.body_pose, fp[0]),
                                    global_orient=fall_orient(s.global_orient, fp[0], direction=fp[1]))
        verts, joints = placed_body(s, model)
        if th is not None and f < throw_release and th[0] >= 0.5:
            hands_at[f] = ball_at_hand(joints, args.throw_hand)
        elif ca is not None and f >= catch_frame:
            hands_at[f] = ball_between_hands(joints, tlm.yaw_of(s.global_orient))
        elif th is None and ca is None and w >= 0.5:
            hands_at[f] = ball_between_hands(joints, tlm.yaw_of(s.global_orient))
        # Fitted COLOUR only: the fit's scale and opacity were tuned to blurry
        # 140-px crops and read as translucent bodies at this distance; the
        # colours were the acceptance-tested part.
        # A stitched player wears the texture fitted to any of its member ids.
        owner = next((m for m in tl.members.get(s.pid, [s.pid]) if m in fitted),
                     s.pid if s.pid in fitted else None)
        team = next((team_of[m] for m in tl.members.get(s.pid, [s.pid]) if m in team_of),
                    team_of.get(s.pid, ""))
        if export is not None:
            export.setdefault(int(f), []).append([int(s.pid), team, np.round(np.asarray(joints[:22], float), 3).tolist()])
            export_ids[int(s.pid)] = team
        if args.uniforms:
            colour = kit_colours.get(team, kit_colours[""])
        elif owner is not None:
            colour = fitted[owner]["colour"]
        else:
            colour = team_texture.get(team, TEAM_RGB.get(team, BODY_RGB))
        if args.pads:
            verts = hm.wear_pads(verts, pads)
        if args.helmets:
            shell = hm.HELMET_RGB.get(team, hm.DEFAULT_HELMET_RGB)
            verts, colour = hm.wear_helmet(verts, colour, head, shell)
            verts, colour = hm.wear_facemask(verts, colour, head, face, hm.FACEMASK_RGB.get(team, hm.DEFAULT_FACEMASK_RGB))
        body = tune(mesh_to_gaussians(verts, faces, colour=colour))
        if args.uniforms and s.pid in decal_of:
            decal, num_rgb = decal_of[s.pid]
            dec = un.decal_gaussians(decal, verts, faces, num_rgb)
            if dec is not None:
                return merge([body, dec])
        return body

    ball = {}
    hold_w: dict = {}            # (pid, frame) -> carry-pose weight for the ball's holder
    throw_w: dict = {}           # (passer, frame) -> (phase, weight) of the throw around the release
    catch_w: dict = {}           # (receiver, frame) -> (phase, weight) of the reach and settle around the catch
    fall_w: dict = {}            # (carrier, frame) -> phase of his fall to the turf, ending on the down frame
    throw_release = None
    catch_frame = None
    if args.ball:
        from nfl_gsplat.render.ball import ball_mesh, load_ball
        from nfl_gsplat.render.carry import catch_schedule, holder_weights, load_ball_meta, load_holders, throw_schedule
        from nfl_gsplat.render.tackle import FALL_FRAMES, TACKLER_M, fall_orient, fall_schedule, tackled_body_pose, tacklers

        ball = load_ball(P)
        holders = load_holders(P)
        by_pid: dict = {}
        for hf, pid in holders.items():
            by_pid.setdefault(pid, set()).add(hf)
        drawn: dict = {}
        for tf, sts in tl.states.items():
            for s in sts:
                drawn.setdefault(s.pid, []).append(tf)
        for pid, held in by_pid.items():
            for hf, w in holder_weights(drawn.get(pid, []), held).items():
                hold_w[(pid, hf)] = w
        meta = load_ball_meta(P)
        if meta.get("release") is not None and meta.get("passer") is not None:
            throw_release = int(meta["release"])
            for tf, pw in throw_schedule(throw_release).items():
                throw_w[(int(meta["passer"]), tf)] = pw
        if meta.get("catch") is not None and meta.get("receiver") is not None:
            catch_frame = int(meta["catch"])
            for cf, pw in catch_schedule(catch_frame).items():
                catch_w[(int(meta["receiver"]), cf)] = pw
        if meta.get("down") is not None and meta.get("carrier") is not None:
            # the carrier goes to the ground over the frames ending on the down frame (render.tackle), along
            # the way he ran; the other-team bodies beside him on that frame fall onto him
            down_f, carrier_ = int(meta["down"]), int(meta["carrier"])
            last_drawn = max(drawn.get(carrier_, [down_f]))
            sched = fall_schedule(down_f, last=last_drawn)
            for ff, fp in sched.items():
                fall_w[(carrier_, ff)] = (fp, None)
            who = tacklers(tl.states.get(down_f, []), carrier_, team_of)
            for tp, direction in who.items():
                for ff, fp in sched.items():
                    fall_w[(int(tp), ff)] = (fp, direction)
            print(f"tackle: the carrier {carrier_} falls over {FALL_FRAMES} frames to the down frame {down_f}, lying to "
                  f"{last_drawn}; tacklers {sorted(who)} within {TACKLER_M} m fall onto him")
        print(f"ball: {len(ball)} frames from ball.json; hands on it for {len(by_pid)} holders on {len(hold_w)} body-frames; "
              f"the throw by {meta.get('passer')} ({args.throw_hand}) on {len(throw_w)} frames around {meta.get('release')}"
              if ball else "ball: no ball.json in the play dir")

    def ball_batch(f):
        xyz, v = ball[f]
        if f in hands_at:
            xyz = hands_at[f]                     # in the holder's hands, wherever his arms are
        verts, bfaces, colours = ball_mesh(xyz, v)
        return tune(mesh_to_gaussians(verts, bfaces, colour=colours))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = []
    for i, f in enumerate(frames):
        out = args.out_dir / f"frame_{f:05d}.png"
        if args.resume and out.exists():
            written.append(out)                      # a stalled or interrupted run resumes here
            continue
        states = tl.states.get(f, [])
        if args.no_render:
            for s in states:
                body_batch(s, f)                          # fills the export and the holder's hands
            continue
        scene = merge([field] + [body_batch(s, f) for s in states] + ([ball_batch(f)] if f in ball else []))
        sp = st.SceneParams.from_batch(scene, device=args.device)
        if path is not None and f in path:
            R_v, t_v = look_at(*path[f])
        if view_track is not None:
            K_v, R_v, t_v = broadcast_view(f)
        with torch.no_grad():
            img = st.render(sp, K_v, R_v, t_v, crop=(0, 0, args.width, args.height),
                            background=(0.06, 0.06, 0.08))
        frame = (255 * img.clamp(0, 1).cpu().numpy()).astype(np.uint8)
        if args.post_blur > 0:
            import cv2

            frame = cv2.GaussianBlur(frame, (0, 0), args.post_blur)
        imageio.imwrite(out, frame)
        written.append(out)
        if i % 20 == 0:
            _LOG.info("rendered %d/%d (%d bodies, %.2f s/frame)", i + 1, len(frames), len(states),
                      (time.time() - t0) / (i + 1))
    if export is not None:
        import json

        from nfl_gsplat.render.foot_lock import SMPLX_BODY_PARENTS

        los = None
        try:
            from nfl_gsplat.render.play_timeline import _los
            los = _los(P)
        except Exception:  # noqa: BLE001
            los = None
        ball_out = {}
        for bf in sorted(ball):
            xyz = hands_at.get(bf, ball[bf][0])
            ball_out[int(bf)] = np.round(np.asarray(xyz, float), 3).tolist()
        cams_out: dict = {}
        try:
            from nfl_gsplat.render.play_timeline import clip_offset as _clip_offset
            off = {"sideline": 0, "endzone": int(_clip_offset(P))}
            for cam_name, trk in tracks.items():
                cams_out[cam_name] = {}
                for f in frames:
                    fc = int(f) + off.get(cam_name, 0)
                    if fc < 0 or fc >= len(trk.conf) or trk.conf[fc] <= 0:
                        continue
                    intr_, pose_ = trk.at(fc)
                    K_ = np.asarray(intr_.K(), float)
                    cams_out[cam_name][str(int(f))] = {"K": [float(K_[0, 0]), float(K_[1, 1]), float(K_[0, 2]), float(K_[1, 2])],
                                                       "R": np.round(np.asarray(pose_.R, float), 6).ravel().tolist(),
                                                       "t": np.round(np.asarray(pose_.t, float), 4).ravel().tolist(),
                                                       "wh": [int(getattr(trk, "width", 1920)), int(getattr(trk, "height", 1080))]}
        except Exception as exc:  # noqa: BLE001
            print(f"cameras not exported: {exc}")
        doc = {"play": str(P.name), "fps": float(args.fps) / max(1, args.stride), "stride": int(args.stride),
               "frames": [int(f) for f in frames if int(f) in export], "cameras": cams_out,
               "parents": [int(v) for v in SMPLX_BODY_PARENTS[:22]],
               "teams": {str(k): v for k, v in export_ids.items()},
               "names": {str(k): (f"{int(getattr(merged[k], 'jersey', 0))} {merged[k].player}".strip()
                                   if int(getattr(merged[k], 'jersey', 0) or 0) > 0 else str(merged[k].player))
                         for k in export_ids if k in merged and not str(getattr(merged[k], 'player', '')).startswith('P')},
               "los": los, "bodies": {str(k): v for k, v in export.items()}, "ball": {str(k): v for k, v in ball_out.items()}}
        args.export_joints.parent.mkdir(parents=True, exist_ok=True)
        args.export_joints.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        print(f"exported joints: {sum(len(v) for v in export.values())} body-frames on {len(export)} frames, "
              f"{len(ball_out)} ball frames -> {args.export_joints}")
    if args.no_render:
        return
    mp4 = args.out_dir / "play.mp4"
    with imageio.get_writer(mp4, fps=args.fps / max(1, args.stride), codec="libx264",
                            quality=8, macro_block_size=None) as w:
        for p in written:
            w.append_data(imageio.imread(p))
    print(f"wrote {len(written)} frames and {mp4} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
