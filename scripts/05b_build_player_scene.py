"""Build one frame of the play as gaussians: procedural field + real players.

This is the first stage that puts a player's ROSTER BODY on the field rather
than a generic one. The decomposition is deliberate and is the whole point:

* ARTICULATION comes from the network. SMPLest-X sees a crop and returns a pose.
* SHAPE comes from the ROSTER, for every identity that resolved. A 315 lb tackle
  and a 193 lb safety look nearly the same to a pose network fitting a
  silhouette a hundred pixels tall, but the roster knows both their height and
  their weight, and fit_height_and_weight solves betas reproducing both exactly.
* GLOBAL PLACEMENT comes from the CALIBRATION, never from the network's own
  translation, which is a monocular guess at depth. The foot point is
  intersected with the z=0 plane.

Players whose identity did not resolve keep the network's own shape and are
counted separately, so the picture never implies more knowledge than we have.

Runs in the SMPL-X env (.venv-smplx): torch+cuda and the smplx package live
there, and the main env is Python 3.14 with neither.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# These meshes are untextured. One neutral tone per team reads as "a body" and
# says which side it is on, without implying we have recovered appearance.
BODY_RGB = (0.72, 0.62, 0.55)
TEAM_RGB = {"ARI": (0.62, 0.11, 0.22), "SEA": (0.11, 0.22, 0.34)}

CROP_W, CROP_H = 192, 256
MIN_BOX_W, MIN_BOX_H = 16, 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--identity", required=True, type=Path,
                    help="pickle with {'merged': {jersey: PlayerIdentity}}")
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--field-res-m", type=float, default=0.15)
    args = ap.parse_args()

    import cv2
    import pandas as pd
    import smplx
    import torch

    from nfl_gsplat.compositing.merge_ply import (batch_from_arrays,
                                                  save_gaussian_ply)
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.field.procedural_field import (render_field_texture,
                                                   texture_to_gaussians)
    from nfl_gsplat.pose.fit_shape import fit_height_and_weight
    from nfl_gsplat.pose.place_on_field import place_mesh, stature
    from nfl_gsplat.pose.smplestx_infer import SMPLestXConfig, infer_crops
    from nfl_gsplat.utils.video import iter_frames

    play = args.play_dir
    track = load_camera_track(play / "cameras.npz")[args.cam]
    if float(np.asarray(track.conf)[args.frame]) <= 0:
        raise SetupError(
            f"frame {args.frame} has conf=0 for {args.cam!r}: that pose was "
            "filled in from a neighbour, not solved, so every placement would "
            "inherit a guess. Pick a verified frame.")
    intr, pose = track.at(args.frame)
    k_mat, rot, tvec = intr.K(), pose.R, pose.t

    df = pd.read_parquet(play / "tracks.parquet")
    rows = df[(df["cam"] == args.cam) & (df["frame"] == args.frame)
              & (df["track_id"] >= 0)]
    if rows.empty:
        raise SetupError(f"no {args.cam} detections on frame {args.frame}")

    ident_blob = pickle.load(open(args.identity, "rb"))
    merged = ident_blob["merged"]
    # Identities are keyed by STITCHED player id; this cache is keyed by the
    # tracker's raw fragment ids. Map every fragment of a player onto that
    # player, or nothing here matches and every body is drawn generic.
    stitched = (ident_blob.get("stitch") or {}).get(args.cam, {})
    by_player = {p.tracks[args.cam]: p for p in merged.values()
                 if args.cam in p.tracks}
    player_of = {frag: by_player[pid] for frag, pid in stitched.items()
                 if pid in by_player}
    for pid, p in by_player.items():          # unstitched ids map to themselves
        player_of.setdefault(pid, p)
    _LOG.info("%d identified players have a %s track", len(player_of), args.cam)

    image = None
    for idx, rgb in iter_frames(play / f"{args.cam}.mp4", stride=1):
        if idx == args.frame:
            image = rgb
            break
    if image is None:
        raise SetupError(f"could not decode frame {args.frame}")
    h_img, w_img = image.shape[:2]

    crops, boxes, track_ids = [], [], []
    for r in rows.itertuples():
        x1, y1 = max(0, int(r.bbox_x1)), max(0, int(r.bbox_y1))
        x2, y2 = min(w_img, int(r.bbox_x2)), min(h_img, int(r.bbox_y2))
        if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
            continue
        crops.append(cv2.resize(image[y1:y2, x1:x2], (CROP_W, CROP_H)))
        boxes.append([x1, y1, x2, y2])
        track_ids.append(int(r.track_id))
    if not crops:
        raise SetupError(f"every box on frame {args.frame} was below the "
                         f"{MIN_BOX_W}x{MIN_BOX_H} px floor")
    _LOG.info("frame %d: %d player crops", args.frame, len(crops))

    out = infer_crops(np.asarray(crops, np.uint8), np.asarray(boxes, float),
                      SMPLestXConfig())

    model = smplx.create(str(args.body_models), model_type="smplx",
                         gender="neutral", use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)

    batches, report = [], []
    n_roster = n_network = 0
    for i, tid in enumerate(track_ids):
        player = player_of.get(tid)
        betas = np.asarray(out["betas"][i], float)
        source, fit_err = "network", ""
        if player is not None and player.weight_lb > 0:
            # The roster shape REPLACES the network's. Measured on this roster
            # the fit is exact (0.000 m / 0 lb), which a monocular network
            # cannot be.
            betas, got_h, got_lb = fit_height_and_weight(
                model, faces, player.height_m, player.weight_lb, n_betas=4)
            # Report what the fit ACHIEVED, not what was asked for. A silent
            # miss here would render the wrong body while the roster column
            # still read correctly.
            fit_err = f"{got_h - player.height_m:+.3f}m {got_lb - player.weight_lb:+.0f}lb"
            source = "roster"
            n_roster += 1
        else:
            n_network += 1

        with torch.no_grad():
            res = model(
                betas=torch.tensor(betas[None, :model.num_betas],
                                   dtype=torch.float32),
                body_pose=torch.tensor(out["body_pose"][i].reshape(1, -1),
                                       dtype=torch.float32),
                global_orient=torch.tensor(out["global_orient"][i].reshape(1, 3),
                                           dtype=torch.float32))
        verts = res.vertices[0].cpu().numpy().astype(np.float64)
        # Joints from the SAME forward pass, so mesh and skeleton share a frame.
        joints = res.joints[0].cpu().numpy().astype(np.float64)

        x1, y1, x2, y2 = boxes[i]
        foot_uv = (0.5 * (x1 + x2), float(y2))
        placed = place_mesh(verts, joints, foot_uv, k_mat, rot, tvec)
        colour = (TEAM_RGB.get(player.team, BODY_RGB) if player is not None
                  else BODY_RGB)
        batches.append(mesh_to_gaussians(placed, faces, colour=colour))
        report.append((tid, player, float(np.mean(placed[:, 0])),
                       float(np.mean(placed[:, 1])), stature(placed), source,
                       fit_err))

    _LOG.info("%d players built from roster height+weight, %d from the "
              "network's own shape", n_roster, n_network)

    # texture_to_gaussians returns the raw PLY arrays; the field is degree-0 by
    # construction (turf and paint are Lambertian), which matches the meshes.
    f_xyz, f_rot, f_scale, f_opac, f_dc = texture_to_gaussians(
        render_field_texture(res_m=args.field_res_m), args.field_res_m)
    field = batch_from_arrays(f_xyz, f_rot, f_scale, f_opac,
                              np.asarray(f_dc, np.float32)[:, :, None])
    scene = merge([field] + batches)
    scene.assert_no_nans()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gaussian_ply(args.out, scene)
    _LOG.info("wrote %s: %d gaussians (%d field + %d body)", args.out,
              scene.xyz.shape[0], field.xyz.shape[0],
              scene.xyz.shape[0] - field.xyz.shape[0])

    print(f"\n{'trk':>5} {'#':>4} {'player':22s} {'lb':>4} "
          f"{'x_m':>7} {'y_m':>7} {'extent_m':>8}  shape     fit error")
    for tid, player, x, y, tall, source, fit_err in sorted(report,
                                                           key=lambda r: -r[4]):
        jersey = f"#{player.jersey}" if player else "-"
        name = player.player if player else "(unidentified)"
        lb = f"{player.weight_lb:.0f}" if player and player.weight_lb else "-"
        print(f"{tid:5d} {jersey:>4} {name:22s} {lb:>4} "
              f"{x:7.1f} {y:7.1f} {tall:8.2f}  {source:8s}  {fit_err}")


if __name__ == "__main__":
    main()
