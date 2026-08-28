"""Compare and fuse the two cameras' independent estimates of the same players.

Needs both feeds posed on the SAME frames (``05c --match-frames``) and an
identity resolution, which is what supplies the correspondence: geometry cannot
say which endzone track is which sideline track, but a shared jersey number can.

The comparison is the point. Every other check in this pipeline validates
geometry against paint or identity against geometry; nothing has ever checked a
POSE against anything outside the network that produced it. Two cameras 131 m
apart, sharing no pixels, no tracker and no calibration, give exactly that.
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


def _load(path: Path):
    blob = pickle.load(open(path, "rb"))
    return blob["cam"], blob["frames"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--poses", required=True, nargs=2, type=Path,
                    metavar=("CACHE_A", "CACHE_B"))
    ap.add_argument("--identity", required=True, type=Path)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import smplx
    import torch

    from nfl_gsplat.pose.fuse_views import fuse_skeletons, summarise
    from nfl_gsplat.pose.place_on_field import place_skeleton

    cam_a, cache_a = _load(args.poses[0])
    cam_b, cache_b = _load(args.poses[1])
    if cam_a == cam_b:
        raise SetupError(f"both caches are for {cam_a!r}; two cameras are needed")
    shared_frames = sorted(set(cache_a) & set(cache_b))
    if not shared_frames:
        raise SetupError(
            f"{cam_a} and {cam_b} share no posed frames. Pose the second camera "
            "with --match-frames pointing at the first's cache.")
    _LOG.info("%s and %s share %d posed frames", cam_a, cam_b, len(shared_frames))

    ident = pickle.load(open(args.identity, "rb"))
    merged = ident["merged"]
    stitch = ident.get("stitch") or {}

    def fragments_of(cam, player_id):
        """Every raw tracker id that stitched into this player."""
        m = stitch.get(cam, {})
        out = {f for f, pid in m.items() if pid == player_id}
        out.add(player_id)
        return out

    pairs = {}
    for jersey, player in merged.items():
        if cam_a in player.tracks and cam_b in player.tracks:
            pairs[jersey] = (fragments_of(cam_a, player.tracks[cam_a]),
                             fragments_of(cam_b, player.tracks[cam_b]))
    if not pairs:
        raise SetupError(
            "no player was named by BOTH cameras, so there is no correspondence "
            "to fuse. Identity is what supplies it; geometry cannot.")
    _LOG.info("%d players named by both cameras: %s", len(pairs),
              sorted(pairs))

    cams = load_camera_track(args.play_dir / "cameras.npz")
    track_a, track_b = cams[cam_a], cams[cam_b]
    centre_a = -track_a.at(shared_frames[0])[1].R.T @ track_a.at(shared_frames[0])[1].t
    centre_b = -track_b.at(shared_frames[0])[1].R.T @ track_b.at(shared_frames[0])[1].t

    model = smplx.create(str(args.body_models), model_type="smplx",
                         gender="neutral", use_pca=False, batch_size=1)

    def placed(cache, track, frame, frag_ids):
        rec = None
        for fid in frag_ids:
            if fid in cache[frame]:
                rec = cache[frame][fid]
                break
        if rec is None:
            return None
        with torch.no_grad():
            res = model(
                betas=torch.tensor(np.asarray(rec["betas"])[None, :model.num_betas],
                                   dtype=torch.float32),
                body_pose=torch.tensor(np.asarray(rec["body_pose"]).reshape(1, -1),
                                       dtype=torch.float32),
                global_orient=torch.tensor(
                    np.asarray(rec["global_orient"]).reshape(1, 3),
                    dtype=torch.float32))
        joints = res.joints[0].cpu().numpy().astype(np.float64)
        intr, pose = track.at(frame)
        x1, _y1, x2, y2 = rec["bbox"]
        return place_skeleton(joints, (0.5 * (x1 + x2), float(y2)),
                              intr.K(), pose.R, pose.t)

    results, fused_out = {}, {}
    for jersey, (frags_a, frags_b) in sorted(pairs.items()):
        stats, n = [], 0
        for frame in shared_frames:
            ja = placed(cache_a, track_a, frame, frags_a)
            jb = placed(cache_b, track_b, frame, frags_b)
            if ja is None or jb is None:
                continue
            stats.append(summarise(ja, jb))
            fused_out.setdefault(jersey, {})[frame] = fuse_skeletons(
                ja, jb, centre_a, centre_b)
            n += 1
        if not stats:
            _LOG.info("#%d: no frame has this player in both caches", jersey)
            continue
        results[jersey] = {
            "frames": n,
            "root_m": float(np.median([s["root_m"] for s in stats])),
            "shape_m": float(np.median([s["shape_median_m"] for s in stats])),
            "median_m": float(np.median([s["median_m"] for s in stats])),
        }

    print(f"\n{'#':>4} {'player':22s} {'frames':>6} {'whole-body':>11} "
          f"{'pose shape':>11}  reading")
    for jersey, r in sorted(results.items(), key=lambda kv: kv[1]["shape_m"]):
        p = merged[jersey]
        reading = ("poses agree, placement differs"
                   if r["shape_m"] < 0.25 and r["root_m"] > 1.0
                   else "both agree" if r["shape_m"] < 0.25
                   else "poses differ")
        print(f"{('#%d' % jersey):>4} {p.player:22s} {r['frames']:6d} "
              f"{r['root_m']:9.2f} m {r['shape_m']:9.2f} m  {reading}")
    if results:
        print(f"\nmedian across players: whole-body "
              f"{np.median([r['root_m'] for r in results.values()]):.2f} m, "
              f"pose shape "
              f"{np.median([r['shape_m'] for r in results.values()]):.2f} m")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump({"stats": results, "fused": fused_out,
                         "cameras": (cam_a, cam_b)}, fh)
        _LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
