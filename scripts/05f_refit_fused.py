#!/usr/bin/env python
"""Refit SMPL-X parameters to the fused two-view joints, for the renderer.

scripts/05e fuses each player's sideline and endzone skeletons into world
joints -- the pose the two cameras agree on, placed where both put it -- but
writes joints, and scripts/05d draws bodies from SMPL-X parameters. This
stage fits those parameters to the fused joints (pose.refit_fused, on the
same forward kinematics the animator uses) and writes 05d's pose cache in
WORLD mode: every rec carries `transl` in metres on the field, so 05d places
the body there instead of grounding a foot pixel through one camera.

Shape (betas) is the sideline network's own estimate averaged over the
track, as 05d would have used; 05d still replaces it with the roster body
when the player is identified. The first frame's body pose starts from the
sideline fit for that frame when there is one.

Runs in the smplx env (needs the SMPL-X body model for the rest skeleton).

    python scripts/05f_refit_fused.py --play-dir data/all22/.../play_001 \\
        --fused .../poses_fused.json --poses .../poses_sideline.json \\
        --identity .../identity_resolved.pkl --out .../poses_refit.json
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from nfl_gsplat.errors import PoseFusionError, SetupError
from nfl_gsplat.pose.forward_kinematics import load_smplx_skeleton
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig
from nfl_gsplat.pose.refit_fused import refit_player, render_blob
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
MIN_FRAMES = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--fused", required=True, type=Path, help="scripts/05e --out")
    ap.add_argument("--poses", required=True, type=Path,
                    help="the sideline pose cache (betas, first-frame body pose)")
    ap.add_argument("--identity", required=True, type=Path)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--min-valid-joints", type=int, default=10,
                    help="joints a frame needs to be fitted (triangulated joints are sparser: 6)")
    ap.add_argument("--min-frame-frac", type=float, default=0.7,
                    help="fraction of a player's frames that must fit, else the player is skipped")
    args = ap.parse_args()

    fused = pickle.load(open(args.fused, "rb"))["fused"]      # pid -> {frame: [J, 3]}
    side = pickle.load(open(args.poses, "rb"))
    side_cam, side_cache, stride = side["cam"], side["frames"], int(side["stride"])
    merged = pickle.load(open(args.identity, "rb"))["merged"]
    if not fused:
        raise SetupError(f"{args.fused} holds no fused players")

    cfg = SMPLXFitConfig(max_iter=args.max_iter, min_valid_joints=args.min_valid_joints,
                         min_frame_validity_frac=args.min_frame_frac)
    fits, betas_of, rms_all, skipped = {}, {}, [], []
    for pid, by_frame in sorted(fused.items()):
        fs = sorted(by_frame)
        if len(fs) < MIN_FRAMES:
            skipped.append((pid, f"{len(fs)} frames"))
            continue
        tid = merged[pid].tracks.get(side_cam) if pid in merged else None
        recs = [side_cache[f][tid] for f in fs
                if tid is not None and f in side_cache and tid in side_cache[f]]
        betas = (np.mean([r["betas"] for r in recs], axis=0) if recs
                 else np.zeros(10, np.float32))
        rest, _parents = load_smplx_skeleton(args.body_models, betas=betas)
        init_bp = recs[0]["body_pose"] if recs else None
        target = np.stack([np.asarray(by_frame[f], float) for f in fs])
        try:
            body_pose, orient, transl, valid, rms = refit_player(
                target, rest, cfg=cfg, init_body_pose=init_bp)
        except PoseFusionError as exc:
            skipped.append((pid, str(exc)))
            continue
        fits[pid] = (fs, body_pose, orient, transl, valid)
        betas_of[pid] = betas
        rms_all.extend(rms[valid].tolist())
        _LOG.info("player %d: %d/%d frames fit, rms %.3f m", pid, int(valid.sum()),
                  len(fs), float(np.nanmedian(rms)))
    if not fits:
        raise PoseFusionError("no player refit; " + "; ".join(f"{p}: {w}" for p, w in skipped))

    blob = render_blob(fits, betas_of, stride=stride, appearance_cam=side_cam)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"refit {len(fits)} players over {len(blob['frames'])} frames; "
          f"median joint rms {np.median(rms_all):.3f} m; skipped {len(skipped)} "
          f"({', '.join(f'{p}: {w}' for p, w in skipped[:6])}{'...' if len(skipped) > 6 else ''})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
