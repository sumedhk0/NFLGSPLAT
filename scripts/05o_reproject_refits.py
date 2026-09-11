"""05o: which joints are right? The cameras are the ruler.

WHY. Two sources of 3-D joints now exist: the monocular regressor per view
fused across views (05e), and detector keypoints triangulated with the
calibrated cameras (05n). Both feed the same SMPL-X refit (05f). Neither
has ground truth, but both refits can be projected back into both
cameras and compared with the detector keypoints -- independent 2-D
evidence for either. Lower pixel error in BOTH views is the win; the
recovered stature against the roster (known to the inch, constrains
nothing upstream) is the second ruler.

WHAT. For each refit blob: forward kinematics of (betas -> rest skeleton,
body_pose, global_orient, transl) to the 22 body joints, projected into
each camera at that frame, error against the keypoints mapped to SMPL-X
order (05n's mapping). Prints median px per view and per joint group, and
ankle-to-head stature vs roster. Runs in the SMPL-X env.

USAGE:
  python scripts/05o_reproject_refits.py --play-dir P --refits P/poses_refit.json P/poses_refit_tri.json
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.pose.forward_kinematics import fk_forward, load_smplx_skeleton  # noqa: E402
from nfl_gsplat.pose.fuse_smplx import _pack_params  # noqa: E402

GROUPS = {"hips/pelvis": [0, 1, 2], "knees": [4, 5], "ankles": [7, 8], "neck/head": [12, 15],
          "shoulders": [16, 17], "elbows": [18, 19], "wrists": [20, 21]}
ANKLES, HEAD = (7, 8), 15


def _load_05n():
    here = Path(__file__).with_name("05n_triangulate_keypoints.py")
    spec = importlib.util.spec_from_file_location("k05n", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refits", required=True, nargs="+", type=Path)
    ap.add_argument("--keypoints", type=Path, default=None)
    ap.add_argument("--identity", type=Path, default=None)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--offset", type=int, default=None,
                    help="endzone frame offset (default: from poses_tri.json)")
    args = ap.parse_args()

    P = args.play_dir
    k05n = _load_05n()
    tracks = load_camera_track(P / "cameras.npz")
    kdf = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    views = k05n.per_player_views(kdf, min_conf=args.min_conf)
    offset = args.offset
    if offset is None and (P / "poses_tri.json").exists():
        offset = int(pickle.load(open(P / "poses_tri.json", "rb")).get("offset", 0))
    offset = offset or 0
    ident = args.identity or (P / "identity_resolved.pkl")
    roster_h = {}
    if ident.exists():
        merged = pickle.load(open(ident, "rb")).get("merged", {})
        roster_h = {int(pid): float(p.height_m) for pid, p in merged.items()
                    if getattr(p, "height_m", None) and getattr(p, "player", "")
                    and not str(p.player).startswith("P")}
    print(f"keypoints for {len(views)} ids, min conf {args.min_conf}, endzone offset {offset:+d}")

    for path in args.refits:
        blob = pickle.load(open(path, "rb"))
        frames = blob["frames"]
        rest_cache: dict[int, np.ndarray] = {}
        forward_cache: dict = {}
        err = {cam: {g: [] for g in GROUPS} for cam in ("sideline", "endzone")}
        stature = []
        n_states = 0
        for f, recs in frames.items():
            for pid, r in recs.items():
                pid = int(pid)
                if pid not in rest_cache:
                    rest, _parents = load_smplx_skeleton(args.body_models,
                                                         betas=np.asarray(r["betas"], float))
                    rest_cache[pid] = rest
                    forward_cache[pid] = fk_forward(rest)
                p = _pack_params(np.asarray(r["body_pose"], float), np.asarray(r["global_orient"], float),
                                 np.asarray(r["transl"], float))
                joints = forward_cache[pid](p)[:22]
                n_states += 1
                stature.append((pid, float(joints[list(ANKLES), 2].min())))      # lowest ankle above turf
                for cam in ("sideline", "endzone"):
                    obs = views.get(pid, {}).get(cam, {}).get(int(f) + (offset if cam == "endzone" else 0))
                    if obs is None:
                        continue
                    uv, conf = obs
                    intr, pose = tracks[cam].at(int(f) + (offset if cam == "endzone" else 0))
                    K = intr.K()
                    cam_pts = joints @ pose.R.T + pose.t
                    z = cam_pts[:, 2]
                    proj = np.column_stack([K[0, 0] * cam_pts[:, 0] / z + K[0, 2],
                                            K[1, 1] * cam_pts[:, 1] / z + K[1, 2]])
                    e = np.linalg.norm(proj - uv, axis=1)
                    for g, idx in GROUPS.items():
                        ok = [j for j in idx if np.isfinite(e[j]) and z[j] > 0]
                        err[cam][g].extend(e[ok].tolist())
        print(f"\n{path.name}: {n_states} body states over {len(frames)} frames")
        for cam in ("sideline", "endzone"):
            allv = [v for g in GROUPS for v in err[cam][g]]
            line = "  " + ", ".join(f"{g} {np.median(err[cam][g]):.1f}" for g in GROUPS if err[cam][g])
            print(f"  {cam}: median {np.median(allv):.1f} px over {len(allv)} joints;" + line)
        # The ground is a ruler the keypoints never touched: a standing or
        # running player's lower ankle sits ~0.08 m above the turf.
        ank = np.array([h for _pid, h in stature])
        if len(ank):
            print(f"  lower ankle above turf: p10 {np.percentile(ank, 10):+.2f}, median {np.median(ank):+.2f} m, IQR {np.percentile(ank, 25):+.2f}"
                  f"..{np.percentile(ank, 75):+.2f}, |z| > 0.3 m in {100 * np.mean(np.abs(ank) > 0.3):.0f}% of states")


if __name__ == "__main__":
    main()
