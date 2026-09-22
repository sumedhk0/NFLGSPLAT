"""Per-camera track audit of a few ids over a frame window: which camera track each id rides, where its box is
(centre x, bottom, height, width) and where that box lands on the ground -- one row per sampled frame.

WHY. On 2026-09-21/22 every identity error at the line was found with this table typed by hand, never with a ruler on
the timeline: two men under four crossed ids (37/211/205/157), the endzone-only ids that were one hidden Raven (198,
136, 112 on id 4), the "phantom" whose sideline rows were fragment boxes of another man, and the quarterback's sideline
id sliding onto the centre after the throw while the quarterback ran on under another id (80/204/157 at 557). A swap
shows as one id's box centre x jumping onto another id's x; a fragment shows as a height a third of the track's; a
twin shows as two ids on one (x, bottom). The fix is always a per-camera-track fold (08z --cam --track-id --frames).

USAGE (nflgsplat env; reads tracks.parquet, cameras.npz, clip_offset.json):
  python scripts/09c_track_audit.py --play-dir P --ids 80 204 157 --frames 548 607 --step 2
  --cam sideline|endzone|both (default both)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.render.play_timeline import clip_offset, ground_positions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--cam", default="both", choices=["sideline", "endzone", "both"])
    a = ap.parse_args()
    P = Path(a.play_dir)
    df = pd.read_parquet(P / "tracks.parquet")
    tracks = load_camera_track(str(P / "cameras.npz"))
    off = clip_offset(P)
    lo, hi = a.frames
    sel = df[df["frame"].between(lo, hi) & df["global_player_id"].isin(a.ids)]
    cams = ["sideline", "endzone"] if a.cam == "both" else [a.cam]
    for cam in cams:
        sub = sel[sel["cam"] == cam]
        if sub.empty:
            print(f"{cam}: no rows for these ids on {lo}-{hi}")
            continue
        g = ground_positions(sub, tracks, frame_shift={"endzone": off} if cam == "endzone" else None)
        rows = {(int(r.frame), int(r.global_player_id)): r for r in sub.itertuples()}
        print(f"{cam}: frame | " + " | ".join(f"{pid}: track (cx, bottom) h w -> (x, y)".ljust(40) for pid in a.ids))
        for f in range(lo, hi + 1, a.step):
            cells = []
            for pid in a.ids:
                r = rows.get((f, pid))
                xy = g.get(f, {}).get(pid)
                if r is None:
                    cells.append("-".ljust(40))
                    continue
                cx, bot = 0.5 * (r.bbox_x1 + r.bbox_x2), r.bbox_y2
                h, w = r.bbox_y2 - r.bbox_y1, r.bbox_x2 - r.bbox_x1
                gs = f"({xy[0]:.1f},{xy[1]:.1f})" if xy is not None else "(off)"
                cells.append(f"t{int(r.track_id)} ({cx:.0f},{bot:.0f}) h{h:.0f} w{w:.0f} -> {gs}".ljust(40))
            print(f"  {f:4d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
