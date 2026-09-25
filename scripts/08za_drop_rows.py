"""Remove one id's boxes of one camera over a frame range from tracks.parquet (keypoints and pose caches follow).

WHY. The tracker re-associates a detection to a track it lost up to its buffer length ago; when that detection is
another man, the id gets one stray box far from where he really is. Play 1 (2026-09-21): id 4's sideline track
ends at 443 behind KC 65 (the film: hidden, then on the turf under him) and has a single box again at 465 on
the man beside him. That box is the edge of his sideline span, so the beyond-span rule joined his own endzone
track (444-478, folded in from the endzone-only id 198) to it, measured the join 2.73 m off and dropped all 20
frames; the hole rule then walked him toward the stray box at 0.26-0.35 m/frame. The fold tools cannot remove
a row; this does, with the same backups and keypoint bookkeeping as 08z. ``--track-id`` drops only that camera
track's rows (play 1 2026-09-25: the left tackle's endzone id alternated between his own box, track 25, and track
37, which had drifted onto the Raven he blocked -- a twin of the Raven's own box).

Every keypoint table (keypoints_2d.parquet, keypoints_2d_ft2.parquet) follows the rows, and the pose caches are
carried by 08v as 08z does (until 2026-09-25 this tool did neither: a dropped row's ft2 keypoints and pose records
stayed under the id).

USAGE (smplx312, the venv that writes the caches):
  python scripts/08za_drop_rows.py --play-dir P --id 4 --cam sideline --frames 465 465 [--track-id T] [--apply]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.tracking.fold import drop_rows, keypoint_map  # noqa: E402
from nfl_gsplat.tracking.relabel import backup_path, relabel_keypoint_tables  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--cam", required=True, choices=["sideline", "endzone"])
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--track-id", type=int, default=None, help="only this camera track's rows")
    ap.add_argument("--no-remap", action="store_true", help="skip 08v (the pose caches keep the dropped rows' records)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    P = Path(args.play_dir)
    df = pd.read_parquet(P / "tracks.parquet")
    out, n = drop_rows(df, args.id, cam=args.cam, frames=tuple(args.frames), track_id=args.track_id)
    sub = df[(df["global_player_id"] == args.id) & (df["cam"] == args.cam)]
    trk = "" if args.track_id is None else f" (track {args.track_id})"
    print(f"id {args.id} {args.cam}: {len(sub)} boxes on {int(sub['frame'].min()) if len(sub) else '-'}-"
          f"{int(sub['frame'].max()) if len(sub) else '-'}; {n} on {args.frames[0]}-{args.frames[1]}{trk} to drop")
    if n == 0:
        raise SystemExit("nothing to drop")
    if not args.apply:
        print("dry run. Re-run with --apply.")
        return
    tb = backup_path(P / "tracks.parquet", ".pre_drop")
    shutil.copy2(P / "tracks.parquet", tb)
    out.to_parquet(P / "tracks.parquet", index=False)
    print(f"wrote tracks.parquet (backup {tb.name})")
    for name, n_rows, kdrop, _changed, kb in relabel_keypoint_tables(P, keypoint_map(df, out), ".pre_drop"):
        print(f"{name}: {n_rows} rows written, {kdrop} dropped (backup {kb})")
    if not args.no_remap:
        subprocess.run([sys.executable, str(Path(__file__).with_name("08v_remap_poses_after_relabel.py")),
                        "--play-dir", str(P), "--before", tb.name, "--apply"], check=True)


if __name__ == "__main__":
    main()
