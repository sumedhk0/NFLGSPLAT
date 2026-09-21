"""Remove one id's boxes of one camera over a frame range from tracks.parquet (keypoints follow).

WHY. The tracker re-associates a detection to a track it lost up to its buffer length ago; when that detection is
another man, the id gets one stray box far from where he really is. Play 1 (2026-09-21): id 4's sideline track
ends at 443 behind KC 65 (the film: hidden, then on the turf under him) and has a single box again at 465 on
the man beside him. That box is the edge of his sideline span, so the beyond-span rule joined his own endzone
track (444-478, folded in from the endzone-only id 198) to it, measured the join 2.73 m off and dropped all 20
frames; the hole rule then walked him toward the stray box at 0.26-0.35 m/frame. The fold tools cannot remove
a row; this does, with the same backups and keypoint bookkeeping as 08z.

USAGE (smplx312, the venv that writes the caches):
  python scripts/08za_drop_rows.py --play-dir P --id 4 --cam sideline --frames 465 465 [--apply]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.tracking.fold import drop_rows, keypoint_map  # noqa: E402
from nfl_gsplat.tracking.relabel import backup_path, relabel_keypoints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--cam", required=True, choices=["sideline", "endzone"])
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    P = Path(args.play_dir)
    df = pd.read_parquet(P / "tracks.parquet")
    out, n = drop_rows(df, args.id, cam=args.cam, frames=tuple(args.frames))
    sub = df[(df["global_player_id"] == args.id) & (df["cam"] == args.cam)]
    print(f"id {args.id} {args.cam}: {len(sub)} boxes on {int(sub['frame'].min()) if len(sub) else '-'}-"
          f"{int(sub['frame'].max()) if len(sub) else '-'}; {n} on {args.frames[0]}-{args.frames[1]} to drop")
    if n == 0:
        raise SystemExit("nothing to drop")
    if not args.apply:
        print("dry run. Re-run with --apply.")
        return
    tb = backup_path(P / "tracks.parquet", ".pre_drop")
    shutil.copy2(P / "tracks.parquet", tb)
    out.to_parquet(P / "tracks.parquet", index=False)
    print(f"wrote tracks.parquet (backup {tb.name})")
    kp = P / "keypoints_2d.parquet"
    if kp.exists():
        kdf = pd.read_parquet(kp)
        kout, kdrop = relabel_keypoints(kdf, keypoint_map(df, out))
        kb = backup_path(kp, ".pre_drop")
        shutil.copy2(kp, kb)
        kout.to_parquet(kp, index=False)
        print(f"keypoints: {len(kout)} rows written, {kdrop} dropped (backup {kb.name})")


if __name__ == "__main__":
    main()
