#!/usr/bin/env python
"""Recover each helmet-dataset play's video-to-tracking offset, and record it.

The clips are windows inside a longer tracking record, cut at no fixed offset
from the snap, so every play needs its own. Without it a labelled helmet box
cannot be paired with the tracked position of the player it belongs to, and
nothing downstream can be scored against truth.

Plays are aligned by plane fit (see nfl_gsplat.data.align_video) and ACCEPTED
only when the two synchronised views independently agree. That is an external
check, not a tuned threshold -- the cameras really are synchronised, so a
disagreement is a failed recovery and belongs in the rejected list rather than
in a manifest something later trusts.

Writes data/helmet/alignment.json: one entry per play with the offset, both
views' residual and contrast, and for rejected plays the reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_gsplat.data.align_video import align_play_plane
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/alignment.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = args.out or (args.root / "alignment.json")
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")

    keys = sorted(plays)
    if args.limit:
        keys = keys[:args.limit]

    records, n_ok = {}, 0
    for i, key in enumerate(keys, 1):
        game, play = key
        got = align_play_plane(labels, plays[key], game, play)
        views = {v: {"offset_s": round(o, 3) if o is not None else None,
                     "residual_px": round(r, 2),
                     "contrast": round(c, 1)}
                 for v, (o, r, c) in got.per_view.items()}
        records[f"{game}_{play}"] = {
            "game_key": game, "play_id": play,
            "offset_s": round(got.offset_s, 3) if got.ok else None,
            "reason": got.reason, "views": views}
        n_ok += got.ok
        flag = "ok  " if got.ok else "SKIP"
        detail = "  ".join(
            f"{v[:4]}={d['offset_s']:+.2f}({d['residual_px']:.0f}px,"
            f"{d['contrast']:.0f}x)"
            for v, d in views.items() if d["offset_s"] is not None)
        print(f"[{i:3d}/{len(keys)}] {flag} {game}/{play:<6d} {detail}"
              + ("" if got.ok else f"   <- {got.reason}"), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n{n_ok}/{len(keys)} plays aligned  ->  {out_path}")
    if n_ok < len(keys):
        print("rejected:")
        for name, r in records.items():
            if r["offset_s"] is None:
                print(f"   {name}: {r['reason']}")


if __name__ == "__main__":
    main()
