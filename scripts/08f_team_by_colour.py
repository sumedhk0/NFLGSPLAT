"""08f: each id's team from its torso colour, measured against identity's label.

WHY. The render, the helmets and the jersey numbers all key on identity's
team. On play 2 identity called 62 of 85 ids KC where 11 play; the torso
saturation of the same ids splits cleanly (26 ids at 24-80, 26 at 102-168,
nothing between) and disagrees with identity on 22 of 52. A red kit and a
white kit are one number apart under any lighting.

WHAT. Per global id, the median HSV saturation of the torso band over its
detections in both views (identity.torso_colours); a two-way split at the
largest gap in the sorted medians, the saturated side the red kit, the
other the white. Refuses when the gap is not clear (no bimodality: a game
of two coloured kits needs hue, not saturation). A two-means on the
chroma plane (saturation along the median hue) was tried and measured
WORSE: 80 of 96 ids on play 1 and 63 of 82 on play 2 fell to the white
side against 11 players each, and it agreed with identity less; median
hue on 140 px crops is too unstable a coordinate. Writes
``<play-dir>/team_by_colour.json``: ``{pid: {"team", "saturation",
"margin"}}`` and prints the agreement with identity.

USAGE:
  python scripts/08f_team_by_colour.py --play-dir data/all22/<game>/play_002 --red KC --white BAL
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.identity.torso_colours import detection_colours  # noqa: E402

MIN_CROPS = 10
MIN_GAP = 15.0            # saturation units (0-255); below this the split is not trusted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--red", required=True, help="team in the saturated kit, e.g. KC")
    ap.add_argument("--white", required=True, help="team in the white kit, e.g. BAL")
    ap.add_argument("--cams", nargs="+", default=["sideline", "endzone"])
    args = ap.parse_args()

    P = args.play_dir
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df.track_id >= 0]
    hsv: dict[int, list] = {}
    for cam in args.cams:
        dv = df[df.cam == cam].reset_index(drop=True)
        video = P / f"{cam}.mp4"
        if not len(dv) or not video.exists():
            continue
        cols = detection_colours(dv, video)
        ok = np.isfinite(cols).all(1)
        for pid, c in zip(dv.global_player_id.to_numpy()[ok], cols[ok]):
            hsv.setdefault(int(pid), []).append(c)
    ids = [pid for pid, v in hsv.items() if len(v) >= MIN_CROPS]
    if len(ids) < 6:
        print(f"only {len(ids)} ids with >= {MIN_CROPS} crops; nothing to split", file=sys.stderr)
        sys.exit(2)
    med = {pid: float(np.median(np.asarray(hsv[pid], float)[:, 1])) for pid in ids}
    order = sorted(med.values())
    gaps = np.diff(order)
    k = int(np.argmax(gaps))
    gap = float(gaps[k])
    threshold = 0.5 * (order[k] + order[k + 1])
    if gap < MIN_GAP or k < 2 or k > len(order) - 3:
        print(f"no clear two-way split: largest gap {gap:.0f} at {threshold:.0f} "
              f"({k + 1} below, {len(order) - k - 1} above); refusing", file=sys.stderr)
        sys.exit(3)
    out = {}
    for pid, s_ in med.items():
        out[pid] = {"team": args.red if s_ > threshold else args.white, "saturation": round(s_, 1),
                    "margin": round(abs(s_ - threshold), 1)}
    ident_path = P / "identity_resolved.pkl"
    agree = n_ident = 0
    if ident_path.exists():
        merged = pickle.load(open(ident_path, "rb")).get("merged", {})
        for pid, rec in out.items():
            t = getattr(merged.get(pid), "team", None)
            if t in (args.red, args.white):
                n_ident += 1
                agree += int(t == rec["team"])
    n_red = sum(1 for r in out.values() if r["team"] == args.red)
    print(f"{len(out)} ids: split at saturation {threshold:.0f} (gap {gap:.0f}); "
          f"{n_red} {args.red}, {len(out) - n_red} {args.white}; "
          f"identity agrees on {agree}/{n_ident}")
    (P / "team_by_colour.json").write_text(json.dumps(
        {"red": args.red, "white": args.white, "threshold": threshold, "gap": gap,
         "teams": {str(k): v for k, v in out.items()}}, indent=1))
    print(f"wrote {P / 'team_by_colour.json'}")


if __name__ == "__main__":
    main()
