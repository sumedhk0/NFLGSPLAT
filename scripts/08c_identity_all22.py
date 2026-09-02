#!/usr/bin/env python
"""Put names on the avatar twin: identity for an All-22 play-dir's fused tracks.

The production identity pass (03c) re-runs its own 2D tracker and overwrites
tracks.parquet. Here the tracks come from 08b -- both views reconciled on the
turf, one global id per player -- and identity is layered on them:

    OCR votes     jersey number per (cam, track), majority over the track's
                  best crops (tracking.jersey_ocr; easyocr on the GPU)
    team          two-way split of dominant jersey colour, both cams together
    roster        number + team -> the season roster (identity_precompute)

Output: tracks_identity.parquet (tracks.parquet plus jersey_number_ocr, team,
player_uid) and identity_resolved.pkl -- the blob 05d reads: one
PlayerIdentity per global id carrying jersey, name, team, height and weight
from the roster, tracks {sideline: id, endzone: id}. Players the roster could
not name keep a generic body (05d falls back), so nothing is invented.

Measured elsewhere (07g): this OCR reads the right number for ~75% of tracks
on the helmet set. Whatever it gets here is reported, not assumed.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.identity.merge_cameras import PlayerIdentity


def crop_provider(videos):
    import cv2

    caps = {cam: cv2.VideoCapture(str(p)) for cam, p in videos.items()}

    def provider(cam, frame, track_id, _df=None):
        cap = caps[cam]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        return img if ok else None

    return provider


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--ocr-backend", default="easyocr")
    ap.add_argument("--ocr-top-k", type=int, default=8)
    ap.add_argument("--rosters", type=Path, default=Path("data/rosters"))
    ap.add_argument("--week", type=int, default=None,
                    help="use the WEEKLY roster for this week (data/rosters/"
                         "{season}/roster_weekly.parquet); the season file "
                         "names a number by whoever wore it last")
    ap.add_argument("--teams", default="KC,BAL",
                    help="the game's two team codes; the colour split is "
                         "mapped onto them by which roster explains its numbers")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse tracks_identity.parquet instead of re-running OCR")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from nfl_gsplat.calibration.identity_precompute import assign_identity_columns
    from nfl_gsplat.tracking.jersey_ocr import JerseyOCRConfig, vote_jersey_numbers

    play = args.play_dir
    if args.from_cache and (play / "tracks_identity.parquet").exists():
        df = pd.read_parquet(play / "tracks_identity.parquet")
        print(f"{len(df)} detections from cache, {df['track_id'].nunique()} global ids")
    else:
        df = pd.read_parquet(play / "tracks.parquet")
        df = df[df["track_id"] >= 0].reset_index(drop=True)
        videos = {cam: play / f"{cam}.mp4" for cam in ("sideline", "endzone")}
        print(f"{len(df)} detections, {df['track_id'].nunique()} global ids")
        cfg = JerseyOCRConfig(backend=args.ocr_backend, top_k_frames=args.ocr_top_k,
                              use_gpu=not args.cpu)
        df = vote_jersey_numbers(df, videos, cfg)
        df = assign_identity_columns(df, crop_provider(videos), season=args.season)
        df.to_parquet(play / "tracks_identity.parquet", index=False)
    read = df.groupby("track_id")["jersey_number_ocr"].agg(
        lambda s: int(s[s >= 0].mode().iloc[0]) if (s >= 0).any() else -1)
    print(f"OCR: {(read >= 0).sum()} of {len(read)} global ids got a number")

    teams = [t.strip() for t in args.teams.split(",")]
    weekly = args.rosters / str(args.season) / "roster_weekly.parquet"
    if args.week is not None and weekly.exists():
        roster = pd.read_parquet(weekly)
        roster = roster[roster["week"] == args.week]
        print(f"weekly roster, week {args.week}: {len(roster)} rows")
    else:
        roster = pd.read_parquet(args.rosters / str(args.season) / "rosters.parquet")
    roster = roster[roster["team"].isin(teams) & roster["jersey_number"].notna()]
    # Latest week wins where a number changed hands during the season.
    roster = (roster.sort_values("week").drop_duplicates(["team", "jersey_number"], keep="last")
              .set_index(["team", "jersey_number"]))
    numbers_of = {t: set(int(n) for _t, n in roster.index if _t == t) for t in teams}

    # Which real team is each colour team? The one whose roster explains more
    # of the numbers read on that colour. Same number can exist on both
    # rosters, so it is a vote, not a lookup.
    team_of = df.groupby("track_id")["team"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "?")
    labels = sorted(set(team_of) - {"?"})
    mapping = {}
    for lab in labels:
        nums = [int(read[g]) for g in read.index if team_of.get(g) == lab and read[g] >= 0]
        score = {t: sum(n in numbers_of[t] for n in nums) for t in teams}
        mapping[lab] = max(score, key=score.get) if nums else "?"
        print(f"colour team {lab}: {len(nums)} numbers read; explained by "
              + ", ".join(f"{t} {score[t]}" for t in teams) + f" -> {mapping[lab]}")
    if len(labels) == 2 and mapping[labels[0]] == mapping[labels[1]]:
        # Both colours claim the same team: give the weaker claim the other.
        a, b = labels
        na = sum(1 for g in read.index if team_of.get(g) == a and read[g] in numbers_of[mapping[a]])
        nb = sum(1 for g in read.index if team_of.get(g) == b and read[g] in numbers_of[mapping[b]])
        other = [t for t in teams if t != mapping[a]][0]
        mapping[b if nb <= na else a] = other
        print(f"   tie broken: {mapping}")

    merged = {}
    for gid in sorted(df["track_id"].unique()):
        gid = int(gid)
        jersey = int(read.get(gid, -1))
        team = mapping.get(team_of.get(gid, "?"), "?")
        key = (team, float(jersey))
        row = roster.loc[key] if (jersey >= 0 and key in roster.index) else None
        if row is not None:
            name = str(row["full_name"])
            height_m = float(row["height"]) * 0.0254 if pd.notna(row["height"]) else 1.85
            weight_lb = float(row["weight"]) if pd.notna(row["weight"]) else 0.0
        else:
            name, height_m, weight_lb = f"P{gid}", 1.85, 0.0
        merged[gid] = PlayerIdentity(jersey=max(jersey, 0), player=name, team=team,
                                     height_m=height_m, weight_lb=weight_lb,
                                     tracks={"sideline": gid, "endzone": gid})
    pickle.dump({"merged": merged, "stitch": {}},
                open(play / "identity_resolved.pkl", "wb"))
    n_named = sum(1 for p in merged.values() if not p.player.startswith("P"))
    print(f"identity_resolved.pkl: {len(merged)} players, {n_named} named from the roster")
    if n_named:
        seen = sorted({(p.team, p.jersey, p.player) for p in merged.values()
                       if not p.player.startswith("P")})
        print("   " + ", ".join(f"{t} #{j} {n}" for t, j, n in seen)[:600])
    print(f"unnamed ids fall back to generic bodies: "
          f"{np.round(100 * (1 - n_named / max(len(merged), 1))):.0f}%")


if __name__ == "__main__":
    main()
