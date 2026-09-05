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

from nfl_gsplat.errors import SetupError

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
                         "{season}/roster_weekly.parquet, written by "
                         "scripts/fetch_nflverse_rosters.py); without it the "
                         "season file names a number by whoever wore it last")
    ap.add_argument("--home", default=None,
                    help="home team code; with it, the colour cluster nearest "
                         "the home team's primary colour is the home team and "
                         "the other is the away team (white on the road), "
                         "instead of the roster-overlap vote")
    ap.add_argument("--teams", default="KC,BAL",
                    help="the game's two team codes; the colour split is "
                         "mapped onto them by which roster explains its numbers")
    ap.add_argument("--saturated", default=None,
                    help="the team in the COLOURED kit (e.g. KC in red against BAL in white): "
                         "colour label T1 maps to it outright and the roster vote is printed as "
                         "a check; without it the roster vote maps the colours")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse tracks_identity.parquet instead of re-running OCR")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--facing-min", type=float, default=0.0,
                    help="drop crops whose posed body is side-on to the lens (identity."
                         "facing score below this); needs poses_<cam>.json; 0 = off")
    ap.add_argument("--pool-views", action="store_true",
                    help="one vote per player over both views instead of per view")
    args = ap.parse_args()

    from nfl_gsplat.calibration.identity_precompute import assign_identity_columns
    from nfl_gsplat.tracking.jersey_ocr import JerseyOCRConfig, vote_jersey_numbers

    play = args.play_dir
    from nfl_gsplat.identity.torso_colours import team_votes
    videos = {cam: play / f"{cam}.mp4" for cam in ("sideline", "endzone")}
    if args.from_cache and (play / "tracks_identity.parquet").exists():
        df = pd.read_parquet(play / "tracks_identity.parquet")
        print(f"{len(df)} detections from cache, {df['track_id'].nunique()} global ids")
        # The OCR is cached; the kit split is cheap (one pass over the videos)
        # and is recomputed so the cache carries rule D too.
        votes, centres = team_votes(df, videos)
        before = df["team"].copy()
        df["team"] = [votes.get((str(c), int(t))) for c, t in zip(df["cam"], df["track_id"])]
        df["team"] = df["team"].map(lambda v: None if v is None else f"T{int(v)}")
        print("kit split by saturation votes: " + ", ".join(
            f"{cam} centres {lo:.0f}/{hi:.0f}" for cam, (lo, hi) in centres.items())
            + f"; {int((before != df['team']).sum())} of {len(df)} detections relabelled")
        df.to_parquet(play / "tracks_identity.parquet", index=False)
    else:
        df = pd.read_parquet(play / "tracks.parquet")
        df = df[df["track_id"] >= 0].reset_index(drop=True)
        print(f"{len(df)} detections, {df['track_id'].nunique()} global ids")
        votes, centres = team_votes(df, videos)
        print("kit split by saturation votes: " + ", ".join(
            f"{cam} centres {lo:.0f}/{hi:.0f}" for cam, (lo, hi) in centres.items()))
        cfg = JerseyOCRConfig(backend=args.ocr_backend, top_k_frames=args.ocr_top_k,
                              use_gpu=not args.cpu, min_facing=args.facing_min,
                              pool_views=args.pool_views)
        facing_of = None
        if args.facing_min > 0:
            from nfl_gsplat.identity import facing as fc
            tables = {}
            for cam in videos:
                cache_path = play / f"poses_{cam}.json"
                if not cache_path.exists():
                    raise SetupError(f"--facing-min needs {cache_path} (scripts/05c)")
                blob = pickle.load(open(cache_path, "rb"))
                tables[cam] = (fc.facing_table(blob["frames"]), int(blob.get("stride", 6)))
            print("facing gate from poses: " + ", ".join(
                f"{cam} {len(t)} posed detections" for cam, (t, _s) in tables.items()))

            def facing_of(cam, frame, tid):
                table, stride = tables[cam]
                return fc.nearest(table, frame, tid, max_gap=stride)
        df = vote_jersey_numbers(df, videos, cfg, facing_of=facing_of)
        df = assign_identity_columns(df, crop_provider(videos), season=args.season,
                                     team_by_key=votes)
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
    # fetch_roster.py writes five columns (no week, name as player_name, no
    # height/weight); fetch_nflverse_rosters.py writes the full weekly table.
    # Take what is there and say so, rather than crash on the poorer file.
    if "full_name" not in roster.columns:
        roster = roster.assign(full_name=roster.get("player_name", "?"))
    for col in ("height", "weight"):
        if col not in roster.columns:
            roster = roster.assign(**{col: np.nan})
    if "week" in roster.columns:
        roster = roster.sort_values("week")          # latest week wins below
    else:
        print("roster has no week column (fetch_roster.py output); using it as is")
    roster = (roster.drop_duplicates(["team", "jersey_number"], keep="last")
              .set_index(["team", "jersey_number"]))
    numbers_of = {t: set(int(n) for _t, n in roster.index if _t == t) for t in teams}

    # Which real team is each colour team? The one whose roster explains more
    # of the numbers read on that colour. Same number can exist on both
    # rosters, so it is a vote, not a lookup.
    team_of = df.groupby("track_id")["team"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "?")
    labels = sorted(set(team_of) - {"?"})
    mapping = {}
    # Only numbers that exist on exactly ONE of the two rosters are evidence;
    # a number both teams carry says nothing, and counting it produced 10-10
    # ties. Ties are broken by the shared numbers only when the unique ones
    # are silent.
    unique = {t: numbers_of[t] - set().union(*(numbers_of[o] for o in teams if o != t))
              for t in teams}
    for lab in labels:
        nums = [int(read[g]) for g in read.index if team_of.get(g) == lab and read[g] >= 0]
        score = {t: sum(n in unique[t] for n in nums) for t in teams}
        if nums and max(score.values()) == 0:
            score = {t: sum(n in numbers_of[t] for n in nums) for t in teams}
        mapping[lab] = max(score, key=score.get) if nums else "?"
        print(f"colour team {lab}: {len(nums)} numbers read; unique to "
              + ", ".join(f"{t} {score[t]}" for t in teams) + f" -> {mapping[lab]}")
    if args.saturated is not None:
        if args.saturated not in teams:
            raise SetupError(f"--saturated {args.saturated} is not one of {teams}")
        other = [t for t in teams if t != args.saturated][0]
        by_colour = {"T1": args.saturated, "T0": other}
        disagree = [lab for lab in labels if mapping.get(lab) not in ("?", by_colour.get(lab))]
        print(f"kit colour names the teams: T1 (saturated) = {args.saturated}, T0 = {other}"
              + (f"; the roster vote DISAGREES on {disagree} (it said {[mapping[l] for l in disagree]})"
                 if disagree else "; the roster vote agrees"))
        mapping = {lab: by_colour.get(lab, "?") for lab in labels}
    elif args.home is not None and len(labels) == 2 and args.home in teams:
        # Uniforms decide. Each colour cluster's mean jersey colour (BGR crops,
        # identity_precompute) against the home team's primary colour; the
        # nearer cluster is home, the other away. The roster vote tied 10-10
        # on play 2; this does not tie.
        from nfl_gsplat.identity.team_color import dominant_jersey_color
        import cv2

        primary = {"KC": (0.89, 0.09, 0.22), "BAL": (0.14, 0.13, 0.44),
                   "SEA": (0.11, 0.22, 0.34), "ARI": (0.62, 0.11, 0.22)}
        home_rgb = np.asarray(primary.get(args.home, (0.5, 0.5, 0.5)))
        cluster_rgb = {}
        caps = {cam: cv2.VideoCapture(str(play / f"{cam}.mp4")) for cam in ("sideline", "endzone")}
        for lab in labels:
            rows = df[df["team"] == lab].sample(n=min(60, int((df["team"] == lab).sum())),
                                                random_state=0)
            cols = []
            for r in rows.itertuples():
                cap = caps[r.cam]
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
                ok, img = cap.read()
                if not ok:
                    continue
                y1, y2 = int(r.bbox_y1), int(r.bbox_y2)
                ya, yb = int(y1 + 0.25 * (y2 - y1)), int(y1 + 0.6 * (y2 - y1))
                crop = img[max(0, ya):max(0, yb), max(0, int(r.bbox_x1)):max(0, int(r.bbox_x2))]
                if crop.size:
                    cols.append(dominant_jersey_color(crop)[::-1] / 255.0)   # BGR -> RGB
            cluster_rgb[lab] = np.mean(cols, axis=0) if cols else np.full(3, np.nan)
        for cap in caps.values():
            cap.release()
        dist = {lab: float(np.linalg.norm(cluster_rgb[lab] - home_rgb)) for lab in labels}
        home_lab = min(dist, key=dist.get)
        away = [t for t in teams if t != args.home][0]
        mapping = {lab: (args.home if lab == home_lab else away) for lab in labels}
        print("uniforms: " + ", ".join(f"{lab} mean rgb {np.round(cluster_rgb[lab], 2)} "
                                       f"dist to {args.home} {dist[lab]:.2f}" for lab in labels)
              + f" -> {mapping}")
    elif len(labels) == 2 and mapping[labels[0]] == mapping[labels[1]]:
        # Both colours claim the same team: give the weaker claim the other.
        a, b = labels
        na = sum(1 for g in read.index if team_of.get(g) == a and read[g] in numbers_of[mapping[a]])
        nb = sum(1 for g in read.index if team_of.get(g) == b and read[g] in numbers_of[mapping[b]])
        other = [t for t in teams if t != mapping[a]][0]
        mapping[b if nb <= na else a] = other
        print(f"   tie broken: {mapping}")

    merged = {}
    overruled = []
    for gid in sorted(df["track_id"].unique()):
        gid = int(gid)
        jersey = int(read.get(gid, -1))
        # The number decides the team where it can: a number on exactly one
        # of the two rosters names the team outright. The colour split mixes
        # the teams (measured: both clusters carried KC-unique numbers, and
        # Mahomes landed on the Ravens' side on play 2), so colour is only
        # the tie-breaker for numbers both rosters carry or none read.
        owners = [t for t in teams if jersey in unique[t]]
        colour_team = mapping.get(team_of.get(gid, "?"), "?")
        if colour_team != "?":
            # The kit decides (rule D, 88-100 % right per track); the number is
            # then looked up on THAT roster. A number unique to the other
            # roster means the OCR misread (75 % per track) or the kit was
            # misjudged; either way a wrong name paints a wrong number on the
            # avatar, so the id stays unnamed in the right kit instead.
            team = colour_team
            if len(owners) == 1 and owners[0] != team:
                overruled.append((gid, jersey, owners[0], team))
        else:
            team = owners[0] if len(owners) == 1 else "?"
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
    if overruled:
        print(f"   kit overruled the number on {len(overruled)} ids: "
              + ", ".join(f"id {g} #{j} is {o}-only, kit {t}" for g, j, o, t in overruled)[:400])
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
