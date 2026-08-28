"""Name the players on a play, from jersey votes plus formation alignment.

Ties together the pieces that were previously wired by hand: grounding,
fragment stitching, truncation-credited jersey votes, formation roles, one-to-one
assignment against the league's own participation record, the cross-camera merge,
and the geometric check that can refute a pairing.

Jersey votes come from a cache written by the OCR pass, one per camera, so this
runs in seconds and can be re-run against different settings without touching a
GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity import formation as fm
from nfl_gsplat.identity.merge_cameras import coverage
from nfl_gsplat.identity.participation import (fetch_nflverse, find_play,
                                               game_id, players_on_play)
from nfl_gsplat.identity.resolve import resolve_camera, resolve_play
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--votes", action="append", default=[], metavar="CAM=PATH",
                    help="jersey-vote cache per camera, e.g. sideline=votes.pkl")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="all22 manifest.jsonl, for the play's title")
    ap.add_argument("--play-index", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truncation-credit", type=float, default=None)
    args = ap.parse_args()

    if not args.votes:
        raise SetupError("--votes CAM=PATH is required at least once")
    vote_paths = {}
    for spec in args.votes:
        if "=" not in spec:
            raise SetupError(f"--votes wants CAM=PATH, got {spec!r}")
        cam, path = spec.split("=", 1)
        vote_paths[cam] = Path(path)

    meta = yaml.safe_load((args.play_dir / "meta.yaml").read_text(encoding="utf-8"))
    season, week = int(meta["season"]), int(meta["week"])
    away, home = str(meta["away_team"]), str(meta["home_team"])
    fps = float(meta.get("fps", 59.94))

    paths = fetch_nflverse(season)
    pbp = pd.read_parquet(paths["pbp"])
    participation = pd.read_parquet(paths["participation"])
    roster = pd.read_parquet(paths["weekly_roster"])
    game = game_id(season, week, away, home)
    titles = [json.loads(line) for line in
              args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    match = [t for t in titles if int(t.get("index", -1)) == args.play_index]
    if not match:
        raise SetupError(
            f"no entry with index {args.play_index} in {args.manifest}")
    play = find_play(pbp, game, match[0]["title"])
    on_field = players_on_play(participation, roster, game,
                               int(play["play_id"]), week)
    prow = participation[(participation.nflverse_game_id == game)
                         & (participation.play_id == int(play["play_id"]))].iloc[0]
    los_x = fm.line_of_scrimmage_x(str(play["yrdln"]), str(play["posteam"]))
    offense = fm.parse_personnel(prow["offense_personnel"])
    defense = fm.parse_personnel(prow["defense_personnel"])
    _LOG.info("%s play %s: LOS %.2f m, %d offence, %d defence", game,
              play["play_id"], los_x, sum(offense.values()), sum(defense.values()))

    cams = load_camera_track(args.play_dir / "cameras.npz")
    tracks_df = pd.read_parquet(args.play_dir / "tracks.parquet")

    per_camera, stitch_maps = {}, {}
    for cam, path in vote_paths.items():
        if cams.get(cam) is None:
            raise SetupError(f"no {cam!r} camera in cameras.npz")
        saved = pickle.load(open(path, "rb"))
        votes = {int(k): collections.Counter(v)
                 for k, v in saved["votes"].items()}
        identities, positions, stitched = resolve_camera(
            cams[cam], tracks_df, cam, on_field, votes,
            team_by_track=saved.get("teams"), los_x=los_x, offense=offense,
            defense=defense, fps=fps,
            truncation_credit=args.truncation_credit)
        per_camera[cam] = (identities, positions)
        stitch_maps[cam] = stitched

    merged, checks = resolve_play(per_camera)
    n_named, missing = coverage(merged, on_field)

    print(f"\n{'#':>4} {'player':24s} {'team':>4} {'pos':>3} {'ht':>5} "
          f"{'lb':>4} {'cameras':10s} {'votes':>5}  cross-camera")
    for jersey in sorted(merged, key=lambda j: (merged[j].team, j)):
        p = merged[jersey]
        row = on_field[on_field.jersey_number == jersey].iloc[0]
        check = checks.get(jersey)
        if check is None:
            verdict = "one camera"
        elif not check.testable:
            verdict = f"untestable ({check.n_frames} shared frames)"
        elif check.contradicted:
            # Reached when a contradicted pairing was REPAIRED to its
            # better-evidenced camera: the player is kept, but the two views
            # disagreed, and calling that "verified" because it was merely
            # measurable is the silent wrongness this pipeline exists to catch.
            verdict = f"REPAIRED, cameras disagreed by {check.separation_m:.1f} m"
        else:
            verdict = f"VERIFIED {check.separation_m:.1f} m"
        print(f"{('#%d' % jersey):>4} {p.player:24s} {p.team:>4} "
              f"{str(row['position']):>3} {p.height_m:.2f} {p.weight_lb:4.0f} "
              f"{'+'.join(c[:4] for c in p.cameras):10s} {p.total_votes:5d}  "
              f"{verdict}")
    print(f"\n{n_named}/{n_named + len(missing)} named; missing:")
    for j, name, team, pos in sorted(missing, key=lambda m: (m[2], m[0])):
        print(f"   #{j:<3d} {team:>4} {name:24s} {pos}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        # The stitch map travels WITH the roster. Identities are keyed by
        # stitched player id while every pose and detection cache downstream is
        # keyed by the tracker's raw fragment id, so without this the renderer
        # silently finds no identified players and draws everyone generic.
        pickle.dump({"merged": merged, "checks": checks,
                     "stitch": stitch_maps}, fh)
    _LOG.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
