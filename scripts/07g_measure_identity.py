#!/usr/bin/env python
"""Measure the identity pipeline against ground truth, for the first time.

Identity is the project's bottleneck. Geometry places a player 0.23 m from
truth once you know WHO they are, and the helmet detector finds 94% of them --
so what is left is deciding which of the 22 each detection belongs to, and until
now that could only be checked by eye on our own footage (16 of 22 named, 4
independently verified).

The Helmet Assignment set labels every box with the player, so each stage can be
scored separately:

    reads       how often OCR produced any number at all for a track
    top-1       of tracks that got a read, how often the most-voted number is
                the right one -- OCR quality alone, before any assignment
    assigned    accuracy after the one-to-one assignment against the roster,
                which is what the pipeline actually outputs
    team        accuracy of the colour-based two-team split

WHAT IS GIVEN AND WHAT IS EARNED. Single-view track ids are given -- each
labelled box already carries its player -- so this measures identity on top of
perfect tracking, not both at once. The ROSTER is also given, which is fair:
production gets it from participation data, and the repo already has
participation.py for that. What is earned is the mapping from tracks to numbers.

BOXES ARE HELMETS, NOT BODIES. jersey_ocr wants a player box and reads the band
18-55% down it; this dataset marks helmets. A body box is synthesised from each
helmet by scaling with the helmet's own size, which is the only distance cue
available per box.
"""
from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.identity.jersey_ocr import read_jerseys
from nfl_gsplat.identity.jersey_vote import assign, restrict_to_known
from nfl_gsplat.identity.team_color import split_two_teams
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking

# A helmet box is about one head high. A standing player is roughly 6.5 heads,
# and about 3 heads wide at the shoulders, so the body box is built from the
# helmet's height -- the only per-box distance cue there is.
BODY_HEIGHT_IN_HELMETS: float = 6.5
BODY_HALF_WIDTH_IN_HELMETS: float = 1.6


@dataclass
class Row:
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    track_id: int


def body_box(left, top, width, height, w_img, h_img):
    """Player box implied by a helmet box."""
    cx = left + width / 2.0
    half = BODY_HALF_WIDTH_IN_HELMETS * height
    return Row(max(0.0, cx - half), max(0.0, top),
               min(w_img, cx + half),
               min(h_img, top + BODY_HEIGHT_IN_HELMETS * height), -1)


def frames_from(path, wanted_frames):
    import cv2

    cap = cv2.VideoCapture(str(path))
    for f in sorted(wanted_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if ok:
            yield int(f), img
    cap.release()


def read_view(root, labels, game, play, view, *, n_frames, gpu):
    """(votes, colours, index, truth_jersey, truth_team) for one camera."""
    name = f"{game}_{play:06d}_{view}.mp4"
    sub = labels[labels["video"] == name]
    if sub.empty:
        return None

    # Ground truth: label "H90" -> team H, jersey 90.
    players = sorted(sub["label"].unique())
    truth_jersey = {p: int(p[1:]) for p in players if p[1:].isdigit()}
    truth_team = {p: p[0] for p in truth_jersey}
    index = {p: i for i, p in enumerate(sorted(truth_jersey))}
    if len(index) < 10:
        return None

    all_frames = sorted(sub["frame"].unique())
    pick = set(np.array(all_frames)[
        np.linspace(0, len(all_frames) - 1, min(n_frames, len(all_frames))).astype(int)])

    wanted = collections.defaultdict(list)
    for f, grp in sub[sub["frame"].isin(pick)].groupby("frame"):
        for r in grp.itertuples():
            if r.label not in index:
                continue
            box = body_box(r.left, r.top, r.width, r.height, 1280, 720)
            box.track_id = index[r.label]
            wanted[int(f)].append(box)
    if not wanted:
        return None

    votes, colours = read_jerseys(frames_from(root / "video" / name, wanted),
                                  wanted, gpu=gpu)
    # Key by PLAYER, not by this view's arbitrary index, so two views can be
    # pooled without their indices having to agree.
    inv0 = {i: p for p, i in index.items()}
    return ({inv0[t]: c for t, c in votes.items()},
            {inv0[t]: c for t, c in colours.items()},
            truth_jersey, truth_team)


def score(votes_by_player, colours_by_player, truth_jersey, truth_team):
    """Run the pipeline on player-keyed votes and score it."""
    index = {p: i for i, p in enumerate(sorted(truth_jersey))}
    inv = {i: p for p, i in index.items()}
    votes = {index[p]: c for p, c in votes_by_player.items() if p in index}
    colours = {index[p]: c for p, c in colours_by_player.items() if p in index}
    roster = sorted({truth_jersey[p] for p in index})
    kept = restrict_to_known(votes, roster)

    n_read = len(kept)
    top1 = sum(1 for t, c in kept.items()
               if c and c.most_common(1)[0][0] == truth_jersey[inv[t]])

    # The roster this dataset can supply: number and team, from the label. It
    # carries no names, positions or heights, so those columns are filled with
    # placeholders -- which means the position bonus and any height reasoning
    # are inert here and the score is on the JERSEY EVIDENCE alone. In
    # production those columns come from participation data and would help.
    on_field = pd.DataFrame({
        "jersey_number": [truth_jersey[inv[i]] for i in sorted(inv)],
        "team": [truth_team[inv[i]] for i in sorted(inv)],
        "position": ["UNK"] * len(inv),
        "full_name": [inv[i] for i in sorted(inv)],
        "height_m": [1.85] * len(inv)})

    # Team split from colour, scored on its own before it is used.
    team_acc = None
    if len(colours) >= 4:
        tracks_c = sorted(colours)
        lab = split_two_teams(np.stack([colours[t] for t in tracks_c]))
        truth_lab = np.array([truth_team[inv[t]] == "H" for t in tracks_c])
        agree = float((lab.astype(bool) == truth_lab).mean())
        team_acc = round(max(agree, 1.0 - agree), 3)   # label order is arbitrary

    got = assign(kept, on_field)
    correct = 0
    for ident in got:
        want = truth_jersey.get(inv.get(ident.track_id, ""), None)
        if want is not None and int(ident.jersey) == want:
            correct += 1

    return {
        "tracks": len(index),
        "tracks_with_reads": n_read,
        "read_rate": round(n_read / len(index), 3),
        "top1_of_read": round(top1 / n_read, 3) if n_read else None,
        "assigned": len(got),
        "assigned_correct": correct,
        "assign_accuracy": round(correct / len(index), 3),
        # Recall names every track; precision asks whether what it DID name was
        # right. assign() deliberately refuses low-margin tracks -- a wrong
        # identity is worse than a missing one -- so the two are very different
        # numbers and only the pair says whether the fix is more evidence or a
        # better assignment.
        "assign_precision": (round(correct / len(got), 3) if got else None),
        "team_accuracy": team_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="Sideline,Endzone")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--merge-views", action="store_true",
                    help="also pool both cameras' votes onto one track")
    args = ap.parse_args()

    out_path = args.out or (args.root / "identity_accuracy.json")
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")
    keys = sorted(plays)
    if args.limit:
        keys = keys[:args.limit]
    views = [v for v in args.views.split(",") if v]
    print(f"{len(keys)} plays x {len(views)} views, {args.frames} frames each\n")

    out = {}
    for i, (game, play) in enumerate(keys, 1):
        per_view_reads = {}
        for view in views:
            got = None
            read = read_view(args.root, labels, game, play, view,
                             n_frames=args.frames, gpu=not args.cpu)
            if read is not None:
                per_view_reads[view] = read
                got = score(*read)
            if got is None:
                got = {"failed": "no usable labels"}
            out[f"{game}_{play}_{view}"] = got
            if "failed" in got:
                print(f"[{i:3d}] {game}/{play:<6d} {view:8s} FAILED {got['failed']}",
                      flush=True)
            else:
                t1 = got["top1_of_read"]
                tm = got["team_accuracy"]
                print(f"[{i:3d}] {game}/{play:<6d} {view:8s} "
                      f"reads {got['read_rate']:5.0%}  "
                      f"top1 {t1 if t1 is not None else float('nan'):5.0%}  "
                      f"assigned {got['assign_accuracy']:5.0%} "
                      f"({got['assigned_correct']}/{got['tracks']})  "
                      f"team {tm if tm is not None else float('nan'):5.0%}",
                      flush=True)
        # Pool the two cameras' jersey evidence onto ONE track per player.
        # The endzone camera is zoomed several times tighter and reads numbers
        # far better, while the sideline sees players the endzone has occluded;
        # neither alone is the best available evidence. This ASSUMES cross-view
        # correspondence, which is measured separately at 98.2% per track, so
        # the realistic gain is that fraction of what is reported here.
        if args.merge_views and len(per_view_reads) == 2:
            a, b = per_view_reads.values()
            merged_votes = collections.defaultdict(collections.Counter)
            for src in (a[0], b[0]):
                for player, counter in src.items():
                    merged_votes[player].update(counter)
            merged_col = dict(a[1])
            merged_col.update(b[1])
            truth_j = dict(a[2]); truth_j.update(b[2])
            truth_t = dict(a[3]); truth_t.update(b[3])
            m = score(dict(merged_votes), merged_col, truth_j, truth_t)
            if m is not None:
                out[f"{game}_{play}_MERGED"] = m
                print(f"[{i:3d}] {game}/{play:<6d} {'MERGED':8s} "
                      f"reads {m['read_rate']:5.0%}  "
                      f"top1 {m['top1_of_read'] or float('nan'):5.0%}  "
                      f"assigned {m['assign_accuracy']:5.0%} "
                      f"({m['assigned_correct']}/{m['tracks']})", flush=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    good = [v for v in out.values() if "failed" not in v]
    print(f"\n{len(good)}/{len(out)} views  ->  {out_path}")
    if good:
        rr = np.array([v["read_rate"] for v in good])
        aa = np.array([v["assign_accuracy"] for v in good])
        t1 = np.array([v["top1_of_read"] for v in good if v["top1_of_read"] is not None])
        tm = np.array([v["team_accuracy"] for v in good if v["team_accuracy"] is not None])
        print(f"read rate      median {np.median(rr):5.0%}")
        print(f"top-1 of read  median {np.median(t1):5.0%}" if len(t1) else "")
        print(f"assignment     median {np.median(aa):5.0%}  "
              f"(best {aa.max():.0%}, worst {aa.min():.0%})")
        if len(tm):
            print(f"team split     median {np.median(tm):5.0%}")


if __name__ == "__main__":
    main()
