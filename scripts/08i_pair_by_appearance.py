#!/usr/bin/env python
"""Pair per-camera tracks across cameras by appearance (number, kit), then position.

Reads a tracks parquet whose ``(cam, track_id)`` are per-camera tracks --
08b with ``--pairing track --pair-gap 0`` keeps every camera track its own
id -- with 08b's ``kit_margin`` and, when 08c's OCR has run on it,
``jersey_number_ocr`` per row. Writes ``tracks_identity.parquet`` (global
ids in ``track_id`` and ``global_player_id``, the OCR column carried) so
``08c --from-cache`` votes names per global id without a second OCR pass.
See tracking.pair_by_appearance for the rules and why.

Printed: pairs by evidence, two-camera ids and how many cross kits (must be
0 by construction where both kits are known), paired players per frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.tracking.pair_by_appearance import (GAP_NUMBER_M, GAP_POSITION_M, cam_tracks_from_frame,
                                                    global_ids, pair_by_appearance)


def ankles_for(play: Path, df: pd.DataFrame):
    """The ankle keypoints' ground points keyed by THIS table's ids. The keypoints (05m) carry
    the ids of the tracks.parquet they were run on; when that is not this table (a re-pairing
    from the unpaired one), the ids are carried across by the boxes (tracking.relabel)."""
    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.render.play_timeline import ankle_ground
    from nfl_gsplat.tracking.relabel import id_map_by_boxes, relabel_keypoints

    kp = play / "keypoints_2d.parquet"
    if not kp.exists():
        return None
    kdf = pd.read_parquet(kp)
    keys = set(zip(df["cam"].astype(str), df["frame"].astype(int), df["track_id"].astype(int)))
    sample = list(zip(kdf["cam"].astype(str), kdf["frame"].astype(int), kdf["global_player_id"].astype(int)))[::97]
    hit = sum(k in keys for k in sample) / max(len(sample), 1)
    if hit < 0.9 and (play / "tracks.parquet").exists():
        old = pd.read_parquet(play / "tracks.parquet")
        old = old[old["track_id"] >= 0]
        kdf, dropped = relabel_keypoints(kdf, id_map_by_boxes(old, df))
        print(f"keypoints carried to this table's ids by their boxes ({dropped} rows without a box dropped)")
    elif hit < 0.9:
        print("keypoints carry other ids and there is no tracks.parquet to map them by; box bottoms only")
        return None
    ank = ankle_ground(kdf, load_camera_track(play / "cameras.npz"))
    print(f"ground points from the ankle keypoints on {len(ank)} (camera, frame, id); box bottoms elsewhere")
    return ank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, default=None,
                    help="input parquet (default <play-dir>/tracks_identity.parquet if it carries OCR, else tracks.parquet)")
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/tracks_identity.parquet")
    ap.add_argument("--gap-number", type=float, default=GAP_NUMBER_M)
    ap.add_argument("--gap-position", type=float, default=GAP_POSITION_M)
    ap.add_argument("--lag", type=int, default=0, help="endzone frame lag (sideline f <-> endzone f + lag)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ankles", action="store_true",
                    help="ground points from the box bottoms alone (default: the ankle keypoints where a camera has them)")
    args = ap.parse_args()
    play = args.play_dir
    src = args.tracks
    if src is None:
        cand = play / "tracks_identity.parquet"
        src = cand if cand.exists() else play / "tracks.parquet"
    df = pd.read_parquet(src)
    df = df[df["track_id"] >= 0].reset_index(drop=True)
    has_ocr = "jersey_number_ocr" in df and int((df["jersey_number_ocr"] >= 0).sum()) > 0
    cams = load_camera_track(play / "cameras.npz")
    ankles = None if args.no_ankles else ankles_for(play, df)
    side, end = cam_tracks_from_frame(df, cams, ankles=ankles)
    n_num = sum(t.number >= 0 for t in side + end)
    n_kit = sum(t.kit >= 0 for t in side + end)
    print(f"{src.name}: sideline {len(side)} tracks, endzone {len(end)}; kit known on {n_kit}, "
          f"number read on {n_num}{'' if has_ocr else ' (no OCR column: kit and position only)'}")
    pairs = pair_by_appearance(side, end, gap_number=args.gap_number, gap_position=args.gap_position, lag=args.lag)
    by = {}
    for p in pairs:
        by[p.evidence] = by.get(p.evidence, 0) + 1
    gs, ge = global_ids(len(side), len(end), pairs)
    print(f"{len(pairs)} pairs: " + ", ".join(f"{k} {v}" for k, v in by.items())
          + f"; mean offset median {np.median([p.offset for p in pairs]) if pairs else float('nan'):.2f} m")
    # rewrite ids
    key_to_gid = {}
    for t, g in zip(side, gs):
        key_to_gid[("sideline", t.tid)] = int(g)
    for t, g in zip(end, ge):
        key_to_gid[("endzone", t.tid)] = int(g)
    n_all = len(key_to_gid)
    keys = list(zip(df["cam"].astype(str), df["track_id"].astype(int)))
    gid = np.array([key_to_gid.get(k, -1) for k in keys], int)
    # camera tracks too short for a series keep a unique id
    nxt = int(gid.max()) + 1 if len(gid) else 0
    for k in sorted(set(keys)):
        if k not in key_to_gid:
            key_to_gid[k] = nxt
            nxt += 1
    gid = np.array([key_to_gid[k] for k in keys], int)
    out = df.copy()
    out["track_id"] = gid
    out["global_player_id"] = gid
    both = out.groupby("track_id")["cam"].nunique()
    per = out.groupby(["frame", "track_id"])["cam"].nunique()
    f = per.groupby("frame").agg(ids="size", paired=lambda s: int((s == 2).sum()))
    cross = -1
    if "kit_margin" in out:
        m = out["kit_margin"].to_numpy(float)
        out_k = out.assign(kit=np.where(np.isfinite(m) & (np.abs(m) >= 0.4), (m > 0).astype(int), -1))
        k = out_k[out_k["kit"] >= 0].groupby(["track_id", "cam"])["kit"].agg(lambda s: int(s.mean() > 0.5)).unstack()
        if {"sideline", "endzone"} <= set(k.columns):
            k = k.dropna()
            cross = int((k["sideline"] != k["endzone"]).sum())
    print(f"global ids {out['track_id'].nunique()} ({int((both == 2).sum())} in both cameras, {cross} cross-kit); "
          f"per frame ids {f['ids'].median():.0f}, paired {f['paired'].median():.0f}, one-view {(f['ids'] - f['paired']).median():.0f}")
    if args.dry_run:
        return
    dst = args.out or play / "tracks_identity.parquet"
    if dst.exists() and dst.resolve() == Path(src).resolve():
        backup = dst.with_name(dst.stem + "_unpaired.parquet")
        Path(src).replace(backup)                 # this run's input, every run (an old keep misled a measurement)
        print(f"kept the input as {backup.name}")
    out.to_parquet(dst, index=False)
    print(f"wrote {dst} ({len(out)} rows)")


if __name__ == "__main__":
    main()
