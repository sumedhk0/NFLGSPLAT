"""Geometry-free identity columns for calibration tracks.

Adds `team` + `player_uid` per track using ONLY jersey OCR + jersey color --
no camera, no geometry -- so joining sideline and endzone on `player_uid` gives
cross-camera identity without either endzone estimate. Team labels come from a
GLOBAL two-color split across BOTH cameras, so the same team gets the same label
(and uid) in both views."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.identity.registry import IdentityMatchConfig, resolve_tracks
from nfl_gsplat.identity.roster import OcrOnlySource
from nfl_gsplat.identity.team_color import dominant_jersey_color, split_two_teams


def _one_crop_per_track(tracks_df, crop_provider):
    """{(cam, track_id): crop} using the first frame each track appears."""
    crops = {}
    for (cam, tid), grp in tracks_df.groupby(["cam", "track_id"]):
        fr = int(grp["frame"].min())
        c = crop_provider(cam, fr, int(tid))
        if c is not None and getattr(c, "size", 0):
            crops[(cam, int(tid))] = c
    return crops


def assign_identity_columns(tracks_df, crop_provider, *, season):
    """Return a copy of tracks_df with `team` + `player_uid` columns."""
    crops = _one_crop_per_track(tracks_df, crop_provider)
    keys = list(crops)
    team_by_key: dict[tuple[str, int], str] = {}
    if keys:
        colors = np.stack([dominant_jersey_color(crops[k]) for k in keys])
        labels = split_two_teams(colors)          # global split, both cams
        team_by_key = {k: f"T{int(lbl)}" for k, lbl in zip(keys, labels)}

    out = tracks_df.copy()
    out["team"] = [team_by_key.get((c, int(t))) for c, t in
                   zip(out["cam"], out["track_id"])]

    cfg = IdentityMatchConfig(season=season)
    src = OcrOnlySource()
    parts = []
    for cam, grp in out.groupby("cam"):
        cand = src.candidates_for_play(str(cam), "0")      # [] -> synthesize
        parts.append(resolve_tracks(grp, cand, cfg, id_col="track_id"))
    return pd.concat(parts, ignore_index=True)
