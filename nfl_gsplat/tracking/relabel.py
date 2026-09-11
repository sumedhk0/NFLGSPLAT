"""Carry per-id caches across a re-pairing of the camera tracks.

WHY. The keypoints (05m), the pose caches (05c) and every refit are keyed by
the global player id that 08i's pairing assigns. Re-pairing -- after the
endzone camera was refined to its paint (08l), which moves every endzone
ground point by up to 1.5 m -- renumbers ids, and the caches took hours of
GPU. The boxes themselves do not change, so a cache row is carried by the
box it came from: (camera, frame, box) -> new id.

RULE. Every old row must match exactly one new row on the box; anything else
is a different tracks table and raises. A pose cache is a pickle written
under numpy 1 (the smplx312 environment); relabelling under numpy 2 rewrites
it unloadable there, so the writer refuses outside numpy 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.errors import SetupError

BOX_COLS = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")


def _keys(df: pd.DataFrame):
    b = df[list(BOX_COLS)].to_numpy(float).round(2)
    return list(zip(df["cam"].astype(str), df["frame"].astype(int), map(tuple, b)))


def id_map_by_boxes(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """``{(cam, frame, old_id): new_id}`` for every old row, by the box it came from."""
    old_keys = _keys(old_df)
    new_keys = _keys(new_df)
    new_id = dict(zip(new_keys, new_df["track_id"].astype(int)))
    if len(new_id) != len(new_keys):
        raise SetupError("the new tracks table has two rows with the same (camera, frame, box)")
    out: dict = {}
    missing = 0
    for key, oid in zip(old_keys, old_df["track_id"].astype(int)):
        nid = new_id.get(key)
        if nid is None:
            missing += 1
            continue
        k = (key[0], key[1], int(oid))
        if k in out and out[k] != int(nid):
            raise SetupError(f"old id {oid} in {key[0]} frame {key[1]} maps to two new ids ({out[k]}, {nid})")
        out[k] = int(nid)
    if missing:
        raise SetupError(f"{missing} of {len(old_keys)} old rows have no box in the new table: "
                         "not the same tracks (the boxes must be identical, only the ids may differ)")
    return out


def relabel_keypoints(kdf: pd.DataFrame, mapping: dict):
    """``(kdf with global_player_id remapped, rows dropped)``; a row whose (cam, frame, id) has no
    mapping (a box below the tracker's threshold) is dropped."""
    keys = list(zip(kdf["cam"].astype(str), kdf["frame"].astype(int), kdf["global_player_id"].astype(int)))
    nid = np.array([mapping.get(k, -1) for k in keys], int)
    keep = nid >= 0
    out = kdf.loc[keep].copy()
    out["global_player_id"] = nid[keep]
    return out, int((~keep).sum())


def relabel_pose_cache(blob: dict, mapping: dict, cam: str):
    """``(blob with each frame's ids remapped, records dropped)``; an id without a mapping is
    dropped, and where two old ids land on one new id the first record stays."""
    frames_out: dict = {}
    dropped = 0
    for f, recs in blob.get("frames", {}).items():
        new: dict = {}
        for pid, rec in recs.items():
            nid = mapping.get((cam, int(f), int(pid)))
            if nid is None or nid in new:
                dropped += 1
                continue
            new[nid] = rec
        if new:
            frames_out[f] = new
    out = dict(blob)
    out["frames"] = frames_out
    return out, dropped


def assert_numpy1_for_pickles():
    if int(np.__version__.split(".")[0]) >= 2:
        raise SetupError("pose caches are numpy-1 pickles; relabel them under C:\\venvs\\smplx312 (numpy 1), "
                         f"not numpy {np.__version__}")
