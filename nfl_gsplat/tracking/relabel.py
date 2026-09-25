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


def backup_path(f, suffix: str):
    """``f`` + suffix, or + suffix.N for the first N not taken. The relabelling stages (08t, 08v)
    are applied more than once -- the fragments of one 08t pass carry switches of their own, play 1
    needed four passes -- and a fixed backup name would overwrite the only copy of the tables before
    ANY cut on the second apply (which it did, 2026-09-15; a manual snapshot saved it)."""
    from pathlib import Path

    f = Path(f)
    b = f.with_name(f.name + suffix)
    n = 1
    while b.exists():
        b = f.with_name(f"{f.name}{suffix}.{n}")
        n += 1
    return b


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


KEYPOINT_TABLES: tuple = ("keypoints_2d.parquet", "keypoints_2d_ft2.parquet")


def relabel_keypoint_tables(play_dir, mapping: dict, suffix: str, *, names: tuple = KEYPOINT_TABLES) -> list:
    """Apply ``mapping`` ({(cam, frame, old id): new id}, fold.keypoint_map) to every keypoint table of ``play_dir``
    named in ``names`` that exists, each backed up first (backup_path(table, ``suffix``)). Returns
    ``[(name, rows written, rows dropped, rows relabelled, backup name)]``. The per-play fine-tuned detector's table
    (keypoints_2d_ft2.parquet) feeds the fits since play 1 v106; until 2026-09-25 08z / 08za relabelled only
    keypoints_2d.parquet and every port patched the ft2 table by hand."""
    import shutil
    from pathlib import Path

    out = []
    for name in names:
        p = Path(play_dir) / name
        if not p.exists():
            continue
        kdf = pd.read_parquet(p)
        keys = zip(kdf["cam"].astype(str), kdf["frame"].astype(int), kdf["global_player_id"].astype(int))
        changed = sum(1 for k in keys if mapping.get(k, -1) not in (-1, k[2]))
        kout, kdrop = relabel_keypoints(kdf, mapping)
        b = backup_path(p, suffix)
        shutil.copy2(p, b)
        kout.to_parquet(p, index=False)
        out.append((name, len(kout), kdrop, changed, b.name))
    return out


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
