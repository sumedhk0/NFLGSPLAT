"""Split a per-camera track where its kit changes for good: one track, two players.

WHY. The ground linker hands a track from one player to another when they
cross (purity p10 0.42-0.56 on the helmet set), and nothing downstream
recovers: identity votes one number for both, the pairing takes one
partner for both, the render draws one avatar that changes team mid-play
-- the "players get switched around" the user sees. The kit margin per
detection (tracking.kits, 2.3 % wrong at |margin| >= 0.4) is the one cue
that flips when the player does: play 1 v22 has 10 sideline tracks of 74
carrying a sustained run of the opposite kit, 20-135 confident detections
long.

WHAT. Per (cam, track), the confident kit signs in frame order, smoothed
by a running majority over ``window`` confident detections; a change of
the smoothed sign that holds for at least ``min_run`` confident detections
cuts the track there (the first frame of the new run), and the tail gets a
fresh id. A blip shorter than ``min_run`` is a shadow or a crossing, not a
handover, and stays. Kit labels are only split points: a track whose kit
never settles is left alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KIT_MARGIN: float = 0.4
WINDOW: int = 9
MIN_RUN: int = 15


def cut_points(margins: np.ndarray, *, margin: float = KIT_MARGIN, window: int = WINDOW,
               min_run: int = MIN_RUN) -> list[int]:
    """Indices (into ``margins``, frame order) where a new player starts.
    ``margins``: the track's kit margins per detection, NaN where unknown."""
    m = np.asarray(margins, float)
    conf = np.flatnonzero(np.isfinite(m) & (np.abs(m) >= margin))
    if len(conf) < 2 * min_run:
        return []
    sign = np.sign(m[conf])
    half = window // 2
    smooth = np.array([np.sign(sign[max(0, i - half): i + half + 1].sum()) for i in range(len(sign))])
    smooth[smooth == 0] = 1
    cuts = []
    cur = smooth[0]
    i = 0
    while i < len(smooth):
        if smooth[i] != cur:
            j = i
            while j < len(smooth) and smooth[j] != cur:
                j += 1
            if j - i >= min_run:
                cuts.append(int(conf[i]))
                cur = smooth[i]
            i = j
        else:
            i += 1
    return cuts


def split_tracks_by_kit(df: pd.DataFrame, *, margin: float = KIT_MARGIN, window: int = WINDOW,
                        min_run: int = MIN_RUN):
    """``(df, cuts)``: ``df`` with ``track_id`` (and ``global_player_id`` when
    present) reassigned after each cut; ``cuts`` a list of (cam, old id, frame,
    new id)."""
    out = df.copy()
    next_id = int(out["track_id"].max()) + 1
    cuts = []
    for (cam, tid), g in df[df["track_id"] >= 0].groupby(["cam", "track_id"]):
        g = g.sort_values("frame")
        idx = cut_points(g["kit_margin"].to_numpy(float), margin=margin, window=window, min_run=min_run)
        if not idx:
            continue
        rows = g.index.to_numpy()
        frames = g["frame"].to_numpy()
        for k, i in enumerate(idx):
            j = idx[k + 1] if k + 1 < len(idx) else len(rows)
            sel = rows[i:j]
            out.loc[sel, "track_id"] = next_id
            if "global_player_id" in out.columns:
                out.loc[sel, "global_player_id"] = next_id
            cuts.append((str(cam), int(tid), int(frames[i]), next_id))
            next_id += 1
    return out, cuts
