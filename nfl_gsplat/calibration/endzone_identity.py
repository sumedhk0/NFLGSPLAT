"""Cross-camera endzone correspondences from jersey IDENTITY (player_uid).

The sideline camera (calibrated) turns each player's foot pixel into a field
point (X, Y, 0); the endzone camera sees the same player's foot pixel. Pairing
is by player_uid (from the jersey/team identity stack) -- not geometry -- so the
correspondences are correct regardless of the unknown endzone camera. Native
endzone pixels (no rotation)."""
from __future__ import annotations

import numpy as np

from nfl_gsplat.identity.registry import OTHER_UID, REFEREE_UID


def _is_player(uid) -> bool:
    return isinstance(uid, str) and uid not in (OTHER_UID, REFEREE_UID)


def field_positions_by_uid(tracks_df, sideline_track, *, cam="sideline", smooth_window=15):
    """{player_uid: {frame: (X, Y)}} from projecting the sideline cam's feet to
    Z=0, temporally smoothed per uid with a centered rolling median."""
    from nfl_gsplat.tracking.cross_cam_reid import project_foot_points_to_field
    sl = tracks_df[tracks_df["cam"] == cam]
    proj = project_foot_points_to_field(sl, {cam: sideline_track})
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for uid, grp in proj.groupby("player_uid"):
        if not _is_player(uid):
            continue
        g = grp[np.isfinite(grp[["foot_x_m", "foot_y_m"]].to_numpy(float)).all(axis=1)]
        if not len(g):
            continue
        # One position per frame FIRST: a uid can have two rows in one frame
        # when two tracks resolve to the same jersey (fragmentation / OCR
        # collision); collapse them to the per-frame median so the frame index
        # is unique (else the reindex below raises) and the collision resolves
        # deterministically instead of arbitrary last-wins.
        per_frame = g.groupby("frame")[["foot_x_m", "foot_y_m"]].median().sort_index()
        fr_idx = per_frame.index.to_numpy().astype(int)
        # Frame-number-aware rolling window: reindex onto the dense integer
        # frame range (missing frames = NaN) BEFORE rolling, so a 15-frame
        # window is measured in FRAMES, not row position. Without this, a
        # track with an occlusion gap (e.g. frame 100 then frame 340) pools
        # temporally-distant frames into the same window, biasing the
        # smoothed field point.
        full_idx = np.arange(int(fr_idx.min()), int(fr_idx.max()) + 1)
        w = max(1, int(smooth_window))
        sx = per_frame["foot_x_m"].reindex(full_idx).rolling(
            w, center=True, min_periods=1).median()
        sy = per_frame["foot_y_m"].reindex(full_idx).rolling(
            w, center=True, min_periods=1).median()
        out[str(uid)] = {int(f): (float(sx.loc[f]), float(sy.loc[f])) for f in fr_idx}
    return out


def endzone_pixels_by_uid(tracks_df, *, cam="endzone"):
    """{player_uid: {frame: (u, v)}} of native endzone foot pixels."""
    ez = tracks_df[tracks_df["cam"] == cam]
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for uid, grp in ez.groupby("player_uid"):
        if not _is_player(uid):
            continue
        # Per-frame median so a uid with two tracks in one frame (jersey
        # collision / fragmentation) resolves deterministically, not last-wins.
        per_frame = grp.groupby("frame")[["foot_u", "foot_v"]].median()
        out[str(uid)] = {int(fr): (float(u), float(v))
                         for fr, (u, v) in per_frame.iterrows()}
    return out


def identity_correspondences(tracks_df, sideline_track, *, sideline_cam="sideline",
                             endzone_cam="endzone", smooth_window=15):
    """{frame: (world (N,3), uv (N,2))} pairing sideline field points to endzone
    foot pixels by shared player_uid, per co-observed frame."""
    field = field_positions_by_uid(tracks_df, sideline_track, cam=sideline_cam,
                                    smooth_window=smooth_window)
    pix = endzone_pixels_by_uid(tracks_df, cam=endzone_cam)
    per_frame: dict[int, tuple[list, list]] = {}
    for uid, fmap in field.items():
        emap = pix.get(uid)
        if not emap:
            continue
        for fr, (x, y) in fmap.items():
            if fr not in emap:
                continue
            w_list, u_list = per_frame.setdefault(int(fr), ([], []))
            w_list.append([x, y, 0.0]); u_list.append(list(emap[fr]))
    return {fr: (np.asarray(w, np.float64), np.asarray(u, np.float64))
            for fr, (w, u) in per_frame.items()}
