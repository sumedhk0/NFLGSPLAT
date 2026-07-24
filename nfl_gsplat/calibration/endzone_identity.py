"""Cross-camera endzone correspondences from jersey IDENTITY (player_uid).

The sideline camera (calibrated) turns each player's foot pixel into a field
point (X, Y, 0); the endzone camera sees the same player's foot pixel. Pairing
is by player_uid (from the jersey/team identity stack) -- not geometry -- so the
correspondences are correct regardless of the unknown endzone camera. Native
endzone pixels (no rotation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import CameraTrack  # noqa: F401 (type)
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
        g = grp.sort_values("frame")
        xy = g[["foot_x_m", "foot_y_m"]].to_numpy(float)
        ok = np.isfinite(xy).all(axis=1)
        g = g[ok]; xy = xy[ok]
        if not len(g):
            continue
        # Frame-number-aware rolling window: reindex onto the dense integer
        # frame range (missing frames = NaN) BEFORE rolling, so a 15-row
        # window is measured in FRAMES, not row position. Without this, a
        # track with an occlusion gap (e.g. frame 100 then frame 340) pools
        # temporally-distant frames into the same window, biasing the
        # smoothed field point.
        fr_arr = g["frame"].to_numpy().astype(int)
        lo, hi = int(fr_arr.min()), int(fr_arr.max())
        full_idx = np.arange(lo, hi + 1)
        sx = pd.Series(xy[:, 0], index=fr_arr).reindex(full_idx)
        sy = pd.Series(xy[:, 1], index=fr_arr).reindex(full_idx)
        w = max(1, int(smooth_window))
        sx = sx.rolling(w, center=True, min_periods=1).median()
        sy = sy.rolling(w, center=True, min_periods=1).median()
        out[str(uid)] = {int(f): (float(sx.loc[f]), float(sy.loc[f])) for f in fr_arr}
    return out


def endzone_pixels_by_uid(tracks_df, *, cam="endzone"):
    """{player_uid: {frame: (u, v)}} of native endzone foot pixels."""
    ez = tracks_df[tracks_df["cam"] == cam]
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for uid, grp in ez.groupby("player_uid"):
        if not _is_player(uid):
            continue
        out[str(uid)] = {int(fr): (float(u), float(v))
                         for fr, u, v in zip(grp["frame"], grp["foot_u"], grp["foot_v"])}
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
