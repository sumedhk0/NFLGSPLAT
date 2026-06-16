"""Cross-camera re-identification via field-plane foot-point matching.

Given per-camera track DataFrames (schema in :mod:`detect_track`), we:

1. Project each detection's ``(foot_u, foot_v)`` to the Z=0 world plane using
   the camera's calibrated pose. Drops detections whose projected foot falls
   outside the playing field + buffer (coaches, refs, sideline staff).
2. Match tracks across cameras using Hungarian assignment on median
   field-plane positions over a sliding window; assign ``global_player_id``.
3. Tracks that match consistently get the same ``global_player_id``;
   unmatched tracks get unique fresh IDs.

This module is CPU-only. It depends on scipy + numpy + pandas. It is safe
to import from any conda env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
from nfl_gsplat.utils.geometry import (
    CameraIntrinsics,
    CameraPose,
    point_in_field_rect,
    project_to_plane_z0,
)


@dataclass(frozen=True)
class CrossCamConfig:
    field_length_m: float = 109.728
    field_width_m: float = 48.768
    field_buffer_m: float = 1.0
    match_threshold_m: float = 2.5
    window_frames: int = 30     # sliding window for median position


def project_foot_points_to_field(
    df: pd.DataFrame,
    cameras: Mapping[str, tuple[CameraIntrinsics, CameraPose]],
) -> pd.DataFrame:
    """Add ``foot_x_m, foot_y_m`` columns by projecting foot points to Z=0.

    Any detection from a camera not in ``cameras`` gets NaN.
    """
    out = df.copy()
    out["foot_x_m"] = np.nan
    out["foot_y_m"] = np.nan
    for cam_name, (intr, pose) in cameras.items():
        mask = out["cam"].values == cam_name
        if not mask.any():
            continue
        uv = out.loc[mask, ["foot_u", "foot_v"]].to_numpy()
        K = intr.K()
        xy = np.full((uv.shape[0], 2), np.nan, dtype=np.float64)
        for i, puv in enumerate(uv):
            xy[i] = project_to_plane_z0(puv, K, pose.R, pose.t)
        out.loc[mask, "foot_x_m"] = xy[:, 0]
        out.loc[mask, "foot_y_m"] = xy[:, 1]
    return out


def filter_to_field(df: pd.DataFrame, cfg: CrossCamConfig) -> pd.DataFrame:
    """Drop detections whose projected foot is outside the playing field
    (plus a small buffer). Rejects NaN projections too."""
    keep = np.zeros(len(df), dtype=bool)
    xy = df[["foot_x_m", "foot_y_m"]].to_numpy()
    for i, (x, y) in enumerate(xy):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        keep[i] = point_in_field_rect(
            np.array([x, y]),
            length_m=cfg.field_length_m,
            width_m=cfg.field_width_m,
            buffer_m=cfg.field_buffer_m,
        )
    return df.loc[keep].reset_index(drop=True)


def _median_positions_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """For each (cam, track_id), median (x, y) across all frames, plus count."""
    grouped = df.groupby(["cam", "track_id"], as_index=False)
    med = grouped[["foot_x_m", "foot_y_m"]].median()
    cnt = grouped.size().rename(columns={"size": "n"})
    return med.merge(cnt, on=["cam", "track_id"])


def assign_global_ids(df: pd.DataFrame, cfg: CrossCamConfig) -> pd.DataFrame:
    """Match tracks across cameras on median foot-position distance.

    For a two-camera rig (sideline, endzone): build a cost matrix of pairwise
    distances between per-track medians in camera A and camera B; run Hungarian
    on it; accept matches whose distance is below ``cfg.match_threshold_m``.

    Unmatched tracks (or tracks from a camera not paired this pass) receive
    unique fresh global_player_ids.
    """
    if df.empty:
        return df.copy()

    cams = sorted(df["cam"].unique())
    out = df.copy()
    medians = _median_positions_per_track(df)

    # Group per-cam median tables.
    per_cam: dict[str, pd.DataFrame] = {c: medians[medians["cam"] == c].reset_index(drop=True)
                                        for c in cams}

    # Assign fresh IDs first; then merge matched ones.
    gid_counter = 0
    cam_track_to_gid: dict[tuple[str, int], int] = {}
    for cam, tbl in per_cam.items():
        for tid in tbl["track_id"].to_list():
            cam_track_to_gid[(cam, int(tid))] = gid_counter
            gid_counter += 1

    # Pairwise match between the first two cameras only (our rig).
    if len(cams) >= 2:
        a, b = cams[0], cams[1]
        A = per_cam[a]
        B = per_cam[b]
        if len(A) > 0 and len(B) > 0:
            xa = A[["foot_x_m", "foot_y_m"]].to_numpy()
            xb = B[["foot_x_m", "foot_y_m"]].to_numpy()
            cost = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] <= cfg.match_threshold_m:
                    tid_a = int(A.iloc[ri]["track_id"])
                    tid_b = int(B.iloc[ci]["track_id"])
                    # Merge: force both keys to the lower existing gid.
                    gid_a = cam_track_to_gid[(a, tid_a)]
                    gid_b = cam_track_to_gid[(b, tid_b)]
                    merged = min(gid_a, gid_b)
                    cam_track_to_gid[(a, tid_a)] = merged
                    cam_track_to_gid[(b, tid_b)] = merged

    # Re-number global IDs to be dense starting from 0.
    unique_gids = sorted(set(cam_track_to_gid.values()))
    renum = {g: i for i, g in enumerate(unique_gids)}

    gids = [
        renum[cam_track_to_gid[(c, int(tid))]]
        for c, tid in zip(out["cam"].to_list(), out["track_id"].to_list())
    ]
    out = out.copy()
    out["global_player_id"] = np.asarray(gids, dtype=np.int64)
    return out[TRACK_COLUMNS + [c for c in out.columns if c not in TRACK_COLUMNS]]


def reid_pipeline(
    tracks_by_cam: Mapping[str, pd.DataFrame],
    cameras: Mapping[str, tuple[CameraIntrinsics, CameraPose]],
    cfg: CrossCamConfig,
) -> pd.DataFrame:
    """Full cross-camera re-ID: concatenate per-cam tracks → project foot
    points → field filter → assign global IDs."""
    if not tracks_by_cam:
        raise ValueError("tracks_by_cam is empty")
    df = pd.concat(list(tracks_by_cam.values()), ignore_index=True)
    df = project_foot_points_to_field(df, cameras)
    df = filter_to_field(df, cfg)
    df = assign_global_ids(df, cfg)
    return df


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    from pathlib import Path

    import typer

    from nfl_gsplat.calibration.cameras_io import load_cameras
    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import PlayDir

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(play_dir: Path = typer.Option(..., "--play-dir"),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pdir = PlayDir.from_dir(play_dir)
        cameras = load_cameras(pdir.cameras_json)
        df = pd.read_parquet(pdir.tracks)
        tracks_by_cam = {cam: g for cam, g in df.groupby("cam")}
        ccfg = CrossCamConfig(field_buffer_m=float(cfg.tracking.field_buffer_m))
        out = reid_pipeline(tracks_by_cam, cameras, ccfg)
        out.to_parquet(pdir.tracks, index=False)

    app()


if __name__ == "__main__":
    _main()
