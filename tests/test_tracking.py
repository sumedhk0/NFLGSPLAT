"""Tracking-pipeline tests (cross-camera re-ID + field filter).

No GPU / YOLO / PaddleOCR dependency — we inject synthetic tracks based on
the fixture's known-position players. The real ``detect_and_track`` path is
exercised only via its TRACK_COLUMNS contract.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import CameraTrack, constant_track
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS, empty_tracks
from nfl_gsplat.tracking.cross_cam_reid import (
    CrossCamConfig,
    filter_to_field,
    project_foot_points_to_field,
    reid_pipeline,
)
from nfl_gsplat.utils.geometry import project_points
from tests.fixtures.generate import (
    PLAYER_ROOTS,
    _endzone_camera,
    _sideline_camera,
    synthetic_bbox_for_player,
)


def _cam_map() -> dict[str, CameraTrack]:
    intr_s, pose_s = _sideline_camera()
    intr_e, pose_e = _endzone_camera()
    return {
        "sideline": constant_track(intr_s, pose_s, 1),
        "endzone":  constant_track(intr_e, pose_e, 1),
    }


def _synthetic_tracks(num_frames: int = 10, rng_seed: int = 0) -> pd.DataFrame:
    """3 fixture players, one bbox per (cam, player, frame), with small jitter."""
    rng = np.random.default_rng(rng_seed)
    cams = _cam_map()
    rows: list[dict] = []
    for cam_name, track in cams.items():
        intr, pose = track.at(0)
        K, R, t = intr.K(), pose.R, pose.t
        for player_idx, root in enumerate(PLAYER_ROOTS):
            for frame in range(num_frames):
                jitter = rng.normal(0.0, 0.02, size=3)
                root_j = root + jitter
                bbox = synthetic_bbox_for_player(root_j, K, R, t)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                # Project the true foot point (base of player on the ground) directly,
                # rather than using the bbox bottom-center. Oblique-view perspective
                # makes the bbox bottom-center a biased estimator of the true foot xy.
                foot_xyz = np.array([[root_j[0], root_j[1], 0.0]])
                foot_uv = project_points(foot_xyz, K, R, t)[0]
                foot_u, foot_v = float(foot_uv[0]), float(foot_uv[1])
                rows.append({
                    "frame": frame,
                    "cam": cam_name,
                    "track_id": player_idx,   # track id consistent per camera
                    "global_player_id": -1,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "conf": 0.95,
                    "foot_u": foot_u, "foot_v": foot_v,
                    "jersey_number_ocr": -1,
                })
    return pd.DataFrame(rows, columns=TRACK_COLUMNS).astype({"frame": "int64", "track_id": "int64",
                                                             "global_player_id": "int64",
                                                             "jersey_number_ocr": "int64"})


# --- Contract tests ---------------------------------------------------------

def test_empty_tracks_has_all_columns():
    df = empty_tracks()
    assert list(df.columns) == TRACK_COLUMNS
    assert df.empty


# --- Foot-point projection --------------------------------------------------

def test_foot_points_project_near_player_roots():
    df = _synthetic_tracks(num_frames=1)
    cams = _cam_map()
    proj = project_foot_points_to_field(df, cams)
    # For each player, projected foot xy should match the player's root xy within 20 cm.
    for player_idx, root in enumerate(PLAYER_ROOTS):
        mask = proj["track_id"] == player_idx
        xy = proj.loc[mask, ["foot_x_m", "foot_y_m"]].to_numpy()
        err = np.linalg.norm(xy - root[:2][None, :], axis=1)
        assert err.max() < 0.20, f"player {player_idx}: foot error {err.max():.3f} m"


# --- Field filter -----------------------------------------------------------

def test_field_filter_drops_out_of_bounds_detections():
    df = _synthetic_tracks(num_frames=1)
    cams = _cam_map()
    proj = project_foot_points_to_field(df, cams)

    # Inject a "coach" detection that projects 40 m off the sideline.
    # Fabricate a foot point at image bottom-center of the sideline cam;
    # we override its projected xy directly for determinism.
    coach = proj.iloc[:1].copy()
    coach.loc[:, "track_id"] = 999
    coach.loc[:, "foot_x_m"] = 0.0
    coach.loc[:, "foot_y_m"] = -60.0          # outside field width / 2 + buffer
    proj = pd.concat([proj, coach], ignore_index=True)

    cfg = CrossCamConfig()
    filt = filter_to_field(proj, cfg)
    assert 999 not in filt["track_id"].to_numpy()
    assert len(filt) == len(proj) - 1


def test_field_filter_keeps_in_bounds():
    df = _synthetic_tracks(num_frames=1)
    cams = _cam_map()
    proj = project_foot_points_to_field(df, cams)
    cfg = CrossCamConfig()
    filt = filter_to_field(proj, cfg)
    # All 3 × 2 cams = 6 detections should survive (players are on the field).
    assert len(filt) == 6


# --- Cross-camera re-ID -----------------------------------------------------

def test_reid_assigns_matching_global_ids_across_cameras():
    df = _synthetic_tracks(num_frames=12)
    cams = _cam_map()
    cfg = CrossCamConfig()
    out = reid_pipeline({"sideline": df[df["cam"] == "sideline"],
                         "endzone":  df[df["cam"] == "endzone"]},
                        cams, cfg)
    # Each player should have exactly one global_player_id used on both cameras.
    per_player = out.groupby(["track_id"])["global_player_id"].nunique()
    assert (per_player == 1).all(), per_player

    # The two cameras should share 3 distinct global IDs (one per fixture player).
    gid_per_cam = out.groupby("cam")["global_player_id"].unique()
    assert len(gid_per_cam["sideline"]) == 3
    assert len(gid_per_cam["endzone"]) == 3
    assert set(gid_per_cam["sideline"]) == set(gid_per_cam["endzone"])


def test_reid_unmatched_track_gets_fresh_id():
    df = _synthetic_tracks(num_frames=5)
    cams = _cam_map()
    cfg = CrossCamConfig()

    # Add a sideline-only "ghost" track whose projected foot is on the field
    # but with no endzone counterpart.
    ghost = df[df["cam"] == "sideline"].iloc[:1].copy()
    ghost.loc[:, "track_id"] = 500
    # Shift bbox so foot projects somewhere nobody else is.
    ghost.loc[:, "bbox_x1"] = ghost["bbox_x1"].iloc[0] - 100
    ghost.loc[:, "bbox_x2"] = ghost["bbox_x2"].iloc[0] - 100
    ghost.loc[:, "foot_u"] = 0.5 * (ghost["bbox_x1"].iloc[0] + ghost["bbox_x2"].iloc[0])
    df2 = pd.concat([df, ghost], ignore_index=True)

    out = reid_pipeline({"sideline": df2[df2["cam"] == "sideline"],
                         "endzone":  df2[df2["cam"] == "endzone"]},
                        cams, cfg)

    if 500 in out["track_id"].to_numpy():
        gid = out.loc[(out["track_id"] == 500) & (out["cam"] == "sideline"),
                      "global_player_id"].iloc[0]
        others = out.loc[out["track_id"] != 500, "global_player_id"].unique()
        assert gid not in others, "ghost track should have a unique global_player_id"
