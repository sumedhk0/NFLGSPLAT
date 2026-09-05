"""Torso colour per detection, read from the video.

The measurement scripts (07j) and the team-by-colour stage (08f) share
it: the middle band of each person box, away from helmet and turf, its
dominant jersey colour in HSV. NaN where the crop is too small or the
frame cannot be read.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.identity.team_color import dominant_jersey_color

TORSO_TOP: float = 0.25
TORSO_BOTTOM: float = 0.60


def detection_colours(df_view, video_path, *, max_per_frame: int = 64) -> np.ndarray:
    """``[N, 3]`` HSV torso colour per detection row of ``df_view`` (one
    camera's rows of tracks.parquet), NaN where unknown."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = df_view["frame"].to_numpy()
    boxes = df_view[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    colours = np.full((len(df_view), 3), np.nan)
    for f in np.unique(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        rows = np.flatnonzero(frames == f)[:max_per_frame]
        h, w = img.shape[:2]
        for r in rows:
            x1, y1, x2, y2 = boxes[r]
            bh = y2 - y1
            ya, yb = int(max(0, y1 + TORSO_TOP * bh)), int(min(h, y1 + TORSO_BOTTOM * bh))
            xa, xb = int(max(0, x1)), int(min(w, x2))
            if yb - ya < 3 or xb - xa < 3:
                continue
            try:
                colours[r] = dominant_jersey_color(img[ya:yb, xa:xb])
            except Exception:                              # noqa: BLE001
                continue
    cap.release()
    return colours


def team_votes(df, videos, *, max_per_frame: int = 64):
    """``({(cam, track_id): label}, {cam: (S_lo, S_hi)})`` by rule D over
    every detection row of ``df`` (tracks.parquet, both cameras) -- see
    ``team_color.split_by_saturation_votes``. Label 1 is the coloured kit.
    Reads each camera's video once (about a minute per camera)."""
    from nfl_gsplat.identity.team_color import split_by_saturation_votes, votes_from_margins

    if "kit_margin" in df.columns and np.isfinite(df["kit_margin"].to_numpy(float)).mean() > 0.5:
        # tracks.parquet from 08b carries each box's signed saturation margin
        # (tracking.kits): the same evidence, no second pass over the videos.
        keys = [(str(c), int(t)) for c, t in zip(df["cam"], df["track_id"])]
        return votes_from_margins(keys, df["kit_margin"].to_numpy(float)), {}
    sats = {}
    for cam, dv in df.groupby("cam"):
        cam = str(cam)
        if cam not in videos:
            raise KeyError(f"team_votes: no video for camera {cam!r}")
        cols = detection_colours(dv, videos[cam], max_per_frame=max_per_frame)
        keys = [(cam, int(t)) for t in dv["track_id"].to_numpy()]
        sats[cam] = (keys, cols[:, 1])
    return split_by_saturation_votes(sats)
