"""Torso colour per detection, read from the video.

The measurement scripts (07j) and the team-by-colour stage (08f) share
it: the middle band of each person box, away from helmet and turf, its
dominant jersey colour in HSV. NaN where the crop is too small or the
frame cannot be read.

THE BAND IS WRONG FOR A BODY IN A STANCE (2026-09-11). A lineman crouched
over the ball is as wide as he is tall, and the 25-60 % band of his box
lands on his white pants and backside, not his jersey: on play 1's endzone
camera that band reads saturation 85-99 for Kansas City's crouched linemen
against 80-90 for Baltimore's white kit -- the teams overlap, and the
linemen are called white. They lost their team label, and the kit clash
that follows vetoed their cross-camera pairs, which is a third of why the
two-view coverage is what it is. With the torso taken from the POSE
instead (the quadrilateral of shoulders and hips, shrunk toward its centre,
away from pads and belt) the same players read 112-215 against 26-72: a
clean gap. The keypoints are per camera and already computed (05m), so
``torso_polygon`` is used wherever they exist and the band is the fallback.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.identity.team_color import dominant_jersey_color

TORSO_TOP: float = 0.25
TORSO_BOTTOM: float = 0.60
COCO_SHOULDERS: tuple[int, int] = (5, 6)
COCO_HIPS: tuple[int, int] = (11, 12)
POLY_SHRINK: float = 0.7          # toward the centre: the pads and the belt are not the jersey
MIN_POLY_PX: int = 60             # a torso smaller than this is not measured


def torso_polygon(joints: dict, *, shrink: float = POLY_SHRINK):
    """``[4, 2]`` the shoulders-hips ring shrunk toward its centre, or None when a corner is
    missing. ``joints``: ``{COCO joint: (x, y)}`` for one person in one frame."""
    pts = [joints.get(j) for j in (COCO_SHOULDERS[0], COCO_SHOULDERS[1], COCO_HIPS[1], COCO_HIPS[0])]
    if any(p is None for p in pts):
        return None
    q = np.asarray(pts, float)
    if not np.isfinite(q).all():
        return None
    c = q.mean(axis=0)
    return c + shrink * (q - c)


def polygon_colour(img, poly, *, min_px: int = MIN_POLY_PX) -> np.ndarray:
    """Mean HSV inside ``poly`` in ``img`` (BGR), NaN when the torso is smaller than ``min_px``."""
    import cv2

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(poly, np.int32), 1)
    m = mask.astype(bool)
    if int(m.sum()) < min_px:
        return np.full(3, np.nan)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[m].astype(np.float64).mean(axis=0)


def detection_colours(df_view, video_path, *, max_per_frame: int = 64, joints_by=None) -> np.ndarray:
    """``[N, 3]`` HSV torso colour per detection row of ``df_view`` (one
    camera's rows of tracks.parquet), NaN where unknown.

    ``joints_by`` ``{(frame, track_id): {COCO joint: (x, y)}}`` -- that camera's own keypoints --
    takes the torso from the POSE wherever it can; the band is the fallback, and the band is wrong
    for a body in a stance (see the module docstring)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = df_view["frame"].to_numpy()
    ids = df_view["track_id"].to_numpy() if "track_id" in df_view else np.full(len(df_view), -1)
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
            if joints_by is not None:
                j = joints_by.get((int(frames[r]), int(ids[r])))
                poly = torso_polygon(j) if j else None
                if poly is not None:
                    c = polygon_colour(img, poly)
                    if np.isfinite(c).all():
                        colours[r] = c
                        continue
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
