"""Align a clip to its tracking, and know when the alignment is untrustworthy.

The clips are a window inside a longer tracking record and are not cut at a
fixed offset from the snap, so the offset has to be recovered per play.

WHY NOT A STATIC CAMERA FIT. The obvious approach -- pair the labelled helmets
with their tracked positions and fit a camera -- does not discriminate. Helmets
vary about 0.3 m in height, which at this geometry is roughly 8 px of
irreducible reprojection error, the same size as the signal. And giving every
helmet one nominal height makes the 3D points COPLANAR, so a full projection
matrix is degenerate: measured, it returned an identical residual for offsets
seven seconds apart. Coplanar points admit a homography and nothing more.

WHAT WORKS. The snap is a sharp motion onset in both signals -- helmets start
moving in the image, players start moving on the field. Correlating the two
speed profiles needs no camera, no height assumption and no intrinsics. The
profiles are normalised before comparison because px/frame and m/s are not
comparable magnitudes.

THE ACCEPTANCE TEST is that the two views are SYNCHRONISED, so an offset
recovered from one must match the other. That is an independent check rather
than a threshold: on play 57583/82 the views agree to 0.00 s at correlations of
0.71 and 0.60, while on 57584/336 they disagree by 3.15 s -- and the second is
exactly the case that must not be silently used as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

VIDEO_FPS: float = 59.94
TRACKING_HZ: float = 10.0

# How far the two views' offsets may differ and still be believed. They are
# synchronised cameras, so any real disagreement is a failed recovery, not a
# property of the play. A quarter second is ~15 video frames, well beyond the
# 0.05 s grid the search runs on.
MAX_VIEW_DISAGREEMENT_S: float = 0.25

# Below this, the profile match is not distinctive enough to act on.
MIN_CORRELATION: float = 0.35


@dataclass(frozen=True)
class Alignment:
    offset_s: float | None
    per_view: dict            # view -> (offset, correlation)
    reason: str

    @property
    def ok(self) -> bool:
        return self.offset_s is not None


def image_speed(labels_for_video):
    """Per-frame helmet motion in px, with the CAMERA's motion removed.

    Raw helmet displacement is dominated by the camera: these angles pan and
    zoom to follow the play, so every helmet moves together whether or not any
    player does. Measured across the 60 plays, correlating raw displacement
    against field speed put the two synchronised views a median of 2.33 s apart
    -- the camera was the signal and the players were the noise.

    Subtracting the median displacement per frame removes the common component,
    which is the camera pan to first order, and leaves motion RELATIVE to the
    group -- which is what the tracking measures.
    """
    sub = labels_for_video.sort_values(["label", "frame"])
    cx = sub["left"] + sub["width"] / 2.0
    cy = sub["top"] + sub["height"] / 2.0
    sub = sub.assign(cx=cx, cy=cy)
    d = sub.groupby("label")[["cx", "cy"]].diff()
    sub = sub.assign(dx=d["cx"], dy=d["cy"])
    common = sub.groupby("frame")[["dx", "dy"]].transform("median")
    sub = sub.assign(step=np.hypot(sub["dx"] - common["dx"],
                                   sub["dy"] - common["dy"]))
    return sub.groupby("frame")["step"].median()


def field_speed(track):
    """``(times, speed)`` of the tracked players, m/s on the tracking clock."""
    v = np.linalg.norm(np.diff(track.xy, axis=0), axis=2) * TRACKING_HZ
    return track.times[1:], np.nanmedian(v, axis=1)


def _best_offset(speed_px, track, search=(-12.0, 4.0), step=0.05,
                 fps: float = VIDEO_FPS):
    t_track, v_track = field_speed(track)
    frames = speed_px.index.to_numpy()
    t_vid = frames / fps
    v_vid = speed_px.to_numpy()
    best = (None, -2.0)
    for off in np.arange(search[0], search[1], step):
        ref = np.interp(t_vid + off, t_track, v_track, left=np.nan, right=np.nan)
        ok = np.isfinite(ref) & np.isfinite(v_vid)
        if ok.sum() < 60:
            continue
        a = v_vid[ok] - v_vid[ok].mean()
        b = ref[ok] - ref[ok].mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            continue
        r = float(a @ b / denom)
        if r > best[1]:
            best = (float(off), r)
    return best


def align_play(labels, track, game_key: int, play_id: int, *,
               max_disagreement_s: float = MAX_VIEW_DISAGREEMENT_S,
               min_correlation: float = MIN_CORRELATION) -> Alignment:
    """Recover one play's video-to-tracking offset, or refuse to.

    ``labels`` is the whole train_labels frame; ``track`` the PlayTracking.
    """
    per_view = {}
    for view in ("Sideline", "Endzone"):
        name = f"{int(game_key)}_{int(play_id):06d}_{view}.mp4"
        sub = labels[labels["video"] == name]
        if sub.empty:
            continue
        per_view[view] = _best_offset(image_speed(sub), track)

    if len(per_view) < 2:
        return Alignment(None, per_view, "only one view has labels")
    offs = [o for o, _r in per_view.values() if o is not None]
    cors = [r for _o, r in per_view.values()]
    if len(offs) < 2:
        return Alignment(None, per_view, "a view produced no offset")
    if min(cors) < min_correlation:
        return Alignment(None, per_view,
                         f"weak profile match (r={min(cors):.2f})")
    spread = abs(offs[0] - offs[1])
    if spread > max_disagreement_s:
        return Alignment(None, per_view,
                         f"views disagree by {spread:.2f} s")
    return Alignment(float(np.mean(offs)), per_view, "ok")


def tracking_time(video_frame: int, offset_s: float,
                  fps: float = VIDEO_FPS) -> float:
    """Seconds from the snap for a video frame, given the play's offset."""
    return offset_s + video_frame / fps
