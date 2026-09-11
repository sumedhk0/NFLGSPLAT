"""A virtual camera that follows the play.

WHY. A fixed camera on the play's median position keeps the formation in
frame and lets the play run out of it; a broadcast camera pans with the
action. Following the players' centroid, smoothed over a second and a
half so detector churn and a body appearing at the edge do not jerk the
view, reads as a camera operator.

WHAT. Per frame, the mean ground position of the drawn bodies; a Gaussian
over frames (``sigma_frames``) smooths it; the target sits 1 m above it
and the eye at a fixed offset, so the view direction never changes and
only the camera translates (a dolly, not a pan: bodies keep their
projected size).
"""
from __future__ import annotations

import numpy as np

SIGMA_FRAMES: float = 45.0            # ~0.75 s at 60 fps
EYE_OFFSET_M = (2.0, -34.0, 13.0)     # behind the near sideline, above it
TARGET_HEIGHT_M: float = 1.0


def smooth_track(frames, xy_by_frame, *, sigma_frames: float = SIGMA_FRAMES) -> dict:
    """``{frame: xy}`` Gaussian-smoothed over the frame index; frames with no
    position borrow from their neighbours through the same kernel."""
    frames = [int(f) for f in frames]
    if not frames:
        return {}
    fs = np.asarray(frames, float)
    pts = np.array([xy_by_frame.get(f, (np.nan, np.nan)) for f in frames], float)
    have = np.isfinite(pts).all(1)
    if not have.any():
        raise ValueError("no frame has a position to follow")
    out = {}
    for i, f in enumerate(frames):
        w = np.exp(-0.5 * ((fs - fs[i]) / max(sigma_frames, 1e-6)) ** 2) * have
        out[f] = (w[:, None] * np.nan_to_num(pts)).sum(0) / w.sum()
    return out


def follow_path(frames, xy_by_frame, *, sigma_frames: float = SIGMA_FRAMES,
                eye_offset=EYE_OFFSET_M, target_height: float = TARGET_HEIGHT_M) -> dict:
    """``{frame: (eye, target)}``, both ``[3]`` metres."""
    path = smooth_track(frames, xy_by_frame, sigma_frames=sigma_frames)
    off = np.asarray(eye_offset, float)
    out = {}
    for f, xy in path.items():
        target = np.array([xy[0], xy[1], target_height])
        out[f] = (target + off, target)
    return out
