"""Align a clip to its tracking, and know when the alignment is untrustworthy.

The clips are a window inside a longer tracking record and are not cut at a
fixed offset from the snap, so the offset has to be recovered per play.

THE METHOD THAT WORKS is ``align_play_plane``. Helmets sit near one horizontal
plane, so at the correct offset a single homography carries the tracked
positions onto the labelled helmet boxes in EVERY frame; at a wrong offset the
players are in the wrong places and no homography fits any frame. Measured over
the 60 plays, the correct offset costs about 7 px of median reprojection --
which is the floor set by real helmet-height variation -- against 40 to 90 px
half a second away. That contrast, not the absolute residual, is the signal.

WHAT IT KEYS ON is players deforming the group NON-AFFINELY -- cutting and
accelerating independently. A homography absorbs an affine error, so a play in
which everyone drifts at constant velocity together is the weak case: measured
on a constant-velocity fixture, a full second of misalignment cost only 6 px,
which is inside the noise floor. Independent acceleration takes the same error
to 17 px, and real footage to 40-90 px. ``contrast`` is what reports this, and
is why it is checked rather than the residual alone.

A CORRECTION, because this file previously argued the opposite. An earlier note
here concluded that a geometric fit "does not discriminate", on two findings
that were each true and together misleading. Helmets do vary about 0.3 m in
height, which is roughly 8 px here -- but that is a noise FLOOR, not the size of
the signal, and the wrong-offset residual is six to ten times larger. And a full
projection matrix genuinely is degenerate on coplanar points, which is why it
returned an identical residual for offsets seven seconds apart -- but the fix
for that is to fit the homography those points do determine, not to abandon
geometry. The real mistake was scoring ONE frame, where 8 px of noise and the
effect being measured are comparable; over hundreds of frames they are not.

``align_play`` below is the older motion-correlation route: correlate helmet
motion in the image against player motion on the field, needing no camera and no
height assumption. It is kept because it depends on nothing geometric and so
fails differently, which makes it a useful second opinion. It aligned 12 of 60
plays where the plane fit aligns far more.

THE ACCEPTANCE TEST for both is that the two views are SYNCHRONISED, so an
offset recovered from one must match the other -- an independent check rather
than a tuned threshold.
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
# 0.04 s grid the search runs on.
MAX_VIEW_DISAGREEMENT_S: float = 0.25

# Below this, the motion-profile match is not distinctive enough to act on.
MIN_CORRELATION: float = 0.35

# A homography needs four points; eight makes it overdetermined enough that
# RANSAC can reject a mislabelled box instead of fitting it.
MIN_HELMETS_PER_FRAME: int = 8

# Inlier distance for the per-frame homography, in pixels. Comfortably above the
# ~8 px floor from helmet-height variation, well below a wrong-offset residual.
RANSAC_PX: float = 10.0

# How much better the best offset must be than everything more than
# CONTRAST_GAP_S away from it. At 3x, a play whose minimum is merely the
# shallowest point of a flat curve is refused rather than reported.
MIN_CONTRAST: float = 3.0
CONTRAST_GAP_S: float = 1.5


@dataclass(frozen=True)
class Alignment:
    offset_s: float | None
    per_view: dict     # view -> (offset, correlation) | (offset, residual, contrast)
    reason: str

    @property
    def ok(self) -> bool:
        return self.offset_s is not None


# --------------------------------------------------------------------------
# The plane fit: preferred.
# --------------------------------------------------------------------------

def helmet_boxes_by_frame(labels_for_video, player_index,
                          min_helmets: int = MIN_HELMETS_PER_FRAME):
    """``{frame: (centres[N,2], player_columns[N])}`` for one video.

    Frames with too few labelled players to overdetermine a homography are
    dropped here rather than guarded at every use.
    """
    out = {}
    for frame, grp in labels_for_video.groupby("frame"):
        sel = grp[grp["label"].isin(player_index)]
        if len(sel) < min_helmets:
            continue
        centres = np.column_stack([
            sel["left"].to_numpy(float) + sel["width"].to_numpy(float) / 2.0,
            sel["top"].to_numpy(float) + sel["height"].to_numpy(float) / 2.0])
        cols = np.array([player_index[p] for p in sel["label"]])
        out[int(frame)] = (centres, cols)
    return out


def plane_residual(byf, track, offset_s: float, frames, *,
                   fps: float = VIDEO_FPS, ransac_px: float = RANSAC_PX,
                   min_frames: int = 5, min_covered: float = 0.5) -> float:
    """Median per-frame cost of explaining the helmets by one plane homography.

    Frames landing outside the tracking record are SKIPPED, not clamped. A
    clamped time freezes every player at an endpoint, and a frozen configuration
    fits itself exactly -- so scoring clamped frames hands a perfect score to
    offsets that simply shove the clip off the end of the record. Left in, that
    is not a small bias: the search preferred -12.5 s to a true -5.25 s.

    ``inf`` when too little of the clip could be scored, so that an offset
    surviving on a handful of frames never beats one scored on all of them.
    """
    import cv2

    errs = []
    for f in frames:
        t = offset_s + f / fps
        if not track.covers(t):
            continue
        uv, cols = byf[f]
        world = track.at(t)[cols]
        ok = np.isfinite(world).all(axis=1)
        if ok.sum() < MIN_HELMETS_PER_FRAME:
            continue
        H, _ = cv2.findHomography(world[ok], uv[ok], cv2.RANSAC, ransac_px)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(
            world[ok].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        errs.append(float(np.median(np.linalg.norm(proj - uv[ok], axis=1))))
    enough = max(min_frames, int(min_covered * len(frames)))
    return float(np.median(errs)) if len(errs) >= enough else float("inf")


def best_offset_plane(byf, track, *, search=(-12.0, 4.0), coarse_step=0.4,
                      fine_step=0.04, stride: int = 10,
                      fps: float = VIDEO_FPS):
    """``(offset, residual, contrast)`` -- coarse sweep, then refine.

    ``contrast`` is how many times worse the best offset more than
    ``CONTRAST_GAP_S`` away is. It is the honest confidence measure: a residual
    of 7 px means nothing on its own if 7 px is also what a two-second error
    costs.
    """
    frames = sorted(byf)[::stride]
    if not frames:
        return None, float("inf"), 0.0
    coarse = sorted((plane_residual(byf, track, o, frames, fps=fps), float(o))
                    for o in np.arange(search[0], search[1], coarse_step))
    if not np.isfinite(coarse[0][0]):
        return None, float("inf"), 0.0
    c0 = coarse[0][1]
    fine = sorted((plane_residual(byf, track, o, frames, fps=fps), float(o))
                  for o in np.arange(c0 - coarse_step - 0.1,
                                     c0 + coarse_step + 0.1, fine_step))
    best_res, best_off = fine[0]
    far = [r for r, o in coarse
           if abs(o - c0) > CONTRAST_GAP_S and np.isfinite(r)]
    contrast = (min(far) / best_res) if far and best_res > 1e-9 else float("inf")
    return best_off, best_res, float(contrast)


def align_play_plane(labels, track, game_key: int, play_id: int, *,
                     max_disagreement_s: float = MAX_VIEW_DISAGREEMENT_S,
                     min_contrast: float = MIN_CONTRAST) -> Alignment:
    """Recover one play's video-to-tracking offset by plane fit, or refuse to."""
    index = {p: i for i, p in enumerate(track.players)}
    per_view = {}
    for view in ("Sideline", "Endzone"):
        name = f"{int(game_key)}_{int(play_id):06d}_{view}.mp4"
        sub = labels[labels["video"] == name]
        if sub.empty:
            continue
        byf = helmet_boxes_by_frame(sub, index)
        if not byf:
            continue
        per_view[view] = best_offset_plane(byf, track)

    if len(per_view) < 2:
        return Alignment(None, per_view, "only one view could be scored")
    offs = [o for o, _r, _c in per_view.values() if o is not None]
    if len(offs) < 2:
        return Alignment(None, per_view, "a view produced no offset")
    worst = min(c for _o, _r, c in per_view.values())
    if worst < min_contrast:
        return Alignment(None, per_view,
                         f"flat residual curve (contrast {worst:.1f}x)")
    spread = abs(offs[0] - offs[1])
    if spread > max_disagreement_s:
        return Alignment(None, per_view, f"views disagree by {spread:.2f} s")
    return Alignment(float(np.mean(offs)), per_view, "ok")


# --------------------------------------------------------------------------
# The motion correlation: kept as an independent second opinion.
# --------------------------------------------------------------------------

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
    """Recover one play's offset by motion correlation, or refuse to.

    ``labels`` is the whole train_labels frame; ``track`` the PlayTracking.
    Prefer ``align_play_plane``; this is the geometry-free cross-check.
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
