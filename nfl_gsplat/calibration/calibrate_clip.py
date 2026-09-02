"""Calibrate one clip from paint, choosing the detector settings by result.

WHY THIS EXISTS. The paint pipeline works, but only once the detector is set up
for the feed it is looking at, and the right settings differ per feed by more
than anyone would guess. On production All-22 the same clip gives:

    white >= 180 (the default)   no camera that could exist
    white >= 135                 no camera that could exist
    white >= 120                 22 frames, 23 deg lens, rms 15.3 px, plausible

while a 1280x720 broadcast clip is fine at 180. The paint is dimmer and the
grass brighter on the wide feed, so a threshold tuned on one silently blinds the
detector on the other -- with zero sidelines found, world Y rests on two hash
rows 5.6 m apart instead of sidelines 48.8 m apart, and the camera runs away to
two kilometres.

So the settings are SEARCHED rather than assumed, and scored by the camera that
comes out: physically possible first, then residual. That is the same rule the
reference-frame search uses, and for the same reason -- every cheaper proxy
measured here (line count, grid consistency, per-frame residual) can be
perfectly happy while the labelling is wrong.

It is not free: each setting means re-detecting every frame and re-solving. The
sweep is small and ordered so the common case is tried first.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration import field_detect
from nfl_gsplat.calibration.from_paint import cameras_from_paint_pooled
from nfl_gsplat.calibration.orientation import (
    cameras_from_rotated,
    rotate_images_90,
)
from nfl_gsplat.calibration.player_scale import height_score
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Ordered by how often they win, cheapest first. 180/0.25 is the broadcast
# default; 0.05 finds yard lines on the wider feed; 120 finds its sidelines.
DETECT_SETTINGS: tuple[tuple[int, float], ...] = (
    (180, 0.25),     # broadcast feeds
    (120, 0.05),     # All-22: dimmer paint, and the only setting that finds
                     # its sidelines at all
    (180, 0.05),
    (135, 0.05),
)

# Stop sweeping once a setting is this good (at the reference height). Every
# further setting costs a full re-detect and several solves, and nothing
# measured has beaten a camera already this consistent.
GOOD_ENOUGH_RMS_PX: float = 20.0

# Player-height cost below which a camera is accepted on the PLAYERS alone,
# without having to satisfy the paint residual gate as well.
#
# Those two gates disagree, and the players are right. The residual bar was set
# when residual was also the selection criterion; now that people decide, a
# camera can be correct about the world and still fit the lines a little worse
# than a nonsense one that fits them beautifully. Measured, the camera with
# players at 2.00 m scored 41 px and was rejected, while the one at 2.5 px --
# under which no player had a plausible height at all -- would have passed.
MAX_PLAYER_COST: float = 0.60


def _player_cost(cams, player_boxes):
    """Median player-height cost for a camera set, or inf if players cannot say."""
    if not player_boxes:
        return float("inf")
    costs = []
    for frame, boxes in player_boxes.items():
        if not len(boxes) or not cams:
            continue
        nearest = min(cams, key=lambda c: abs(c - frame))
        K, R, t = cams[nearest]
        cost, _used, _median = height_score(K, R, t, boxes)
        if np.isfinite(cost):
            costs.append(cost)
    return float(np.median(costs)) if costs else float("inf")


def detect_all(images, *, white_thresh: int, min_line_len_frac: float):
    cfg = replace(field_detect.FieldDetectConfig(),
                  white_thresh=white_thresh,
                  min_line_len_frac=min_line_len_frac)
    return {f: field_detect.detect_field_features(img, cfg=cfg)
            for f, img in images.items()}


def calibrate_clip(images, width: int, height: int, *,
                   settings=DETECT_SETTINGS, propagate: bool = True,
                   require_quality: bool = True, player_boxes=None, **kwargs):
    """``(cams, focal, centre, quality, settings)`` for one clip, or raise.

    ``images`` maps frame index to a BGR frame. Returns the settings that won so
    a caller can reuse them on the rest of a game rather than re-searching.

    ``player_boxes`` maps frame index to person boxes, and when given it DECIDES
    between cameras the paint cannot separate. It has to, because residual gets
    this backwards. Measured on one All-22 clip:

        fov 61.7 deg, 201 m out   rms  2.5 px   no player a plausible height
        fov 60.1 deg, 183 m out   rms  0.9 px   no player a plausible height
        fov 24.9 deg,  50 m out   rms 22.9 px   players 1.95 m
        fov 13.3 deg,  87 m out   rms 12.8 px   players 2.00 m

    The best-fitting cameras are the wrong ones -- a field is a plane covered in
    parallel lines and says nothing about scale, so a camera twice as far away
    behind twice the lens fits it just as well. People are the ruler.
    """
    best = None
    for white, frac in settings:
        try:
            feats = detect_all(images, white_thresh=white,
                               min_line_len_frac=frac)
            cams, focal, centre, mirrored, quality = cameras_from_paint_pooled(
                feats, width, height, images=images, propagate=propagate,
                require_quality=False, **kwargs)
        except (CalibrationError, ValueError):
            continue
        if mirrored or not quality["plausible_mount"]:
            continue
        score = _player_cost(cams, player_boxes)
        quality = dict(quality, player_cost=score, ranked_on=(
            "players" if np.isfinite(score) else "residual"))
        _LOG.info("white>=%d len>=%.2f: %d frames, %.1f deg, rms %.1f px, "
                  "player cost %.2f", white, frac, len(cams),
                  quality["fov_deg"], quality["rms_px"], score)
        # Players first when they can speak, residual only when they cannot.
        key = ((0, score) if np.isfinite(score) else (1, quality["rms_px"]))
        if best is None or key < best[5]:
            best = (cams, focal, centre, quality, (white, frac), key)
        if (not np.isfinite(score)
                and quality["rms_px"] <= GOOD_ENOUGH_RMS_PX * (height / 720.0)):
            _LOG.info("stopping the sweep early: %.1f px is good enough",
                      quality["rms_px"])
            break
    if best is None:
        raise CalibrationError(
            "no detector setting produced a camera that could exist for this "
            f"clip; tried {len(settings)}.")
    cams, focal, centre, quality, won, _key = best
    if require_quality:
        from nfl_gsplat.calibration.from_paint import (
            PAINT_AUDIT_REF_HEIGHT,
            _enforce_quality,
        )

        by_players = (quality.get("ranked_on") == "players"
                      and quality.get("player_cost", np.inf) <= MAX_PLAYER_COST)
        if by_players:
            # Accepted on the people. Still refuse an impossible mount, but do
            # not also demand the residual bar, which ranks these backwards.
            if not quality["plausible_mount"]:
                raise CalibrationError(
                    "paint solve produced a camera no broadcast mount could "
                    f"be: centre {np.round(centre, 1)}, "
                    f"{quality['fov_deg']:.1f} deg field of view.")
        else:
            _enforce_quality(quality, centre, None,
                             height / PAINT_AUDIT_REF_HEIGHT)
    _LOG.info("clip calibrated with white>=%d len>=%.2f: centre %s, "
              "%.1f deg, rms %.1f px", won[0], won[1], np.round(centre, 1),
              quality["fov_deg"], quality["rms_px"])
    return cams, focal, centre, quality, won


# How many different frame samples to try before giving up on a clip.
#
# One sample is a lottery. Which frames are drawn decides which labellings exist
# to choose from, and measured on production clips a single sample solved 2 of 3
# times on one and 1 of 3 on another -- while the cameras it DID produce agreed
# to a few metres. So the failure is per-sample, not per-clip, and re-drawing is
# the cheapest fix available now that the search is fast.
DEFAULT_ATTEMPTS: int = 4


def sample_frames(path, n_frames: int, offset: int = 0):
    """``{frame: image}`` spread over a clip, starting at ``offset``."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = {}
    for f in np.linspace(offset, max(1, total - 2), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out[int(f)] = img
    cap.release()
    return out, width, height, total


def detect_players(model, images, *, probe: int = 6, conf: float = 0.3):
    """Person boxes on a few frames, for the player-height check."""
    frames = sorted(images)[::max(1, len(images) // probe)][:probe]
    out = {}
    for f in frames:
        res = model.predict(images[f], classes=[0], conf=conf, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            out[f] = res.boxes.xyxy.cpu().numpy()
    return out


def calibrate_video(path, *, attempts: int = DEFAULT_ATTEMPTS,
                    n_frames: int = 28, model=None, **kwargs):
    """Calibrate a clip, re-drawing frames until the players are satisfied.

    Returns ``(cams, focal, centre, quality, settings, images)`` -- the images of
    the winning attempt come back so a caller can reuse them without decoding
    again.

    Attempts are ranked by the PLAYER cost, not by which one happened to
    succeed first, so a later sample that describes the world better wins. It
    stops early on a good one, since further draws cost a full search each.
    """
    from ultralytics import YOLO

    model = model or YOLO("yolov8n.pt")
    best = None
    for attempt in range(attempts):
        offset = attempt * 7
        count = n_frames + 4 * attempt
        images, width, height, total = sample_frames(path, count, offset)
        if len(images) < 10:
            continue
        players = detect_players(model, images)
        try:
            cams, focal, centre, quality, won = calibrate_clip(
                images, width, height, player_boxes=players, **kwargs)
        except CalibrationError as exc:
            _LOG.info("attempt %d (%d frames, offset %d): %s",
                      attempt + 1, count, offset, str(exc)[:70])
            continue
        cost = quality.get("player_cost", float("inf"))
        _LOG.info("attempt %d: centre %s, %.1f deg, player cost %.2f",
                  attempt + 1, np.round(centre, 1), quality["fov_deg"], cost)
        if best is None or cost < best[3].get("player_cost", float("inf")):
            best = (cams, focal, centre, quality, won, images)
        if np.isfinite(cost) and cost <= MAX_PLAYER_COST * 0.75:
            break
    if best is None:
        raise CalibrationError(
            f"{Path(path).name} did not calibrate in {attempts} attempts.")
    return best


def rotate_boxes_90(boxes, width: int):
    """Person boxes under the quarter turn, still axis-aligned."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([b[:, 1], (width - 1) - b[:, 2],
                            b[:, 3], (width - 1) - b[:, 0]])


def calibrate_candidates(images, width: int, height: int, *,
                         settings=DETECT_SETTINGS, player_boxes=None,
                         orientations=("upright", "quarter-turn"), **kwargs):
    """Every physically possible camera this clip admits, best first.

    One view cannot always tell its candidates apart -- see joint_views for the
    measured case where two cameras both looked fine alone and disagreed about
    the width of the field by a factor of three. Handing the alternatives on
    lets the OTHER view break the tie.

    Both ORIENTATIONS are tried, because the endzone camera looks down the field
    and its two line families are swapped relative to everything the labeller
    assumes. Solved upright, a production endzone clip put players across 101 m
    of a 48.8 m field. The quarter turn is exact, not a fit, so trying it costs
    only the re-detect.
    """
    out = []
    for how in orientations:
        if how == "upright":
            imgs, w, h, boxes, back = images, width, height, player_boxes, None
        else:
            imgs, w, h = rotate_images_90(images)
            boxes = ({f: rotate_boxes_90(b, width)
                      for f, b in player_boxes.items()}
                     if player_boxes else None)
            back = width
        for white, frac in settings:
            try:
                feats = detect_all(imgs, white_thresh=white,
                                   min_line_len_frac=frac)
                cams, focal, centre, mirrored, quality = (
                    cameras_from_paint_pooled(
                        feats, w, h, images=imgs, propagate=True,
                        require_quality=False, **kwargs))
            except (CalibrationError, ValueError):
                continue
            if mirrored or not quality["plausible_mount"] or not cams:
                continue
            cost = _player_cost(cams, boxes)
            if back is not None:
                cams = cameras_from_rotated(cams, back)
            out.append({"cams": cams, "centre": centre, "focal": focal,
                        "quality": dict(quality, player_cost=cost),
                        "settings": (white, frac), "orientation": how})
    out.sort(key=lambda d: (d["quality"]["player_cost"]
                            if np.isfinite(d["quality"]["player_cost"])
                            else 1e9, d["quality"]["rms_px"]))
    _LOG.info("clip admits %d possible cameras", len(out))
    return out


# Two camera centres this far apart are different answers, not the same answer
# twice. Well under the errors being separated, which are tens of metres.
SAME_CAMERA_M: float = 5.0


def candidates_for_video(path, *, attempts: int = DEFAULT_ATTEMPTS,
                         n_frames: int = 28, model=None, **kwargs):
    """Candidate cameras pooled over several frame samples, best first.

    WHY POOL RATHER THAN RETRY. Which frames are drawn decides which labellings
    exist to choose from, and the endzone view is the extreme case: one sample
    put the camera at (161, -5.8, 10.6), on the field's long axis where an
    endzone camera belongs, and another at (12.1, -174.3, 26.1), out beyond the
    sideline. Both passed every check their own view can apply.

    calibrate_video handles that by keeping the best attempt, which cannot work
    when no single view can tell the attempts apart. So every attempt's cameras
    are KEPT instead, and the choice is deferred to joint_views, where the other
    camera of the play gets a vote.
    """
    from ultralytics import YOLO

    model = model or YOLO("yolov8n.pt")
    pool = []
    for attempt in range(attempts):
        images, width, height, _total = sample_frames(
            path, n_frames + 4 * attempt, attempt * 7)
        if len(images) < 10:
            continue
        players = detect_players(model, images)
        for cand in calibrate_candidates(images, width, height,
                                         player_boxes=players, **kwargs):
            cand["attempt"] = attempt
            pool.append(cand)

    pool.sort(key=lambda d: (d["quality"]["player_cost"]
                             if np.isfinite(d["quality"]["player_cost"])
                             else 1e9, d["quality"]["rms_px"]))
    kept = []
    for cand in pool:
        c = np.asarray(cand["centre"], float)
        if any(np.linalg.norm(c - np.asarray(k["centre"], float)) < SAME_CAMERA_M
               for k in kept):
            continue
        kept.append(cand)
    _LOG.info("%s: %d distinct cameras from %d attempts",
              Path(path).name, len(kept), attempts)
    return kept
