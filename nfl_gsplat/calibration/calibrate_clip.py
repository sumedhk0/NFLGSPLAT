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

import numpy as np

from nfl_gsplat.calibration import field_detect
from nfl_gsplat.calibration.from_paint import cameras_from_paint_pooled
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
