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
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Ordered by how often they win, cheapest first. 180/0.25 is the broadcast
# default; 0.05 finds yard lines on the wider feed; 120 finds its sidelines.
DETECT_SETTINGS: tuple[tuple[int, float], ...] = (
    (180, 0.25),
    (180, 0.05),
    (135, 0.05),
    (120, 0.05),
    (150, 0.15),
)


def detect_all(images, *, white_thresh: int, min_line_len_frac: float):
    cfg = replace(field_detect.FieldDetectConfig(),
                  white_thresh=white_thresh,
                  min_line_len_frac=min_line_len_frac)
    return {f: field_detect.detect_field_features(img, cfg=cfg)
            for f, img in images.items()}


def calibrate_clip(images, width: int, height: int, *,
                   settings=DETECT_SETTINGS, propagate: bool = True,
                   require_quality: bool = True, **kwargs):
    """``(cams, focal, centre, quality, settings)`` for one clip, or raise.

    ``images`` maps frame index to a BGR frame. Returns the settings that won so
    a caller can reuse them on the rest of a game rather than re-searching.
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
        _LOG.info("white>=%d len>=%.2f: %d frames, %.1f deg, rms %.1f px",
                  white, frac, len(cams), quality["fov_deg"], quality["rms_px"])
        if best is None or quality["rms_px"] < best[3]["rms_px"]:
            best = (cams, focal, centre, quality, (white, frac))
    if best is None:
        raise CalibrationError(
            "no detector setting produced a camera that could exist for this "
            f"clip; tried {len(settings)}.")
    cams, focal, centre, quality, won = best
    if require_quality:
        from nfl_gsplat.calibration.from_paint import (
            PAINT_AUDIT_REF_HEIGHT,
            _enforce_quality,
        )

        _enforce_quality(quality, centre, None, height / PAINT_AUDIT_REF_HEIGHT)
    _LOG.info("clip calibrated with white>=%d len>=%.2f: centre %s, "
              "%.1f deg, rms %.1f px", won[0], won[1], np.round(centre, 1),
              quality["fov_deg"], quality["rms_px"])
    return cams, focal, centre, quality, won
