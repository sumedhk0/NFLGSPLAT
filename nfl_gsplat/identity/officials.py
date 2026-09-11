"""Are these detections the officials? Stripes say so.

WHY. The person detector finds the referees, the linker gives them ids, and
the render drew them as beige players: 26-27 bodies per frame where 22 play.
Nothing else in the pipeline tells an official from a player -- the roster
gate only names numbers, and officials wear none.

WHAT. An official's shirt is black-and-white vertical stripes about 2-3 cm
wide; a jersey is one colour with a number. In the torso band of a person
box the stripes give a large mean absolute HORIZONTAL gradient relative to
the band's overall contrast; a plain jersey gives a small one. The score
is that ratio, averaged over a track's crops, and the threshold is measured
on real tracks rather than assumed.

MEASURED 2026-09-04, play 1: NOT discriminative. Per-id median scores form
a continuum, 0.21-0.43 on sideline crops (bodies ~140 px) and 0.18-0.48 on
endzone crops (200-260 px), with KC and BAL players at the top both times:
jersey numbers, logos and folds carry as much horizontal gradient as
stripes do at this resolution. Kept as the record; nothing calls it. An
official is still drawn as a beige player.
"""
from __future__ import annotations

import numpy as np

# Torso band of the person box, as jersey_ocr uses.
TORSO_TOP: float = 0.18
TORSO_BOTTOM: float = 0.55
MIN_BAND_PX: int = 8


def stripe_score(img_bgr, box) -> float:
    """Horizontal-gradient energy of the torso band over its contrast, or NaN.

    Grey-level, resized to a fixed width so the score does not depend on the
    box's size in pixels: ``mean |dI/dx| / (std I + eps)``."""
    import cv2

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bh = y2 - y1
    ya, yb = int(max(0, y1 + TORSO_TOP * bh)), int(min(h, y1 + TORSO_BOTTOM * bh))
    xa, xb = int(max(0, x1)), int(min(w, x2))
    if yb - ya < MIN_BAND_PX or xb - xa < MIN_BAND_PX:
        return float("nan")
    band = cv2.cvtColor(img_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = cv2.resize(band, (48, 32), interpolation=cv2.INTER_AREA)
    gx = np.abs(np.diff(band, axis=1)).mean()
    return float(gx / (band.std() + 1e-3))


def is_official(scores, *, threshold: float) -> bool:
    """A track is an official when the median of its crop scores clears the threshold."""
    s = np.asarray([v for v in scores if np.isfinite(v)], float)
    return bool(len(s) >= 3 and np.median(s) > threshold)
