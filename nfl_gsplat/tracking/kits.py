"""Kit labels per detection box: rule D (identity.team_color) per camera.

The cheapest cue position-only linking lacks is the team, and the reliable
form of it is the torso SATURATION split per camera (measured 2026-09-05 on
plays 1, 2 and 4 of the All-22: 2.3 % wrong at margin 0.4 over 83 % of the
detections; 4.6 % endzone, 2.1 % sideline). Label 1 is the coloured kit,
0 the white one, -1 unknown or too close to the split to say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.identity.team_color import two_means_1d
from nfl_gsplat.identity.torso_colours import detection_colours

KIT_MARGIN: float = 0.4      # |(S - midpoint) / (centre_hi - centre_lo)| to carry a label
MIN_COLOURED: int = 8


class KitError(RuntimeError):
    pass


def kit_margins(det: dict, video) -> tuple[dict, tuple[float, float]]:
    """``det``: ``{frame: boxes [N, 4]}`` for one camera. Returns
    ``({frame: margins [N]}, (centre_lo, centre_hi))`` -- the signed,
    centre-gap-normalised distance of each box's torso saturation from the
    camera's split midpoint (NaN unknown). Reads the video once."""
    frames, boxes = [], []
    for f in sorted(det):
        for b in det[f]:
            frames.append(int(f))
            boxes.append(b)
    if not frames:
        return {}, (float("nan"), float("nan"))
    boxes = np.asarray(boxes, float)
    df = pd.DataFrame({"frame": frames, "bbox_x1": boxes[:, 0], "bbox_y1": boxes[:, 1],
                       "bbox_x2": boxes[:, 2], "bbox_y2": boxes[:, 3]})
    sat = detection_colours(df, video, max_per_frame=512)[:, 1]
    ok = np.isfinite(sat)
    if int(ok.sum()) < MIN_COLOURED:
        raise KitError(f"{video}: only {int(ok.sum())} detections carry a torso colour")
    c, _ = two_means_1d(sat[ok])
    mid, span = 0.5 * (c[0] + c[1]), max(float(c[1] - c[0]), 1e-6)
    m = np.full(len(sat), np.nan)
    m[ok] = (sat[ok] - mid) / span
    out, i = {}, 0
    for f in sorted(det):
        n = len(det[f])
        out[f] = m[i:i + n]
        i += n
    return out, (float(c[0]), float(c[1]))


def labels_from_margins(m, *, margin: float = KIT_MARGIN) -> np.ndarray:
    """``[N]`` ints: 1 coloured, 0 white, -1 unknown or within ``margin``."""
    m = np.asarray(m, float)
    lab = np.full(len(m), -1, int)
    sure = np.isfinite(m) & (np.abs(m) >= margin)
    lab[sure] = (m[sure] > 0).astype(int)
    return lab
