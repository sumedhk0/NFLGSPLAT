"""Pick each player's best-reference SMPL-X shape across the whole season.

The avatar/shape library freezes one ``betas`` per player. Rather than locking
the first estimate, we choose the betas from the player's **highest-quality
appearance** (largest, most confident reference) seen anywhere in the season —
reusing the same ``bbox_area * conf`` metric as avatar reference selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.avatars.lhm_wrapper import select_reference_index


@dataclass(frozen=True)
class BetasAppearance:
    bbox_area: float
    conf: float
    betas: np.ndarray


def select_best_betas(appearances: list[BetasAppearance]) -> np.ndarray | None:
    """Return the betas of the appearance maximizing ``bbox_area * conf``.

    Returns None if there are no appearances clearing the confidence gate.
    """
    if not appearances:
        return None
    areas = np.array([a.bbox_area for a in appearances], dtype=np.float64)
    confs = np.array([a.conf for a in appearances], dtype=np.float64)
    idx = select_reference_index(areas, confs)
    if idx < 0:
        return None
    return np.asarray(appearances[idx].betas)
