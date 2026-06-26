"""Ordered landmark class schema for the keypoint detector.

The model has one output channel per class; this fixes the class↔index mapping and
restricts classes to a world-X window (the footage's yard range) so K and per-class
data stay tractable (per-dataset model)."""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS


class LandmarkSchema:
    def __init__(self, yard_min: float, yard_max: float) -> None:
        names = []
        for name in sorted(NFL_LANDMARKS):
            x = float(NFL_LANDMARKS[name][0])
            if yard_min <= x <= yard_max:
                names.append(name)
        if not names:
            raise ValueError(f"no landmarks in X window [{yard_min}, {yard_max}]")
        self._names = names
        self._index = {n: i for i, n in enumerate(names)}

    def class_names(self) -> list[str]:
        return list(self._names)

    @property
    def num_classes(self) -> int:
        return len(self._names)

    def index(self, name: str) -> int:
        return self._index[name]

    def world_xyz(self, name: str) -> np.ndarray:
        return np.asarray(NFL_LANDMARKS[name], dtype=np.float64)
