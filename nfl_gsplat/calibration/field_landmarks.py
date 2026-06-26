"""Canonical NFL field landmarks in the world frame.

World frame
-----------
Right-handed, metric (meters), origin at the center of the field, +X toward
the home endzone, +Z up. The field surface is at Z = 0.

Dimensions (from the NFL rulebook)
----------------------------------
- Total length, goal-line to goal-line inclusive of both endzones:
    120 yd = 109.728 m     → half-length = 54.864 m
- Playing field, goal-line to goal-line (no endzones):
    100 yd =  91.440 m
- Endzone depth: 10 yd = 9.144 m  (each)
- Width:        53⅓ yd = 48.768 m  → half-width = 24.384 m
- Hash marks: 70'9" from each sideline  → offset from centerline:
    80 ft − 70.75 ft = 9.25 ft = 2.8194 m
- Yard-line spacing (painted): every 5 yd = 4.572 m
- Goalposts: at the center of the back (end) line of each endzone,
    (±54.864, 0, 0).

Naming convention
-----------------
Yard lines go from the **away** end (−X) to the **home** end (+X):

    away_goal, away_45, away_40, ..., away_5,  mid_50,
    home_5,    home_10, ..., home_45,          home_goal

For each yard line ``YL`` we expose intersection points:

    {YL}_left_sideline     at y = +half_width
    {YL}_right_sideline    at y = −half_width
    {YL}_left_hash         at y = +hash_offset
    {YL}_right_hash        at y = −hash_offset

Endzone corners and goalposts are separate named points.

The dict :data:`NFL_LANDMARKS` maps ``name`` → np.ndarray([x, y, z]) float64.
It is frozen (wrapped in :data:`MappingProxyType`). To add a new landmark,
extend :func:`_build_landmarks`, not the frozen dict.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

# --- Frozen field constants ---------------------------------------------------

FIELD_LENGTH_M: float = 109.728      # 120 yd, includes both endzones
FIELD_WIDTH_M: float = 48.768        # 53⅓ yd
ENDZONE_DEPTH_M: float = 9.144       # 10 yd
HALF_LENGTH_M: float = FIELD_LENGTH_M / 2.0                  # 54.864
HALF_WIDTH_M: float = FIELD_WIDTH_M / 2.0                    # 24.384
HASH_OFFSET_M: float = 2.8194        # 9.25 ft from centerline
YARD_TO_M: float = 0.9144
YARD_LINE_SPACING_M: float = 5.0 * YARD_TO_M                 # 4.572

NUMBER_BOTTOM_Y_M: float = HALF_WIDTH_M - 12.0 * YARD_TO_M   # 13.4112 (12 yd from sideline)
NUMBER_TOP_Y_M: float = NUMBER_BOTTOM_Y_M + 6.0 * 0.3048     # 15.24  (numbers are 6 ft tall)

# Goal lines are at the far edges of the playing field, i.e.
#   X = ± (HALF_LENGTH_M − ENDZONE_DEPTH_M) = ± 45.72
GOAL_LINE_X_M: float = HALF_LENGTH_M - ENDZONE_DEPTH_M       # 45.720


def _yardline_x_m(name: str) -> float:
    """World X for a named yard line.

    NFL yard lines are numbered by distance from each team's own goal line
    (e.g. ``home_35`` is 35 yards into home territory, 15 yards short of
    midfield). Given +X toward the home endzone, the signed X is therefore:

        x = (50 - yd) * 0.9144    on the home side  → positive
        x = -(50 - yd) * 0.9144   on the away side  → negative
    """
    if name == "mid_50":
        return 0.0
    if name == "home_goal":
        return GOAL_LINE_X_M
    if name == "away_goal":
        return -GOAL_LINE_X_M
    side, yd_s = name.split("_")
    yd = int(yd_s)
    if yd < 5 or yd > 45 or yd % 5 != 0:
        raise ValueError(f"yard line name {name!r} invalid — must be 5..45 in 5-yd steps")
    x_from_mid = (50 - yd) * YARD_TO_M
    return x_from_mid if side == "home" else -x_from_mid


def _yardline_names() -> list[str]:
    names = ["away_goal"]
    for yd in range(5, 50, 5):
        names.append(f"away_{yd}")
    names.append("mid_50")
    for yd in range(45, 0, -5):
        names.append(f"home_{yd}")
    names.append("home_goal")
    return names


def _build_landmarks() -> dict[str, np.ndarray]:
    lm: dict[str, np.ndarray] = {}
    for yl in _yardline_names():
        x = _yardline_x_m(yl)
        lm[f"{yl}_left_sideline"]  = np.array([x, +HALF_WIDTH_M, 0.0])
        lm[f"{yl}_right_sideline"] = np.array([x, -HALF_WIDTH_M, 0.0])
        lm[f"{yl}_left_hash"]      = np.array([x, +HASH_OFFSET_M, 0.0])
        lm[f"{yl}_right_hash"]     = np.array([x, -HASH_OFFSET_M, 0.0])

    # Painted field numbers (only at 10/20/30/40 and mid-50), centered on the yard
    # line. Top/bottom anchors give Y far from the hashes → vertical conditioning.
    number_yls = ["away_10", "away_20", "away_30", "away_40", "mid_50",
                  "home_40", "home_30", "home_20", "home_10"]
    for yl in number_yls:
        x = _yardline_x_m(yl)
        for sgn, lr in [(+1.0, "left"), (-1.0, "right")]:
            lm[f"{yl}_{lr}_number_bottom"] = np.array([x, sgn * NUMBER_BOTTOM_Y_M, 0.0])
            lm[f"{yl}_{lr}_number_top"]    = np.array([x, sgn * NUMBER_TOP_Y_M, 0.0])

    # End line × sideline corners (back of each endzone).
    for sx, sx_name in [(+HALF_LENGTH_M, "home"), (-HALF_LENGTH_M, "away")]:
        lm[f"{sx_name}_endline_left_corner"]  = np.array([sx, +HALF_WIDTH_M, 0.0])
        lm[f"{sx_name}_endline_right_corner"] = np.array([sx, -HALF_WIDTH_M, 0.0])

    # Goalpost bases: center of end line, each endzone. Z=0.
    lm["home_goalpost_base"] = np.array([+HALF_LENGTH_M, 0.0, 0.0])
    lm["away_goalpost_base"] = np.array([-HALF_LENGTH_M, 0.0, 0.0])

    # Pylons sit on the inside corner of each endzone (at the goal line × sideline)
    # — these duplicate the goal-line sideline intersections, which is fine:
    # both names are retained for annotator convenience.
    for sx, sx_name in [(+GOAL_LINE_X_M, "home"), (-GOAL_LINE_X_M, "away")]:
        lm[f"{sx_name}_pylon_front_left"]  = np.array([sx, +HALF_WIDTH_M, 0.0])
        lm[f"{sx_name}_pylon_front_right"] = np.array([sx, -HALF_WIDTH_M, 0.0])
    for sx, sx_name in [(+HALF_LENGTH_M, "home"), (-HALF_LENGTH_M, "away")]:
        lm[f"{sx_name}_pylon_back_left"]  = np.array([sx, +HALF_WIDTH_M, 0.0])
        lm[f"{sx_name}_pylon_back_right"] = np.array([sx, -HALF_WIDTH_M, 0.0])

    # Force float64 and immutability of each array.
    out: dict[str, np.ndarray] = {}
    for k, v in lm.items():
        a = np.asarray(v, dtype=np.float64)
        a.setflags(write=False)
        out[k] = a
    return out


NFL_LANDMARKS: Mapping[str, np.ndarray] = MappingProxyType(_build_landmarks())


def landmark_points(names: list[str]) -> np.ndarray:
    """Stack named landmarks into an ``(N, 3)`` float64 array."""
    try:
        return np.stack([NFL_LANDMARKS[n] for n in names], axis=0)
    except KeyError as e:
        raise KeyError(
            f"unknown landmark {e.args[0]!r}; valid names include e.g. "
            "'home_35_left_hash', 'mid_50_right_sideline', 'home_goalpost_base'"
        ) from None


def list_landmark_names() -> list[str]:
    """All valid landmark names, sorted."""
    return sorted(NFL_LANDMARKS.keys())
