"""A synthetic uniform on the avatar: the body's look from identity, not pixels.

WHY. Textures fitted from the footage cannot give a jersey: at 140 px a
limb is 3-5 px wide, a vertex gets one to three clean samples over a
play, turf bleeds in and the colours speckle; after de-mixing and
smoothing the fit's measured gain over the plain median is about zero
(2026-09-04). Flat team colours with a helmet already read cleaner than
any fitted texture. The avatar twin's look therefore comes from what the
pipeline KNOWS -- the team, from identity -- painted by body region.
The footage keeps the one job it does well, the field.

WHAT. Regions are fixed in the template's vertex order (SMPL-X, upright,
y up, x across the shoulders): helmet, jersey (torso and the sleeves to
the elbow), forearm skin, gloves, pants (hips to the knee), socks, shoes.
A kit names a colour per region; ``dress`` returns ``[V, 3]`` colours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.render.helmet import HEAD_ABOVE_NECK_M, NECK_JOINT

PELVIS_JOINT: int = 0
KNEE_JOINTS = (4, 5)
ANKLE_JOINTS = (7, 8)
ELBOW_JOINTS = (18, 19)
WRIST_JOINTS = (20, 21)

SKIN_RGB = (0.55, 0.38, 0.28)
SHOE_RGB = (0.10, 0.10, 0.10)


@dataclass(frozen=True)
class Kit:
    jersey: tuple
    pants: tuple
    socks: tuple
    helmet: tuple
    gloves: tuple
    number: tuple            # the numeral colour on the jersey


# BAL @ KC 2024 week 1: KC in red over white, BAL in white over black.
KITS: dict[str, Kit] = {
    "KC": Kit(jersey=(0.89, 0.09, 0.22), pants=(0.94, 0.94, 0.94), socks=(0.89, 0.09, 0.22),
              helmet=(0.89, 0.09, 0.22), gloves=(0.94, 0.94, 0.94), number=(0.98, 0.98, 0.98)),
    "BAL": Kit(jersey=(0.95, 0.95, 0.95), pants=(0.09, 0.09, 0.11), socks=(0.09, 0.09, 0.11),
               helmet=(0.08, 0.08, 0.10), gloves=(0.09, 0.09, 0.11), number=(0.16, 0.10, 0.35)),
}
DEFAULT_KIT = Kit(jersey=(0.75, 0.75, 0.75), pants=(0.85, 0.85, 0.85), socks=(0.75, 0.75, 0.75),
                  helmet=(0.85, 0.85, 0.85), gloves=(0.85, 0.85, 0.85), number=(0.1, 0.1, 0.1))


def regions(v_template, joints_template) -> dict[str, np.ndarray]:
    """Boolean ``[V]`` masks per region from the template geometry. Every
    vertex is in exactly one region."""
    vt = np.asarray(v_template, float)
    J = np.asarray(joints_template, float)
    y, x = vt[:, 1], np.abs(vt[:, 0])
    neck_y = J[NECK_JOINT, 1]
    hip_y = J[PELVIS_JOINT, 1] - 0.02
    knee_y = float(np.mean([J[j, 1] for j in KNEE_JOINTS]))
    ankle_y = float(np.mean([J[j, 1] for j in ANKLE_JOINTS]))
    elbow_x = float(np.mean([abs(J[j, 0]) for j in ELBOW_JOINTS]))
    wrist_x = float(np.mean([abs(J[j, 0]) for j in WRIST_JOINTS]))
    helmet = y > neck_y + HEAD_ABOVE_NECK_M
    arm = (~helmet) & (y > hip_y) & (x > elbow_x)            # past the elbow: forearm and hand
    gloves = arm & (x > wrist_x - 0.02)
    skin = arm & ~gloves
    jersey = (~helmet) & (~arm) & (y > hip_y)
    below = (~helmet) & (~arm) & (y <= hip_y)
    shoes = below & (y < ankle_y + 0.02)
    socks = below & (~shoes) & (y < knee_y)
    pants = below & (~shoes) & (~socks)
    return {"helmet": helmet, "jersey": jersey, "skin": skin, "gloves": gloves,
            "pants": pants, "socks": socks, "shoes": shoes}


def dress(masks: dict[str, np.ndarray], kit: Kit) -> np.ndarray:
    """``[V, 3]`` colours for the regions under ``kit``."""
    n = len(next(iter(masks.values())))
    out = np.zeros((n, 3), np.float32)
    paint = {"helmet": kit.helmet, "jersey": kit.jersey, "skin": SKIN_RGB, "gloves": kit.gloves,
             "pants": kit.pants, "socks": kit.socks, "shoes": SHOE_RGB}
    for name, m in masks.items():
        out[m] = np.asarray(paint[name], np.float32)
    return out
