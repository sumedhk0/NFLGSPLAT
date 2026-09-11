"""Helmets: the head of a rendered body wears the team's helmet.

WHY. A SMPL-X body is a bare mannequin. At 15-25 px a head is the one part
of a player the eye checks for "football": a red or black shell reads as a
helmet, a beige blob reads as a shop dummy. The fitted texture cannot
supply it -- the head is a dozen turf-mixed pixels in the footage.

WHAT. The vertices above the neck joint (in the template) take the team's
helmet colour and move outward from the head's centre by a couple of
centimetres, so the shell is larger than the skull as a helmet is. Pose
does not matter: the mask is fixed in the template's vertex order and the
centre is taken from the placed vertices.
"""
from __future__ import annotations

import numpy as np

# Above the neck joint by this much, in the template: skull and face, not the throat.
HEAD_ABOVE_NECK_M: float = 0.03
NECK_JOINT: int = 12
HELMET_INFLATE_M: float = 0.02
HELMET_RGB: dict[str, tuple[float, float, float]] = {
    "KC": (0.89, 0.09, 0.22),      # red shell
    "BAL": (0.08, 0.08, 0.10),     # black shell
}
DEFAULT_HELMET_RGB = (0.85, 0.85, 0.85)


# Shoulder pads: the vertices within this radius of either shoulder joint,
# pushed out from the shoulders' centre and up. Pads make the football
# silhouette; a bare SMPL-X torso reads as a swimmer.
SHOULDER_JOINTS = (16, 17)
PAD_RADIUS_M: float = 0.13
PAD_OUT_M: float = 0.035
PAD_UP_M: float = 0.02


def pads_mask(v_template, joints_template, *, radius_m: float = PAD_RADIUS_M) -> np.ndarray:
    """Boolean ``[V]`` mask of the shoulder vertices from the template geometry."""
    vt = np.asarray(v_template, float)
    J = np.asarray(joints_template, float)
    m = np.zeros(len(vt), bool)
    for j in SHOULDER_JOINTS:
        d = np.linalg.norm(vt - J[j][None, :], axis=1)
        m |= (d < radius_m) & (vt[:, 1] > J[j, 1] - 0.06)          # not the armpit
    return m


def wear_pads(vertices, mask, *, out_m: float = PAD_OUT_M, up_m: float = PAD_UP_M):
    """Copy of ``vertices`` with the masked shoulders pushed ``out_m`` away
    from the shoulders' centre in the horizontal plane and ``up_m`` up
    (world z); the placed bodies are upright."""
    v = np.array(vertices, float, copy=True)
    m = np.asarray(mask, bool)
    if not m.any():
        return v
    centre = v[m].mean(axis=0)
    r = v[m] - centre
    r[:, 2] = 0.0
    n = np.linalg.norm(r, axis=1, keepdims=True)
    v[m] = v[m] + out_m * r / np.maximum(n, 1e-9) + np.array([0.0, 0.0, up_m])
    return v


def head_mask(v_template, joints_template) -> np.ndarray:
    """Boolean ``[V]`` mask of the head vertices from the template geometry."""
    vt = np.asarray(v_template, float)
    neck_y = float(np.asarray(joints_template, float)[NECK_JOINT, 1])
    return vt[:, 1] > neck_y + HEAD_ABOVE_NECK_M


def wear_helmet(vertices, colours, mask, rgb, *, inflate_m: float = HELMET_INFLATE_M):
    """Copies of ``vertices`` ``[V, 3]`` and ``colours`` ``[V, 3]`` with the
    masked head coloured ``rgb`` and pushed ``inflate_m`` away from the head
    centre (mean of the masked vertices, in whatever frame they are in)."""
    v = np.array(vertices, float, copy=True)
    c = np.array(colours, float, copy=True)
    if c.ndim == 1:
        c = np.broadcast_to(c, v.shape).copy()
    m = np.asarray(mask, bool)
    if not m.any():
        return v, c
    centre = v[m].mean(axis=0)
    r = v[m] - centre
    n = np.linalg.norm(r, axis=1, keepdims=True)
    v[m] = v[m] + inflate_m * r / np.maximum(n, 1e-9)
    c[m] = np.asarray(rgb, float)
    return v, c
