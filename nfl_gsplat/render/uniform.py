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


NUMBER_HEIGHT_M: float = 0.24      # NFL back numerals are 10 in; the front 8 in, drawn the same here
NUMBER_TOP_M: float = 0.06         # below the neck joint, in the template
NUMBER_HALF_WIDTH_M: float = 0.17


def number_raster(number: int, size: int = 96) -> np.ndarray:
    """``[size, size]`` bool: the digits of ``number`` as ink, centred, bold."""
    import cv2

    img = np.zeros((size, size), np.uint8)
    text = str(int(number))
    scale = 1.0
    thick = max(2, size // 20)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    scale = 0.8 * size / max(tw, th * 1.3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    org = ((size - tw) // 2, (size + th) // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick, cv2.LINE_AA)
    return img > 127


def paint_number(colours, v_template, joints_template, masks, number: int, colour, *,
                 height_m: float = NUMBER_HEIGHT_M) -> np.ndarray:
    """Copy of ``colours`` with ``number`` painted on the jersey's back and
    front in the template plane (x across, y up), for ids whose identity is
    sure. The digits are a raster sampled at each jersey vertex."""
    out = np.array(colours, np.float32, copy=True)
    vt = np.asarray(v_template, float)
    J = np.asarray(joints_template, float)
    top = J[NECK_JOINT, 1] - NUMBER_TOP_M
    bottom = top - height_m
    ink = number_raster(int(number))
    size = ink.shape[0]
    jersey = masks["jersey"]
    for side in (-1.0, 1.0):                                # back (z < 0) and front (z > 0)
        sel = (jersey & (np.sign(vt[:, 2]) == side) & (vt[:, 1] <= top) & (vt[:, 1] >= bottom)
               & (np.abs(vt[:, 0]) <= NUMBER_HALF_WIDTH_M))
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        # Digits read left-to-right from OUTSIDE the body. SMPL-X +x is the
        # body's own LEFT, which a viewer facing the front sees on their
        # right: reading order runs with +x on the front, against it on the back.
        u = (side * vt[idx, 0] / NUMBER_HALF_WIDTH_M + 1.0) * 0.5 * (size - 1)
        v = (top - vt[idx, 1]) / height_m * (size - 1)
        ui = np.clip(np.rint(u).astype(int), 0, size - 1)
        vi = np.clip(np.rint(v).astype(int), 0, size - 1)
        hit = ink[vi, ui]
        out[idx[hit]] = np.asarray(colour, np.float32)
    return out


DECAL_RASTER: int = 64            # texels across the number window (~0.5 cm)
DECAL_SIGMA_M: float = 0.004       # in-plane extent of a decal Gaussian: strokes must not fill the counters
DECAL_LIFT_M: float = 0.008        # above the surface along the normal, so it wins the depth order


@dataclass(frozen=True)
class Decal:
    """Ink texels of a number pinned to the jersey surface: for each, the
    template face it lies on and its barycentric weights, so it rides the
    posed body (``decal_points``)."""
    faces: np.ndarray        # [M] face index
    bary: np.ndarray         # [M, 3]
    n_back: int              # the first n_back texels are on the back


def _point_in_triangles(pts2, tri2):
    """``[P, F]`` bool: 2-D points inside 2-D triangles, and the barycentric
    weights ``[P, F, 3]``."""
    a, b, c = tri2[:, 0], tri2[:, 1], tri2[:, 2]                       # [F, 2]
    v0, v1 = (b - a), (c - a)
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    v2 = pts2[:, None, :] - a[None, :, :]                                # [P, F, 2]
    d20 = (v2 * v0[None]).sum(2)
    d21 = (v2 * v1[None]).sum(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6) & np.isfinite(u)
    return inside, np.stack([u, v, w], axis=2)


def number_decal(v_template, joints_template, faces, masks, number: int, *,
                 raster: int = DECAL_RASTER, height_m: float = NUMBER_HEIGHT_M) -> Decal:
    """Where the digits of ``number`` sit on the jersey's back and front, in
    the template. Vertex colours cannot carry a digit (the torso lattice is
    2.5 cm, a stroke 3 cm; 76 vertices caught ink and rendered as a smear);
    a decal samples the raster at sub-vertex positions on the surface."""
    vt = np.asarray(v_template, float)
    J = np.asarray(joints_template, float)
    faces = np.asarray(faces, np.int64)
    top = J[NECK_JOINT, 1] - NUMBER_TOP_M
    ink = number_raster(int(number), size=raster)
    jersey = masks["jersey"]
    face_on_jersey = jersey[faces].all(1)
    out_f, out_b, n_back = [], [], 0
    for side in (-1.0, 1.0):
        # faces whose vertices all lie on this side of the torso plane
        side_ok = face_on_jersey & (np.sign(vt[faces, 2]) == side).all(1)
        fidx = np.flatnonzero(side_ok)
        if not len(fidx):
            continue
        tri2 = vt[faces[fidx]][:, :, :2]                                  # (x, y) of each face
        rows, cols = np.nonzero(ink)
        # texel centre -> template (x, y); reading order runs with +x on the
        # front (SMPL-X +x is the body's own left, the viewer's right) and
        # against it on the back
        x = side * ((cols + 0.5) / raster * 2.0 - 1.0) * NUMBER_HALF_WIDTH_M
        y = top - (rows + 0.5) / raster * height_m
        pts = np.stack([x, y], 1)
        inside, bary = _point_in_triangles(pts, tri2)
        hit = inside.any(1)
        first = np.argmax(inside, axis=1)
        sel = np.flatnonzero(hit)
        out_f.append(fidx[first[sel]])
        out_b.append(bary[sel, first[sel]])
        if side < 0:
            n_back = len(sel)
    if not out_f:
        return Decal(np.zeros(0, np.int64), np.zeros((0, 3)), 0)
    return Decal(np.concatenate(out_f), np.concatenate(out_b), n_back)


def decal_points(decal: Decal, vertices, faces, *, lift_m: float = DECAL_LIFT_M):
    """``(xyz [M, 3], normals [M, 3])`` of the decal on the posed body."""
    from nfl_gsplat.compositing.mesh_to_gaussians import vertex_normals

    v = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if not len(decal.faces):
        return np.zeros((0, 3)), np.zeros((0, 3))
    tri = v[faces[decal.faces]]                                          # [M, 3, 3]
    xyz = (decal.bary[:, :, None] * tri).sum(1)
    vn = vertex_normals(v, faces)
    n = (decal.bary[:, :, None] * vn[faces[decal.faces]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return xyz + lift_m * n, n


def decal_gaussians(decal: Decal, vertices, faces, colour, *, sigma_m: float = DECAL_SIGMA_M,
                    lift_m: float = DECAL_LIFT_M):
    """A GaussianBatch of the decal on the posed body, or None when empty."""
    from nfl_gsplat.compositing.mesh_to_gaussians import GaussianBatch, _SH_C0, _quat_from_z_to

    xyz, n = decal_points(decal, vertices, faces, lift_m=lift_m)
    m = len(xyz)
    if not m:
        return None
    scale = np.tile(np.log([sigma_m, sigma_m, 0.25 * sigma_m]).astype(np.float32), (m, 1))
    alpha = 0.995
    rgb = np.tile(np.asarray(colour, np.float32).reshape(1, 3), (m, 1))
    return GaussianBatch(
        xyz=xyz.astype(np.float32),
        rot=_quat_from_z_to(n).astype(np.float32),
        scale=scale,
        opacity=np.full(m, np.log(alpha / (1 - alpha)), np.float32),
        sh=((rgb - 0.5) / _SH_C0)[:, :, None].astype(np.float32),
        sh_degree=0,
    )


def dress(masks: dict[str, np.ndarray], kit: Kit) -> np.ndarray:
    """``[V, 3]`` colours for the regions under ``kit``."""
    n = len(next(iter(masks.values())))
    out = np.zeros((n, 3), np.float32)
    paint = {"helmet": kit.helmet, "jersey": kit.jersey, "skin": SKIN_RGB, "gloves": kit.gloves,
             "pants": kit.pants, "socks": kit.socks, "shoes": SHOE_RGB}
    for name, m in masks.items():
        out[m] = np.asarray(paint[name], np.float32)
    return out
