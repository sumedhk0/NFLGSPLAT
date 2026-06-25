"""Robust field-line labeling via planar-homography consensus.

The per-camera hint pins one detected yard line's absolute yardage (the anchor).
Real yard lines are monotonic in world-X; the two hash rows are world Y = ±HASH.
A correct labeling makes every (yard-line × hash-row) point consistent with one
field→image homography; spurious lines (painted numbers, jersey scraps) fit no
consensus and are rejected. Hypotheses are enumerated deterministically — the
space is tiny, so no randomness is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.field_features import landmark_name
from nfl_gsplat.calibration.field_identify import _seg_intersection, _yard_step, line_x_at
from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, YARD_LINE_SPACING_M


@dataclass(frozen=True)
class LabelResult:
    correspondences: list[tuple[str, tuple[float, float]]]
    homography: np.ndarray | None
    inlier_count: int


def fit_plane_homography(world_xy, image_uv) -> np.ndarray | None:
    """Least-squares field→image homography (3×3) from ≥4 point pairs, else None."""
    import cv2
    world = np.asarray(world_xy, dtype=np.float64)
    image = np.asarray(image_uv, dtype=np.float64)
    if len(world) < 4 or len(image) != len(world):
        return None
    H, _ = cv2.findHomography(world, image, 0)
    return H


def _in_image(p, W, H) -> bool:
    return p is not None and 0 <= p[0] <= W and 0 <= p[1] <= H


def label_lines_by_consensus(
    lines, hash_rows, *, anchor_idx, anchor_world_x, anchor_side, anchor_yard,
    direction, image_size, inlier_px: float = 5.0, max_offset: int = 12,
) -> LabelResult:
    """Label detected yard lines by homography consensus; reject noise lines.

    Each kept line contributes 2 field↔image points (its intersections with the
    two hash rows at world Y = ±HASH_OFFSET_M). Enumerate hypotheses (anchor +
    one other line at a signed yard-line offset), fit a homography, score by how
    many kept lines map onto valid yard-line positions, keep the best, refit, and
    emit (landmark_name, uv) correspondences. ``direction`` (+1/−1) pins the
    image-x→yard orientation (the hint's ``increasing``)."""
    import cv2
    W, H = image_size
    if len(hash_rows) < 2:
        return LabelResult([], None, 0)
    row_top, row_bot = hash_rows[0], hash_rows[1]
    mid = H / 2.0

    kept = []  # (x_mid, p_top, p_bot)
    for seg in lines:
        pt = _seg_intersection(seg, row_top)
        pb = _seg_intersection(seg, row_bot)
        if _in_image(pt, W, H) and _in_image(pb, W, H):
            kept.append((line_x_at(seg, mid), pt, pb))
    if len(kept) < 2:
        return LabelResult([], None, 0)
    kept.sort(key=lambda r: r[0])

    anchor_x0 = line_x_at(lines[anchor_idx], mid)
    a = min(range(len(kept)), key=lambda i: abs(kept[i][0] - anchor_x0))
    ax, a_top, a_bot = kept[a]
    Hp = HASH_OFFSET_M

    def homog_for(world_x_j, j):
        world = np.array([[anchor_world_x, Hp], [anchor_world_x, -Hp],
                          [world_x_j, Hp], [world_x_j, -Hp]], np.float64)
        image = np.array([a_top, a_bot, kept[j][1], kept[j][2]], np.float64)
        return fit_plane_homography(world, image)

    def score(Hm):
        try:
            Hinv = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return []
        inliers = []
        for i, (xm, pt, pb) in enumerate(kept):
            img = np.array([[pt], [pb]], np.float64)
            fld = cv2.perspectiveTransform(img, Hinv).reshape(2, 2)
            avg_x = 0.5 * (fld[0, 0] + fld[1, 0])
            k = int(round((avg_x - anchor_world_x) / YARD_LINE_SPACING_M))
            if abs(k) > max_offset:
                continue
            x_snap = anchor_world_x + k * YARD_LINE_SPACING_M
            snap = np.array([[[x_snap, Hp]], [[x_snap, -Hp]]], np.float64)
            rep = cv2.perspectiveTransform(snap, Hm).reshape(2, 2)
            resid = max(np.linalg.norm(rep[0] - pt), np.linalg.norm(rep[1] - pb))
            if resid <= inlier_px:
                inliers.append((i, k, pt, pb, resid))
        return inliers

    best = []
    best_key = (-1, float("inf"), float("inf"))
    for j in range(len(kept)):
        if j == a:
            continue
        side = 1 if kept[j][0] > ax else -1
        for d in range(1, max_offset + 1):
            k = direction * side * d
            world_x_j = anchor_world_x + k * YARD_LINE_SPACING_M
            Hm = homog_for(world_x_j, j)
            if Hm is None:
                continue
            inl = score(Hm)
            max_abs_k = max((abs(r[1]) for r in inl), default=0)
            total = sum(r[4] for r in inl)
            key = (len(inl), -max_abs_k, -total)
            if key > (best_key[0], -best_key[1], -best_key[2]):
                best, best_key = inl, (len(inl), max_abs_k, total)
    if len(best) < 2:
        return LabelResult([], None, 0)

    wpts, ipts, labels = [], [], []
    for (_i, k, pt, pb, _r) in best:
        side, yard = _yard_step(anchor_side, anchor_yard, k)
        if side == "":
            continue
        x_snap = anchor_world_x + k * YARD_LINE_SPACING_M
        wpts += [[x_snap, Hp], [x_snap, -Hp]]
        ipts += [pt, pb]
        labels.append((side, yard, pt, pb))
    H_refit = fit_plane_homography(np.array(wpts), np.array(ipts))
    corrs, seen = [], set()
    for (side, yard, pt, pb) in labels:
        for lr, p in (("left", pt), ("right", pb)):
            name = landmark_name(side, yard, lr, "hash")
            if name not in seen:
                seen.add(name)
                corrs.append((name, (float(p[0]), float(p[1]))))
    return LabelResult(corrs, H_refit, len(labels))
