"""Pseudo-labels for a 2D keypoint fine-tune on one play: the fused SMPL-X fit's joints reprojected into each
camera, kept where a camera anchors them.

WHY. The odd arm angles are not at unseen joints (the in-filler's verdict, 2026-09-23): they are the detector's --
a COCO-trained pose model reading padded, helmeted men in piles, weakest on the arm away from the sideline camera.
The play carries a signal the detector never had: a joint the sideline missed but the endzone saw sits in the
two-view fit, and its reprojection into the sideline image is a label in exactly the appearance the detector failed
on. Fine-tuning the detector on this play's own crops with those labels is the lever; overfitting to the play is
the point (a per-play model, retrained per play).

WHAT. Per (camera, frame, player) with a fused refit record, the 12 COCO limb keypoints get the fit's projected
point when the joint is ANCHORED in either camera -- that camera's detector saw it with confidence >= ANCHOR_CONF
within AGREE_PX of the projection (the fit agrees with a confident detection there, so the 3D joint is trusted). A
label anchored only by the other camera is the new signal (``fit_cross``); one anchored by this camera restates the
detector (``fit_self``). Where nothing anchors the joint, the detector's own point stands if confident (``det``),
else the keypoint is unlabelled (visibility 0, no loss). The face keypoints have no body joint and take the
detector's own. The yaw trap: a body fitted 90 degrees wrong in yaw reprojects fine in one view and badly in the
other; single-view agreement is why the anchor asks the detector, never the fit alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.pose.coco import COCO_FACE, COCO_TO_SMPLX

AGREE_PX: float = 12.0        # a projection this close to a confident detection anchors the joint in that camera
ANCHOR_CONF: float = 0.5      # ... with the detection at least this confident
N_COCO: int = 17
COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left <-> right for horizontal flips
SOURCES = ("none", "det", "fit_self", "fit_cross")


@dataclass
class FrameLabels:
    uv: dict            # cam -> [17, 2] pixels (nan where unlabelled)
    vis: dict           # cam -> [17] 2 = labelled, 0 = not
    source: dict        # cam -> [17] index into SOURCES


def label_frame(proj: dict, det: dict, *, agree_px: float = AGREE_PX, anchor_conf: float = ANCHOR_CONF) -> FrameLabels:
    """``proj``: cam -> [17, 2] the fit's projection per COCO keypoint (nan for the face and where no camera pose);
    ``det``: cam -> (xy [17, 2], conf [17]) the detector's keypoints, absent cameras missing. Returns the labels per
    camera by the rule in the module docstring. Cameras present in ``det`` but not in ``proj`` get detector labels
    only."""
    cams = sorted(set(proj) | set(det))
    anchored = {}
    for c in cams:
        a = np.zeros(N_COCO, bool)
        if c in proj and c in det:
            xy, conf = det[c]
            p = np.asarray(proj[c], float)
            d = np.linalg.norm(p - np.asarray(xy, float), axis=1)
            a = np.isfinite(d) & (np.asarray(conf, float) >= anchor_conf) & (d <= agree_px)
        anchored[c] = a
    uv, vis, src = {}, {}, {}
    for c in cams:
        u = np.full((N_COCO, 2), np.nan)
        v = np.zeros(N_COCO, int)
        s = np.zeros(N_COCO, int)
        others = [o for o in cams if o != c]
        p = np.asarray(proj[c], float) if c in proj else np.full((N_COCO, 2), np.nan)
        xy, conf = det[c] if c in det else (np.full((N_COCO, 2), np.nan), np.zeros(N_COCO))
        xy, conf = np.asarray(xy, float), np.asarray(conf, float)
        for k in range(N_COCO):
            body = k in COCO_TO_SMPLX
            if body and np.isfinite(p[k]).all() and anchored[c][k]:
                u[k], v[k], s[k] = p[k], 2, SOURCES.index("fit_self")
            elif body and np.isfinite(p[k]).all() and any(anchored[o][k] for o in others):
                u[k], v[k], s[k] = p[k], 2, SOURCES.index("fit_cross")
            elif conf[k] >= anchor_conf and np.isfinite(xy[k]).all():
                u[k], v[k], s[k] = xy[k], 2, SOURCES.index("det")
        uv[c], vis[c], src[c] = u, v, s
    return FrameLabels(uv, vis, src)


def detector_labels(xy, conf, *, anchor_conf: float = ANCHOR_CONF, min_joints: int = 4):
    """Labels for a tracked box WITHOUT a fit record: the detector's own keypoints at confidence >= ``anchor_conf``
    (visibility 2, source ``det``), the rest unlabelled; ``None`` when fewer than ``min_joints`` qualify -- the
    instance is then left out of the label file altogether. Never a box with all 17 keypoints at visibility 0:
    ultralytics' keypoint-objectness loss trains the confidence head toward 0 on every unlabelled joint, and a
    dataset that was 54 % such boxes (2026-09-24) taught the detector that endzone men have no visible joints
    (median confidence 0.95 -> 0.01)."""
    xy, conf = np.asarray(xy, float), np.asarray(conf, float)
    ok = (conf >= anchor_conf) & np.isfinite(xy).all(1)
    if int(ok.sum()) < int(min_joints):
        return None
    u = np.where(ok[:, None], xy, np.nan)
    v = np.where(ok, 2, 0).astype(int)
    s = np.where(ok, SOURCES.index("det"), 0).astype(int)
    return u, v, s


def coco_projection(joints22: np.ndarray, K, R, t) -> np.ndarray:
    """``[17, 2]`` the fit's body joints projected with ``(K, R, t)`` at the COCO indices that have a body joint;
    nan elsewhere (the face)."""
    from nfl_gsplat.pose.fit_mono2d import project

    pix, z = project(K, R, t, np.asarray(joints22, float))
    out = np.full((N_COCO, 2), np.nan)
    for k, s in COCO_TO_SMPLX.items():
        if z[s] > 0:
            out[k] = pix[s]
    return out


def yolo_pose_line(box_xyxy, uv: np.ndarray, vis: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    """One YOLO-pose label line: class, the box as normalised centre and size, then 17 (x, y, v) normalised; an
    unlabelled keypoint is written as 0 0 0. The box is clipped to the image and a keypoint outside it is written
    unlabelled: ultralytics drops a whole image as corrupt on one negative or out-of-bounds coordinate (20 endzone
    frames of the first dataset, 2026-09-24). ``None`` when the clipped box is empty."""
    x1, y1, x2, y2 = (float(b) for b in box_xyxy)
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    cx, cy, w, h = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    n_lab = 0
    for k in range(N_COCO):
        inside = np.isfinite(uv[k]).all() and 0.0 <= uv[k, 0] <= width and 0.0 <= uv[k, 1] <= height
        if vis[k] > 0 and inside:
            parts += [f"{min(uv[k, 0] / width, 1.0):.6f}", f"{min(uv[k, 1] / height, 1.0):.6f}", str(int(vis[k]))]
            n_lab += 1
        else:
            parts += ["0", "0", "0"]
    if n_lab == 0:                  # every labelled joint fell outside the image: not an instance either
        return None
    return " ".join(parts)


def pck(pred_xy: np.ndarray, label_uv: np.ndarray, vis: np.ndarray, *, thr_px: float = 10.0) -> tuple[int, int]:
    """``(hits, n)`` over the labelled keypoints: a prediction within ``thr_px`` of its label is a hit."""
    m = (np.asarray(vis) > 0) & np.isfinite(np.asarray(label_uv)).all(1) & np.isfinite(np.asarray(pred_xy)).all(1)
    if not m.any():
        return 0, 0
    d = np.linalg.norm(np.asarray(pred_xy, float)[m] - np.asarray(label_uv, float)[m], axis=1)
    return int((d <= thr_px).sum()), int(m.sum())
