"""COCO-17 keypoints (05m, YOLOv8-pose) in SMPL-X body-joint order.

Shared by the two-view triangulation (05n) and the one-view refit (05p) so
both read the detector the same way. 8 of the 22 SMPL-X body joints (spine
x3, collars x2, feet x2, and the head only through the face) have no COCO
keypoint: whatever consumes this sees at most 14 of 22 joints.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS

# COCO-17 index -> SMPL-X body joint (pose.forward_kinematics tree)
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
                 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_L_HIP, COCO_R_HIP, COCO_L_SHO, COCO_R_SHO = 11, 12, 5, 6
COCO_FACE = (0, 1, 2, 3, 4)
SMPLX_PELVIS, SMPLX_NECK, SMPLX_HEAD = 0, 12, 15


def coco_to_body(xy, conf, *, min_conf: float):
    """``(uv [22, 2], conf [22])`` in SMPL-X body order; 0 / 0 where unknown
    (the triangulator's DLT needs finite pixels; conf 0 fails its gate)."""
    xy = np.asarray(xy, float)
    conf = np.asarray(conf, float)
    uv = np.zeros((NUM_BODY_JOINTS, 2))
    c = np.zeros(NUM_BODY_JOINTS)
    ok = conf >= min_conf
    for k, s in COCO_TO_SMPLX.items():
        if ok[k]:
            uv[s], c[s] = xy[k], conf[k]
    if ok[COCO_L_HIP] and ok[COCO_R_HIP]:
        uv[SMPLX_PELVIS] = 0.5 * (xy[COCO_L_HIP] + xy[COCO_R_HIP])
        c[SMPLX_PELVIS] = min(conf[COCO_L_HIP], conf[COCO_R_HIP])
    if ok[COCO_L_SHO] and ok[COCO_R_SHO]:
        uv[SMPLX_NECK] = 0.5 * (xy[COCO_L_SHO] + xy[COCO_R_SHO])
        c[SMPLX_NECK] = min(conf[COCO_L_SHO], conf[COCO_R_SHO])
    face = [k for k in COCO_FACE if ok[k]]
    if len(face) >= 2:
        uv[SMPLX_HEAD] = xy[face].mean(0)
        c[SMPLX_HEAD] = float(np.mean(conf[face]))
    return uv, c
