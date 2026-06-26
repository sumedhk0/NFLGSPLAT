"""Landmark inference: heatmaps → (name, uv, conf) in source image coords."""
from __future__ import annotations

from nfl_gsplat.landmarks.heatmap import extract_peak


def detect_landmarks(heatmaps, schema, *, src_hw, in_hw, heat_stride, conf_thresh=0.5):
    """Per-class peak → source-image (u,v). ``heatmaps`` is (K,Hh,Ww) numpy."""
    src_h, src_w = src_hw
    in_h, in_w = in_hw
    sx = src_w / in_w * heat_stride
    sy = src_h / in_h * heat_stride
    names = schema.class_names()
    out = []
    for k, name in enumerate(names):
        got = extract_peak(heatmaps[k], thresh=conf_thresh)
        if got is None:
            continue
        (u, v), conf = got
        out.append((name, (u * sx, v * sy), float(conf)))
    return out


def landmarks_to_correspondences(detections, schema):
    """Drop confidence → [(name, (u,v))] for solve_pnp / fit_plane_homography."""
    return [(name, uv) for (name, uv, _conf) in detections]


def run_model(model, bgr, *, in_hw):                     # pragma: no cover (gpu path)
    """Forward a BGR frame through the model → (K,Hh,Ww) numpy heatmaps."""
    import cv2
    import numpy as np
    import torch
    in_h, in_w = in_hw
    img = cv2.resize(bgr, (in_w, in_h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy((img.astype(np.float32) / 255.0).transpose(2, 0, 1))[None]
    model.eval()
    with torch.no_grad():
        y = model(x.to(next(model.parameters()).device))
    return y[0].cpu().numpy()
