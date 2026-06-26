"""Heatmap dataset from hand-clicked landmark labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nfl_gsplat.landmarks.heatmap import render_gaussian


class LandmarkDataset:
    def __init__(self, label_json, frames_dir, schema, *, in_hw=(540, 960),
                 heat_stride=4, sigma=2.0, augment=False):
        self.schema = schema
        self.in_h, self.in_w = in_hw
        self.stride = heat_stride
        self.sigma = sigma
        self.augment = augment
        self.frames_dir = Path(frames_dir)
        data = json.loads(Path(label_json).read_text())
        self.src_w, self.src_h = data["image_size"]
        self.frames = data["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        import cv2
        rec = self.frames[i]
        bgr = cv2.imread(str(self.frames_dir / rec["file"]))
        if bgr is None:
            raise FileNotFoundError(self.frames_dir / rec["file"])
        img = cv2.resize(bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
        sx, sy = self.in_w / self.src_w, self.in_h / self.src_h
        if self.augment:
            img = _augment_color(img)
        chw = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

        K = self.schema.num_classes
        hh, ww = self.in_h // self.stride, self.in_w // self.stride
        heat = np.zeros((K, hh, ww), np.float32)
        vis = np.zeros((K, ), np.float32)
        for pt in rec["points"]:
            name = pt["name"]
            if name not in self.schema._index:
                continue
            k = self.schema.index(name)
            u = pt["uv"][0] * sx / self.stride
            v = pt["uv"][1] * sy / self.stride
            heat[k] = render_gaussian((hh, ww), (u, v), self.sigma)
            vis[k] = 1.0
        return chw, heat, vis


def _augment_color(bgr):
    import cv2
    img = bgr.astype(np.float32)
    img *= np.random.uniform(0.7, 1.3)
    img = np.clip(img, 0, 255)
    if np.random.rand() < 0.3:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img.astype(np.uint8)
