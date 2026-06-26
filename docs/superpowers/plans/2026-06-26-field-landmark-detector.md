# Learned Field-Landmark Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a keypoint heatmap network that detects named NFL field landmarks (including number anchors that give vertical spread) from a frame, feeding labeled, well-conditioned correspondences into the existing homography/PnP path.

**Architecture:** New `nfl_gsplat/landmarks/` package: a landmark schema (extends `field_landmarks` with number anchors), a heatmap dataset built from hand-clicked points, a compact UNet keypoint net, GPU training, and inference that emits `(name, uv, conf)`. A multi-frame labeling GUI (extends `annotate_gui`) produces the data; `02_autocalibrate` gains a learned mode. The field-overlay diagnostic is the acceptance test.

**Tech Stack:** Python 3.10, PyTorch (already in stack), numpy, OpenCV, pytest. No mmpose. GPU training via SLURM on PACE.

**Reference spec:** `docs/superpowers/specs/2026-06-26-field-landmark-detector-design.md`

## Global Constraints
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`. Local `python -m pytest`.
- **All GPU jobs run on PACE Phoenix `embers` partition** (`--partition=embers`, account `paceship-pso`, GPU gres). embers is preemptible → GPU code MUST checkpoint every epoch and resume from the latest checkpoint.
- Field constants (`field_landmarks`): `HALF_WIDTH_M=24.384`, `HASH_OFFSET_M=2.8194`, `YARD_TO_M=0.9144`. Number geometry (NFL rulebook): bottom of numbers 12 yd from sideline → `NUMBER_BOTTOM_Y_M = HALF_WIDTH_M - 12*YARD_TO_M = 13.4112`; numbers 6 ft tall → `NUMBER_TOP_Y_M = NUMBER_BOTTOM_Y_M + 6*0.3048 = 15.24`. Numbers painted only at yards {10,20,30,40} and mid_50.
- Fail loud with `SetupError`/`CalibrationError` + a pointer; never silently fabricate calibration.
- GPU/slow tests gated behind `@pytest.mark.gpu` / `@pytest.mark.slow`; the default suite (`-m "not gpu and not slow and not real_video"`) stays green and CPU-only.

---

## Task 1: Number-anchor landmarks + schema

**Files:**
- Modify: `nfl_gsplat/calibration/field_landmarks.py`
- Create: `nfl_gsplat/landmarks/__init__.py`, `nfl_gsplat/landmarks/schema.py`
- Test: `tests/test_landmark_schema.py`

**Interfaces:**
- Produces: `field_landmarks` constants `NUMBER_BOTTOM_Y_M`, `NUMBER_TOP_Y_M` and new `NFL_LANDMARKS` entries `{yl}_{left|right}_number_{top|bottom}` for yl ∈ {away_10,away_20,away_30,away_40,mid_50,home_40,home_30,home_20,home_10}. `schema.LandmarkSchema(yard_min, yard_max).class_names() -> list[str]`, `.world_xyz(name) -> np.ndarray(3,)`, `.index(name) -> int`, `.num_classes -> int`.

- [ ] **Step 1: Write the failing test** `tests/test_landmark_schema.py`:

```python
import numpy as np


def test_number_anchor_landmarks_exist_with_correct_Y():
    from nfl_gsplat.calibration.field_landmarks import (
        NFL_LANDMARKS, NUMBER_BOTTOM_Y_M, NUMBER_TOP_Y_M, _yardline_x_m,
    )
    assert abs(NUMBER_BOTTOM_Y_M - 13.4112) < 1e-6
    assert abs(NUMBER_TOP_Y_M - 15.24) < 1e-6
    p = NFL_LANDMARKS["away_30_left_number_bottom"]
    assert np.allclose(p, [_yardline_x_m("away_30"), +NUMBER_BOTTOM_Y_M, 0.0])
    p = NFL_LANDMARKS["home_20_right_number_top"]
    assert np.allclose(p, [_yardline_x_m("home_20"), -NUMBER_TOP_Y_M, 0.0])
    # numbers only at 10/20/30/40 + mid_50, not at 5/15/25/35/45
    assert "away_25_left_number_top" not in NFL_LANDMARKS
    assert "mid_50_left_number_top" in NFL_LANDMARKS


def test_schema_classes_scoped_and_indexable():
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)   # near midfield, meters in X
    names = s.class_names()
    assert s.num_classes == len(names) == len(set(names))      # unique, ordered
    assert s.index(names[0]) == 0 and s.index(names[-1]) == s.num_classes - 1
    # every class has a 3-vector world point and lies within the X window
    for n in names:
        xyz = s.world_xyz(n)
        assert xyz.shape == (3,) and -20.0 <= xyz[0] <= 20.0
    # includes a number anchor and a hash within the window
    assert any("number" in n for n in names) and any("hash" in n for n in names)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_landmark_schema.py -q` → FAIL.

- [ ] **Step 3a: Edit `field_landmarks.py`.** Add constants after `YARD_LINE_SPACING_M`:

```python
NUMBER_BOTTOM_Y_M: float = HALF_WIDTH_M - 12.0 * YARD_TO_M   # 13.4112 (12 yd from sideline)
NUMBER_TOP_Y_M: float = NUMBER_BOTTOM_Y_M + 6.0 * 0.3048     # 15.24  (numbers are 6 ft tall)
```
In `_build_landmarks`, after the yard-line loop, add number anchors:

```python
    # Painted field numbers (only at 10/20/30/40 and mid-50), centered on the yard
    # line. Top/bottom anchors give Y far from the hashes → vertical conditioning.
    number_yls = ["away_10", "away_20", "away_30", "away_40", "mid_50",
                  "home_40", "home_30", "home_20", "home_10"]
    for yl in number_yls:
        x = _yardline_x_m(yl)
        for sgn, lr in [(+1.0, "left"), (-1.0, "right")]:
            lm[f"{yl}_{lr}_number_bottom"] = np.array([x, sgn * NUMBER_BOTTOM_Y_M, 0.0])
            lm[f"{yl}_{lr}_number_top"]    = np.array([x, sgn * NUMBER_TOP_Y_M, 0.0])
```

- [ ] **Step 3b: Create `nfl_gsplat/landmarks/__init__.py`** (empty) and `nfl_gsplat/landmarks/schema.py`:

```python
"""Ordered landmark class schema for the keypoint detector.

The model has one output channel per class; this fixes the class↔index mapping and
restricts classes to a world-X window (the footage's yard range) so K and per-class
data stay tractable (per-dataset model)."""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS


class LandmarkSchema:
    def __init__(self, yard_min: float, yard_max: float) -> None:
        # deterministic, sorted order so class indices are stable across runs
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
```

- [ ] **Step 4: Run** `python -m pytest tests/test_landmark_schema.py -q` → PASS. Also run `tests/test_field_landmarks.py` if present (number anchors are additive; existing landmark tests must still pass).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_landmarks.py nfl_gsplat/landmarks tests/test_landmark_schema.py
git add nfl_gsplat/calibration/field_landmarks.py nfl_gsplat/landmarks tests/test_landmark_schema.py
git commit -m "landmarks: number-anchor world points + LandmarkSchema (X-windowed classes)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Heatmap render + peak extraction utils

**Files:**
- Create: `nfl_gsplat/landmarks/heatmap.py`
- Test: `tests/test_landmark_heatmap.py`

**Interfaces:**
- Produces: `render_gaussian(hw, uv, sigma) -> np.ndarray(H,W) float32`; `extract_peak(heat, *, thresh) -> tuple[(u,v), conf] | None` (subpixel via local centroid).

- [ ] **Step 1: Write the failing test**:

```python
import numpy as np


def test_render_gaussian_peaks_at_uv():
    from nfl_gsplat.landmarks.heatmap import render_gaussian
    h = render_gaussian((40, 60), (30.0, 20.0), sigma=2.0)   # (H=40,W=60), uv=(x=30,y=20)
    assert h.shape == (40, 60) and h.dtype == np.float32
    iy, ix = np.unravel_index(int(np.argmax(h)), h.shape)
    assert (ix, iy) == (30, 20)
    assert abs(h.max() - 1.0) < 1e-5


def test_extract_peak_subpixel_and_threshold():
    from nfl_gsplat.landmarks.heatmap import extract_peak, render_gaussian
    h = render_gaussian((40, 60), (30.4, 20.0), sigma=2.0)
    got = extract_peak(h, thresh=0.3)
    assert got is not None
    (u, v), conf = got
    assert abs(u - 30.4) < 0.5 and abs(v - 20.0) < 0.5 and conf > 0.9
    # flat/empty heatmap → None
    assert extract_peak(np.zeros((40, 60), np.float32), thresh=0.3) is None
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/heatmap.py`**:

```python
"""Gaussian heatmap rendering (training targets) + subpixel peak extraction."""
from __future__ import annotations

import numpy as np


def render_gaussian(hw, uv, sigma: float) -> np.ndarray:
    """(H,W) float32 heatmap, peak 1.0 at ``uv=(x,y)`` (image coords in heatmap res)."""
    h, w = hw
    x, y = uv
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def extract_peak(heat, *, thresh: float):
    """Argmax + 3×3 centroid subpixel refine. Returns ((u,v), conf) or None."""
    h, w = heat.shape
    idx = int(np.argmax(heat))
    iy, ix = divmod(idx, w)
    conf = float(heat[iy, ix])
    if conf < thresh:
        return None
    x0, x1 = max(0, ix - 1), min(w, ix + 2)
    y0, y1 = max(0, iy - 1), min(h, iy + 2)
    patch = heat[y0:y1, x0:x1].astype(np.float64)
    s = patch.sum()
    if s <= 1e-9:
        return (float(ix), float(iy)), conf
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = float((xx * patch).sum() / s)
    v = float((yy * patch).sum() / s)
    return (u, v), conf
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/heatmap.py tests/test_landmark_heatmap.py
git add nfl_gsplat/landmarks/heatmap.py tests/test_landmark_heatmap.py
git commit -m "landmarks: gaussian heatmap render + subpixel peak extraction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Dataset

**Files:**
- Create: `nfl_gsplat/landmarks/dataset.py`
- Test: `tests/test_landmark_dataset.py`

**Interfaces:**
- Consumes: `LandmarkSchema` (Task 1), `render_gaussian` (Task 2).
- Produces: label-file format = JSON `{"image_size":[W,H], "frames":[{"file":"<png>", "points":[{"name":str,"uv":[u,v]}]}]}`. `LandmarkDataset(label_json, frames_dir, schema, *, in_hw=(540,960), heat_stride=4, sigma=2.0, augment=False)`; `__len__`, `__getitem__(i) -> (image CHW float32 tensor, heatmaps (K,Hh,Ww) float32, vis_mask (K,) float32)`.

- [ ] **Step 1: Write the failing test** (synthetic PNGs + JSON in tmp_path — no real video):

```python
import json
import numpy as np


def _make_dataset(tmp_path):
    import cv2
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    frames_dir = tmp_path / "frames"; frames_dir.mkdir()
    names = [n for n in sorted(NFL_LANDMARKS) if -20 <= NFL_LANDMARKS[n][0] <= 20][:3]
    frames = []
    for fi in range(2):
        img = np.full((1080, 1920, 3), 60, np.uint8)
        cv2.imwrite(str(frames_dir / f"f{fi}.png"), img)
        pts = [{"name": names[0], "uv": [960.0, 540.0]},
               {"name": names[1], "uv": [400.0, 300.0]}]   # names[2] absent → invisible
        frames.append({"file": f"f{fi}.png", "points": pts})
    label = {"image_size": [1920, 1080], "frames": frames}
    p = tmp_path / "labels.json"; p.write_text(json.dumps(label))
    return p, frames_dir, names


def test_dataset_shapes_and_visibility(tmp_path):
    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    label, frames_dir, names = _make_dataset(tmp_path)
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    ds = LandmarkDataset(label, frames_dir, s, in_hw=(540, 960), heat_stride=4)
    assert len(ds) == 2
    img, heat, vis = ds[0]
    assert img.shape == (3, 540, 960)
    assert heat.shape == (s.num_classes, 540 // 4, 960 // 4)
    # visible classes have a hot pixel; invisible have flat-zero + vis 0
    assert vis[s.index(names[0])] == 1.0 and heat[s.index(names[0])].max() > 0.9
    assert vis[s.index(names[2])] == 0.0 and float(heat[s.index(names[2])].max()) == 0.0


def test_dataset_scales_uv_to_input_then_heatmap(tmp_path):
    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    label, frames_dir, names = _make_dataset(tmp_path)
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    ds = LandmarkDataset(label, frames_dir, s, in_hw=(540, 960), heat_stride=4)
    _, heat, _ = ds[0]
    ch = heat[s.index(names[0])]                       # uv (960,540) full → heatmap center
    iy, ix = np.unravel_index(int(ch.argmax()), ch.shape)
    assert abs(ix - (960 / 4) / 2) <= 1 and abs(iy - (540 / 4) / 2) <= 1
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/dataset.py`**:

```python
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
            if name not in self.schema._index:          # outside this schema window
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
    img *= np.random.uniform(0.7, 1.3)                  # brightness
    img = np.clip(img, 0, 255)
    if np.random.rand() < 0.3:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img.astype(np.uint8)
```

(Augmentation is color-only here; geometric augmentation is deferred — keep the uv↔heatmap mapping exact for now. The test runs with `augment=False`.)

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/dataset.py tests/test_landmark_dataset.py
git add nfl_gsplat/landmarks/dataset.py tests/test_landmark_dataset.py
git commit -m "landmarks: heatmap dataset from clicked labels (PNG frames + JSON)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Model

**Files:**
- Create: `nfl_gsplat/landmarks/model.py`
- Test: `tests/test_landmark_model.py`

**Interfaces:**
- Produces: `LandmarkNet(num_classes, *, stride=4)`; `forward(x: (N,3,H,W)) -> (N,K,H//stride,W//stride)`, sigmoid-activated (0..1).

- [ ] **Step 1: Write the failing test** (mark `slow` — builds a torch module but runs on CPU):

```python
import pytest


@pytest.mark.slow
def test_landmark_net_forward_shape():
    import torch
    from nfl_gsplat.landmarks.model import LandmarkNet
    net = LandmarkNet(num_classes=7, stride=4)
    x = torch.zeros(2, 3, 540, 960)
    y = net(x)
    assert y.shape == (2, 7, 135, 240)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
```

- [ ] **Step 2: Run** `python -m pytest tests/test_landmark_model.py -q -m slow` → FAIL.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/model.py`** (compact UNet-style encoder/decoder to ¼ res):

```python
"""Compact UNet keypoint heatmap network (one sigmoid channel per landmark)."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LandmarkNet(nn.Module):
    def __init__(self, num_classes: int, *, stride: int = 4):
        super().__init__()
        assert stride == 4, "decoder returns 1/4 resolution"
        self.e1 = _block(3, 32)
        self.e2 = _block(32, 64)
        self.e3 = _block(64, 128)
        self.e4 = _block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = _block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = _block(128, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)                 # full
        e2 = self.e2(self.pool(e1))     # 1/2
        e3 = self.e3(self.pool(e2))     # 1/4
        e4 = self.e4(self.pool(e3))     # 1/8
        d3 = self.d3(torch.cat([self.up3(e4), e3], 1))   # 1/4
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))   # 1/2
        # return at 1/4 res: downsample d2 by 2 via stride-2 conv-free avgpool
        out = self.head(nn.functional.avg_pool2d(d2, 2))  # 1/4
        return torch.sigmoid(out)
```

- [ ] **Step 4: Run** `python -m pytest tests/test_landmark_model.py -q -m slow` → PASS (135 = 540/4, 240 = 960/4). If the spatial size is off by ±1 due to pooling on odd sizes, pad inputs to multiples of 8 in the test/model — 540 and 960 are both divisible by 4 but 540/8 is not integer; use input 544×960 OR add `nn.functional.interpolate(out, size=(H//4, W//4))` at the end to force exact shape. Prefer the explicit interpolate to `(x.shape[2]//4, x.shape[3]//4)` for robustness.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/model.py tests/test_landmark_model.py
git add nfl_gsplat/landmarks/model.py tests/test_landmark_model.py
git commit -m "landmarks: compact UNet keypoint heatmap model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Inference → correspondences

**Files:**
- Create: `nfl_gsplat/landmarks/infer.py`
- Test: `tests/test_landmark_infer.py`

**Interfaces:**
- Consumes: `LandmarkSchema`, `extract_peak`, `LandmarkNet`.
- Produces: `landmarks_to_correspondences(detections, schema) -> list[(name, (u,v))]`; `detect_landmarks(heatmaps_np, schema, *, src_hw, in_hw, heat_stride, conf_thresh=0.5) -> list[(name,(u,v),conf)]` (maps peaks back to SOURCE image coords). A thin `run_model(model, bgr, *, in_hw) -> heatmaps_np` wraps the torch forward (gpu-gated, not unit-tested).

- [ ] **Step 1: Write the failing test** (drive `detect_landmarks` with synthetic heatmaps — no torch):

```python
import numpy as np


def test_detect_landmarks_maps_peaks_to_source_coords():
    from nfl_gsplat.landmarks.heatmap import render_gaussian
    from nfl_gsplat.landmarks.infer import detect_landmarks, landmarks_to_correspondences
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    K = s.num_classes
    hh, ww = 135, 240
    heat = np.zeros((K, hh, ww), np.float32)
    # put a peak for class 0 at heatmap (ix=120, iy=67) → source coords
    heat[0] = render_gaussian((hh, ww), (120.0, 67.0), 2.0)
    dets = detect_landmarks(heat, s, src_hw=(1080, 1920), in_hw=(540, 960),
                            heat_stride=4, conf_thresh=0.5)
    assert len(dets) == 1
    name, (u, v), conf = dets[0]
    assert name == s.class_names()[0] and conf > 0.9
    # heatmap (120,67) → input (480,268) → source (×2) ≈ (960,536)
    assert abs(u - 960.0) < 3 and abs(v - 536.0) < 3
    corrs = landmarks_to_correspondences(dets, s)
    assert corrs == [(name, (u, v))]
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/infer.py`**:

```python
"""Landmark inference: heatmaps → (name, uv, conf) in source image coords."""
from __future__ import annotations

from nfl_gsplat.landmarks.heatmap import extract_peak


def detect_landmarks(heatmaps, schema, *, src_hw, in_hw, heat_stride, conf_thresh=0.5):
    """Per-class peak → source-image (u,v). ``heatmaps`` is (K,Hh,Ww) numpy."""
    src_h, src_w = src_hw
    in_h, in_w = in_hw
    sx = src_w / in_w * heat_stride          # heatmap px → source px
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
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/infer.py tests/test_landmark_infer.py
git add nfl_gsplat/landmarks/infer.py tests/test_landmark_infer.py
git commit -m "landmarks: inference (heatmaps→named source-coord detections + correspondences)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Training (PACE embers GPU) + tiny CPU smoke

**Files:**
- Create: `nfl_gsplat/landmarks/train.py`, `scripts/train_landmarks.sbatch`
- Test: `tests/test_landmark_train_smoke.py`

**Interfaces:**
- Consumes: all of Tasks 1–4.
- Produces: `train(label_json, frames_dir, schema, *, out_dir, epochs, batch_size, lr, device, resume=True) -> Path` (best checkpoint). Checkpoints `{out_dir}/ckpt_last.pt` every epoch (resume-safe for embers preemption) + `ckpt_best.pt`.

- [ ] **Step 1: Write the failing smoke test** (`slow`; overfit 2 synthetic frames a few epochs on CPU, assert loss drops):

```python
import json
import numpy as np
import pytest


@pytest.mark.slow
def test_train_overfits_tiny(tmp_path):
    import cv2
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    from nfl_gsplat.landmarks.train import train
    frames_dir = tmp_path / "frames"; frames_dir.mkdir()
    names = [n for n in sorted(NFL_LANDMARKS) if -20 <= NFL_LANDMARKS[n][0] <= 20][:3]
    recs = []
    for fi in range(2):
        cv2.imwrite(str(frames_dir / f"f{fi}.png"), np.full((1080, 1920, 3), 60, np.uint8))
        recs.append({"file": f"f{fi}.png",
                     "points": [{"name": names[0], "uv": [960.0, 540.0]}]})
    label = tmp_path / "labels.json"
    label.write_text(json.dumps({"image_size": [1920, 1080], "frames": recs}))
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    out = tmp_path / "run"
    ck = train(label, frames_dir, s, out_dir=out, epochs=3, batch_size=2, lr=1e-3,
               device="cpu", resume=False)
    assert ck.exists() and (out / "ckpt_last.pt").exists()
    # resume path: a second call with resume=True loads ckpt_last and runs
    ck2 = train(label, frames_dir, s, out_dir=out, epochs=4, batch_size=2, lr=1e-3,
                device="cpu", resume=True)
    assert ck2.exists()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_landmark_train_smoke.py -q -m slow` → FAIL.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/train.py`**:

```python
"""Heatmap keypoint training. GPU jobs run on PACE `embers` (preemptible) → every
epoch is checkpointed and resumable."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _masked_mse(pred, target, vis):
    import torch
    # vis: (N,K) → (N,K,1,1); only supervise channels whose landmark was labeled
    w = vis[:, :, None, None]
    return (w * (pred - target) ** 2).sum() / (w.sum().clamp_min(1.0) * pred.shape[-1] * pred.shape[-2])


def train(label_json, frames_dir, schema, *, out_dir, epochs, batch_size, lr,
          device="cuda", resume=True) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.model import LandmarkNet

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = LandmarkDataset(label_json, frames_dir, schema, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=_collate)
    net = LandmarkNet(schema.num_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start_ep, best = 0, float("inf")
    last = out / "ckpt_last.pt"
    if resume and last.exists():
        st = torch.load(last, map_location=device)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        start_ep, best = st["epoch"] + 1, st["best"]
    for ep in range(start_ep, epochs):
        net.train(); total = 0.0
        for img, heat, vis in dl:
            img, heat, vis = img.to(device), heat.to(device), vis.to(device)
            pred = net(img)
            loss = _masked_mse(pred, heat, vis)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": min(best, total),
                    "classes": schema.class_names()}, last)
        if total < best:
            best = total
            torch.save({"net": net.state_dict(), "classes": schema.class_names()},
                       out / "ckpt_best.pt")
    return out / "ckpt_best.pt"


def _collate(batch):
    import torch
    imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
    heats = torch.from_numpy(np.stack([b[1] for b in batch]))
    vis = torch.from_numpy(np.stack([b[2] for b in batch]))
    return imgs, heats, vis
```

- [ ] **Step 4: Run** `python -m pytest tests/test_landmark_train_smoke.py -q -m slow` → PASS (checkpoints written; resume loads). If model output size ≠ heatmap size, fix Task 4's final interpolate to match `(in_h//4, in_w//4)`.

- [ ] **Step 5: Create `scripts/train_landmarks.sbatch`** (embers + resume-safe):

```bash
#!/bin/bash
#SBATCH --job-name=nfl-landmarks
#SBATCH --partition=embers
#SBATCH --account=paceship-pso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
# embers is preemptible: --requeue + per-epoch checkpoints make the job resume.
module load anaconda3
conda run -n nfl_smplx python -m nfl_gsplat.landmarks.train_cli \
  --label "$1" --frames "$2" --out "$3" \
  --yard-min "${4:--25}" --yard-max "${5:-25}" \
  --epochs "${6:-200}" --batch-size 8 --lr 1e-3 --device cuda --resume
```

Add a `train_cli` `argparse`/`typer` entry in `train.py` (guarded by `if __name__`/a `main()`) that constructs `LandmarkSchema(yard_min, yard_max)` and calls `train(...)`. Confirm it parses: `python -c "import ast; ast.parse(open('nfl_gsplat/landmarks/train.py').read())"`.

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/train.py tests/test_landmark_train_smoke.py
git add nfl_gsplat/landmarks/train.py scripts/train_landmarks.sbatch tests/test_landmark_train_smoke.py
git commit -m "landmarks: training loop + embers SLURM script (resume-safe checkpoints)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Multi-frame labeling tool

**Files:**
- Create: `scripts/label_landmarks.py`
- Modify: `nfl_gsplat/calibration/annotate_gui.py` (extract a reusable single-frame `annotate_frame(img, names, prefill) -> list[{name,uv}]`)
- Test: `tests/test_label_export.py` (test the pure export/sampling helpers, not the GUI)

**Interfaces:**
- Consumes: `LandmarkSchema`; the classical homography (`identify_correspondences`/`fit_plane_homography`) for prefill.
- Produces: a labeling CLI that writes the Task-3 label JSON + extracted frame PNGs. Pure helper `sample_frame_indices(num_frames, count) -> list[int]` and `build_label_record(file, points) -> dict` are unit-tested.

- [ ] **Step 1: Write the failing test** (pure helpers only — GUI is manual):

```python
def test_sample_frame_indices_even_spread():
    from scripts.label_landmarks import sample_frame_indices
    idx = sample_frame_indices(num_frames=1000, count=10)
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 999
    assert all(b > a for a, b in zip(idx, idx[1:]))     # strictly increasing


def test_build_label_record_shape():
    from scripts.label_landmarks import build_label_record
    rec = build_label_record("f0.png", [("away_30_left_hash", (10.0, 20.0))])
    assert rec == {"file": "f0.png",
                   "points": [{"name": "away_30_left_hash", "uv": [10.0, 20.0]}]}
```

- [ ] **Step 2: Run** `python -m pytest tests/test_label_export.py -q` → FAIL (module missing — note `scripts` import: add `scripts/__init__.py` if needed or import via path; the repo already runs scripts by path, so put the pure helpers near the top and ensure `tests` can import them — if `scripts` is not a package, move `sample_frame_indices`/`build_label_record` into `nfl_gsplat/landmarks/labeling.py` and import from there in both the script and the test). **Decision: put the pure helpers in `nfl_gsplat/landmarks/labeling.py`** and have `scripts/label_landmarks.py` import them. Update the test imports to `from nfl_gsplat.landmarks.labeling import sample_frame_indices, build_label_record`.

- [ ] **Step 3: Create `nfl_gsplat/landmarks/labeling.py`**:

```python
"""Pure helpers for the landmark labeling tool (frame sampling + record build)."""
from __future__ import annotations


def sample_frame_indices(num_frames: int, count: int) -> list[int]:
    """Evenly spread ``count`` frame indices over [0, num_frames-1] inclusive."""
    if count <= 1 or num_frames <= 1:
        return [0]
    count = min(count, num_frames)
    step = (num_frames - 1) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def build_label_record(file: str, points) -> dict:
    return {"file": file,
            "points": [{"name": n, "uv": [float(u), float(v)]} for n, (u, v) in points]}
```

- [ ] **Step 4: Run** `python -m pytest tests/test_label_export.py -q` → PASS.

- [ ] **Step 5: Create `scripts/label_landmarks.py`** — the CLI (manual GUI; not unit-tested). It: opens the clip, samples frames via `sample_frame_indices`, extracts each to `<out>/frames/fNNNNN.png`, runs the existing single-frame annotator (refactor `annotate_gui` to expose `annotate_frame(img, schema_names, prefill_uv)`) with classical-homography prefill when available, accumulates `build_label_record`, and writes `<out>/labels.json` (`{"image_size":[W,H],"frames":[...]}`). Refactor `annotate_gui.annotate` to call a new `annotate_frame(...)` so both the keyframe tool and this tool share the click loop. Confirm both parse. (No assertion step — manual tool; the pure helpers carry the tests.)

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/landmarks/labeling.py scripts/label_landmarks.py nfl_gsplat/calibration/annotate_gui.py tests/test_label_export.py
python -c "import ast; ast.parse(open('scripts/label_landmarks.py').read())"
git add nfl_gsplat/landmarks/labeling.py scripts/label_landmarks.py nfl_gsplat/calibration/annotate_gui.py tests/test_label_export.py
git commit -m "landmarks: multi-frame labeling tool (shared annotate_frame + frame sampling)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Learned mode in 02_autocalibrate + docs

**Files:**
- Modify: `scripts/02_autocalibrate.py`, `nfl_gsplat/calibration/run_autocalib.py`
- Modify: `SETUP.md`
- Test: `tests/test_run_autocalib.py` (add a learned-path unit test with a stub detector)

**Interfaces:**
- Consumes: `detect_landmarks`/`run_model` (Task 5), `LandmarkNet` checkpoint, `LandmarkSchema`.
- Produces: `build_autocalib_npz_learned(*, play_dir, videos, fps, model_ckpt, yard_min, yard_max, conf_thresh=0.5, ...)` that, per frame, detects landmarks → correspondences → `register_frame`-style PnP → `assemble_track_from_results`. A `landmarks_provider` seam (callable `frame_bgr -> [(name,(u,v))]`) keeps it unit-testable without torch.

- [ ] **Step 1: Add a learned-path test** to `tests/test_run_autocalib.py` (stub the detector; reuse projected synthetic field):

```python
def test_learned_register_sequence_with_stub_detector():
    import numpy as np
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, NUMBER_BOTTOM_Y_M, _yardline_x_m
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    intr = CameraIntrinsics(1400.0, 1400.0, 960, 540, 1920, 1080)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    pose = CameraPose(R=R, t=np.array([0.0, 6.0, 55.0]))
    pts = {}
    for y in [20, 30, 40]:
        for lr, sgn in (("left", +1), ("right", -1)):
            for tag, Y in (("hash", sgn * HASH_OFFSET_M),
                           ("number_bottom", sgn * NUMBER_BOTTOM_Y_M)):
                name = f"away_{y}_{lr}_{tag}"
                X = _yardline_x_m(f"away_{y}")
                uv = project_points(np.array([[X, Y, 0.0]]), intr.K(), pose.R, pose.t)[0]
                pts[name] = (float(uv[0]), float(uv[1]))

    def stub_detector(frame_bgr):
        return list(pts.items())                          # same labels every frame

    results = ra._register_sequence_learned(
        ["f0", "f1", "f2"], detector=stub_detector, image_size=(1920, 1080))
    assert len(results) == 3 and all(r is not None for r in results)
```

- [ ] **Step 2: Run** → FAIL (`_register_sequence_learned` missing).

- [ ] **Step 3: Add to `run_autocalib.py`**:

```python
def _register_sequence_learned(frames, *, detector, image_size,
                               max_reproj_px=6.0, min_landmarks=6):
    """Per-frame: detector(frame)->[(name,(u,v))] → PnP. No hint/consensus needed
    (the learned detector outputs labeled, well-spread correspondences)."""
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
    from nfl_gsplat.errors import CalibrationError

    results = []
    for fr in frames:
        if fr is None:
            results.append(None); continue
        corrs = detector(fr)
        if len(corrs) < min_landmarks:
            results.append(None); continue
        try:
            results.append(solve_pnp_from_correspondences(
                corrs, image_size=image_size, max_reproj_px=max_reproj_px,
                min_landmarks=min_landmarks))
        except CalibrationError:
            results.append(None)
    return results


def build_autocalib_npz_learned(*, play_dir, videos, fps, model_ckpt, yard_min,
                                yard_max, conf_thresh=0.5, in_hw=(540, 960), heat_stride=4):
    """Learned-mode calibration: a trained LandmarkNet drives per-frame PnP."""
    import torch

    from nfl_gsplat.landmarks.infer import detect_landmarks, landmarks_to_correspondences, run_model
    from nfl_gsplat.landmarks.model import LandmarkNet
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    schema = LandmarkSchema(yard_min=yard_min, yard_max=yard_max)
    st = torch.load(model_ckpt, map_location="cpu")
    net = LandmarkNet(schema.num_classes)
    net.load_state_dict(st["net"])
    tracks = {}
    for cam, video in videos.items():
        meta = ffprobe_meta(video)

        def detector(bgr):
            hm = run_model(net, bgr, in_hw=in_hw)
            dets = detect_landmarks(hm, schema, src_hw=(meta.height, meta.width),
                                    in_hw=in_hw, heat_stride=heat_stride, conf_thresh=conf_thresh)
            return landmarks_to_correspondences(dets, schema)
        frames = [None] * meta.num_frames
        for fidx, fr in iter_frames(video, start_frame=0):
            if 0 <= fidx < meta.num_frames:
                frames[fidx] = fr
        results = _register_sequence_learned(frames, detector=detector,
                                             image_size=(meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width, height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)
```

- [ ] **Step 4: Run** `python -m pytest tests/test_run_autocalib.py -q` → all pass (existing + the learned stub test). The stub projects hashes + number_bottom anchors (vertical spread) → PnP solves cleanly. If `min_landmarks` rejects, confirm the stub emits ≥6 well-spread points.

- [ ] **Step 5: Wire `scripts/02_autocalibrate.py`** — add a `--mode {hint,learned}` flag (default `hint`); in learned mode call `build_autocalib_npz_learned(play_dir=..., videos=..., fps=meta.fps, model_ckpt=<cfg/flag>, yard_min=..., yard_max=...)`. Add `# TODO(bring-up): per-game model_ckpt path + yard window in meta.yaml`. Confirm it parses.

- [ ] **Step 6: Update `SETUP.md`** — document the learned pipeline: (1) `python scripts/label_landmarks.py` to label ~100–150 frames/clip (with classical prefill); (2) submit training on embers: `sbatch scripts/train_landmarks.sbatch <labels.json> <frames_dir> <out_dir> <yard_min> <yard_max>`; (3) `python scripts/02_autocalibrate.py --play-dir <dir> --mode learned`; (4) validate with `scripts/diag_calib.py` field-overlay. Note embers preemption → the job requeues and resumes from `ckpt_last.pt`.

- [ ] **Step 7: Full suite + commit**

```bash
python -m pytest -m "not gpu and not slow and not real_video" -q
python -c "import ast; ast.parse(open('scripts/02_autocalibrate.py').read())"
python -m ruff check nfl_gsplat scripts tests
git add -A
git commit -m "autocalibrate: learned landmark-detector mode + SETUP docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** number-anchor conditioning fix + schema → Task 1; heatmap targets/peaks → Task 2; dataset → Task 3; model → Task 4; inference→correspondences → Task 5; embers GPU training (resume-safe) → Task 6; multi-frame labeling tool (classical prefill, shared annotate_frame) → Task 7; learned mode in `02_autocalibrate` + field-overlay validation in docs + retiring consensus to fallback → Task 8. Failure handling (too-few detections → gap; missing weights → load error) → Tasks 5/8. Testing (CPU unit + slow smoke + real validation via existing diagnostic) → throughout.
- **Type consistency:** `LandmarkSchema(yard_min, yard_max)` w/ `.class_names()/.num_classes/.index/.world_xyz` (Tasks 1,3,5,8); `render_gaussian(hw, uv, sigma)`/`extract_peak(heat,*,thresh)` (Tasks 2,3,5); `LandmarkDataset(...) -> (chw, heat, vis)` (Tasks 3,6); `LandmarkNet(num_classes,*,stride)` (Tasks 4,5,6,8); `detect_landmarks(heatmaps, schema,*,src_hw,in_hw,heat_stride,conf_thresh)` + `landmarks_to_correspondences` + `run_model` (Tasks 5,8); `train(label_json, frames_dir, schema,*,out_dir,epochs,batch_size,lr,device,resume)` (Task 6); `sample_frame_indices`/`build_label_record` in `nfl_gsplat/landmarks/labeling.py` (Task 7); `_register_sequence_learned`/`build_autocalib_npz_learned` (Task 8). Consistent.
- **Placeholder scan:** the only deferred item is geometric augmentation (explicitly noted as out-of-this-version in Task 3), not a placeholder in tested logic.

## Known follow-ups (bring-up + next cycle)
- Label a clip, train on embers, run learned mode, check the field-overlay aligns across the frame (the classical-failure test).
- Geometric augmentation (affine with uv transform) once basic training works.
- Per-game `model_ckpt` + yard window recorded in `meta.yaml`.
- **Next cycle (project memory):** per-frame focal/K,R,t from the now-better-conditioned correspondences.
