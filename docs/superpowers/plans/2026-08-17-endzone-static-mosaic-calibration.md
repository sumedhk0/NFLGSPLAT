# Endzone Static-Mosaic Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover per-frame endzone cameras accurate to a few pixels by registering every frame to one reference frame, accumulating static field paint there, fitting the metric field model, solving one reference camera, and propagating it through the homographies.

**Architecture:** The endzone camera is a fixed tripod, so frames are related exactly by homographies (measured 0.55 px median). Register each frame directly to a reference frame; warp the player-masked white mask into the reference and accumulate so paint reinforces and transients wash out; fit the metric field to the accumulated paint (labeling pinned by an absolute anchor plus the sideline's yard range); solve one camera; propagate per frame.

**Tech Stack:** Python 3.14, OpenCV 5 (SIFT, `findHomography`), NumPy, SciPy, pytest. Reuses `field_detect`, `player_masks`, `joint_solve.solve_fixed_center`, `EndzonePrior`.

## Global Constraints

- **Repo is now `C:\Users\sumedh\NFLGSPLAT`** (moved off OneDrive). Interpreter: **`C:\venvs\nflgsplat\Scripts\python.exe`** — never the system Python.
- `ffprobe` may not be on PATH; prepend `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0-full_build\bin` when running anything that probes video.
- Do NOT `pip install -e .` (the `numpy<2` pin would break the cu128 torch). Run tests as `python -m pytest` from the repo root.
- **Work in NATIVE endzone pixels**: `view_deg=0`, no view rotation, no roll. The 90° rotation only ever served the dead field-marking detector.
- `utils.video.iter_frames` yields **RGB**; OpenCV expects **BGR**. Convert explicitly (`cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)`) or index `[..., ::-1]`. Getting this wrong silently corrupts colour-based masks.
- Player world points are arbitrary `(X, Y, 0)`, NOT named `NFL_LANDMARKS` — any `solve_fixed_center` call must use `_frame_data_override`, never the named-landmark path.
- Fail loud with `CalibrationError`/`SetupError` + an actionable pointer. No silent fallback that changes numerical results.
- Field constants live in `nfl_gsplat/calibration/field_landmarks.py`: `YARD_LINE_SPACING_M=4.572`, `HALF_WIDTH_M=24.384`, `HASH_OFFSET_M=2.8194`, `GOAL_LINE_X_M=45.720`, `HALF_LENGTH_M=54.864`.
- Never commit real NFL video/frames; `data/` and `kp_eval/` are gitignored. Diagnostics go to `C:\Users\sumedh\diag\` or scratch.
- `pytest -m "not gpu and not slow"` green (**361 passing** baseline) and ruff clean before each commit. One pre-existing repo-wide `B008` (typer.Option defaults) is expected — leave it.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## Deliberate simplification vs the spec (read before Task 1)

The spec calls for a **globally bundled graph** of homographies. The measurements say a simpler thing suffices and should be tried first: **direct frame→reference links kept 36–47 inliers all the way out to a 420-frame gap**. Naive *chaining* is what drifted (6 px → 282 px) — direct links never chain, so they cannot accumulate drift.

Task 1 therefore implements **direct-to-reference registration** with a **local-chain fallback** for frames whose direct link is too weak, plus a **consistency check that fails loud**. A full bundle is deliberately NOT built up front (YAGNI); Task 6 measures whether it is needed. This is an intentional deviation, recorded here so a reviewer does not treat it as a missed requirement.

---

## File Structure

- `nfl_gsplat/calibration/endzone_mosaic.py` — NEW. Registration + accumulation only (Tasks 1–2).
- `nfl_gsplat/calibration/field_model_fit.py` — NEW. Field-model fitting + labeling (Task 3). Separate file: labeling is the subtle, independently-testable part and does not belong tangled with image registration.
- `nfl_gsplat/calibration/run_autocalib.py` — MODIFY. Add `build_endzone_mosaic` driver (Task 5).
- `scripts/02_autocalibrate.py` — MODIFY. Add `--mode mosaic-endzone` (Task 5).
- Tests: `tests/test_endzone_mosaic.py`, `tests/test_field_model_fit.py`, additions to `tests/test_run_autocalib.py`.

---

## Task 1: Frame→reference registration

**Files:**
- Create: `nfl_gsplat/calibration/endzone_mosaic.py`
- Test: `tests/test_endzone_mosaic.py`

**Interfaces:**
- Consumes: `player_masks.boxes_provider_from_tracks(tracks_path) -> masks_provider(cam) -> boxes_for(frame)`; `utils.video.iter_frames(video, start_frame, stride) -> (idx, RGB)`.
- Produces:
  - `keep_mask(shape, boxes, pad=10) -> np.ndarray` (uint8, 255 = keep/field, 0 = player)
  - `register_to_reference(frames, *, ref_idx, min_inliers=25) -> (dict[int, np.ndarray], dict[int, int])` returning `H_by_frame` (frame → 3×3 mapping that frame's pixels INTO the reference) and `inliers_by_frame`. `frames` is `dict[int, np.ndarray]` of BGR images already player-masked-aware.

- [ ] **Step 1: Write the failing test**

Create `tests/test_endzone_mosaic.py`:

```python
import cv2
import numpy as np

from nfl_gsplat.calibration import endzone_mosaic as em


def _textured_field(w=640, h=480, seed=0):
    """Deterministic textured image with strong corners so SIFT has features."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, np.uint8)
    for i in range(9):                       # bright 'yard lines'
        y = 40 + i * 45
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 2)
    for _ in range(300):                     # texture for feature matching
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), 2, (200, 200, 200), -1)
    return img


def _warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def test_register_to_reference_recovers_known_homographies():
    base = _textured_field()
    truth = {0: np.eye(3)}
    frames = {0: base}
    for i, (dx, s) in enumerate([(12.0, 1.00), (25.0, 1.04), (-18.0, 0.97)], start=1):
        H = np.array([[s, 0.0, dx], [0.0, s, 0.5 * dx], [0.0, 0.0, 1.0]])
        truth[i] = H
        frames[i] = _warp(base, H)           # frame i = base warped by H

    H_by, inl = em.register_to_reference(frames, ref_idx=0)
    assert set(H_by) == {0, 1, 2, 3}
    assert np.allclose(H_by[0], np.eye(3), atol=1e-6)
    # H_by[i] maps frame i back INTO the reference, so it must invert truth[i]
    pts = np.float32([[100, 100], [500, 120], [300, 400]]).reshape(-1, 1, 2)
    for i in (1, 2, 3):
        back = cv2.perspectiveTransform(
            cv2.perspectiveTransform(pts, truth[i]), H_by[i])
        assert np.abs(back - pts).max() < 2.0, f"frame {i} round-trip off"
        assert inl[i] >= 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -v`
Expected: FAIL — `module 'endzone_mosaic' has no attribute 'register_to_reference'`.

- [ ] **Step 3: Write the implementation**

Create `nfl_gsplat/calibration/endzone_mosaic.py`:

```python
"""Register endzone frames into one reference frame and accumulate static paint.

The endzone camera is a fixed tripod: it pans/tilts/zooms but never translates.
With no translational parallax the frames are related EXACTLY by homographies
(measured 0.2-0.6 px on real footage), which is what makes this far more precise
than player correspondences (87 px at the true camera).

Registration is DIRECT frame->reference wherever possible. Direct links were
measured to hold across a whole play (36-47 inliers at a 420-frame gap), and
they cannot accumulate drift the way sequential chaining does (chaining drifted
6 px -> 282 px on real footage). A local-chain fallback covers frames whose
direct link is too weak.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.errors import CalibrationError

_RATIO = 0.78          # Lowe ratio for descriptor matching
_RANSAC_PX = 2.5


def keep_mask(shape, boxes, pad: int = 10) -> np.ndarray:
    """255 where features may be taken (field), 0 over players.

    Moving players would contribute non-rigid matches and break the
    pure-rotation model, so they are excluded before feature detection."""
    m = np.full(shape[:2], 255, np.uint8)
    for x1, y1, x2, y2 in boxes or []:
        a = max(0, int(x1) - pad)
        b = max(0, int(y1) - pad)
        c = min(shape[1], int(x2) + pad)
        d = min(shape[0], int(y2) + pad)
        m[b:d, a:c] = 0
    return m


def _detector():
    return cv2.SIFT_create(nfeatures=4000)


def _features(img_bgr, mask=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return _detector().detectAndCompute(gray, mask)


def _homography(fa, fb, min_inliers):
    """Homography mapping image A's pixels into image B. Returns (H, inliers)."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < _RATIO * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, msk = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_PX, maxIters=8000)
    if H is None or msk is None:
        return None, 0
    return H, int(msk.sum())


def register_to_reference(frames, *, ref_idx, min_inliers: int = 25):
    """{frame: H mapping that frame's pixels INTO the reference}, {frame: inliers}.

    Direct link first; if it is too weak, fall back to composing through the
    nearest already-registered frame (a SHORT chain, so drift stays bounded)."""
    if ref_idx not in frames:
        raise CalibrationError(
            f"endzone mosaic: reference frame {ref_idx} not among the sampled "
            "frames — pick a reference that was actually decoded.")
    feats = {i: _features(img) for i, img in frames.items()}
    H_by = {ref_idx: np.eye(3)}
    inl_by = {ref_idx: int(len(feats[ref_idx][0]))}
    pending = []
    for i in sorted(frames):
        if i == ref_idx:
            continue
        H, n = _homography(feats[i], feats[ref_idx], min_inliers)
        if H is not None and n >= min_inliers:
            H_by[i], inl_by[i] = H, n
        else:
            pending.append(i)

    # Fallback: compose through the nearest registered neighbour.
    for i in list(pending):
        done = sorted(H_by)
        if not done:
            break
        j = min(done, key=lambda d: abs(d - i))
        H, n = _homography(feats[i], feats[j], min_inliers)
        if H is not None and n >= min_inliers:
            H_by[i] = H_by[j] @ H
            inl_by[i] = n
            pending.remove(i)
    if pending:
        raise CalibrationError(
            f"endzone mosaic: {len(pending)} frames could not be registered "
            f"(e.g. {pending[:5]}) — too little static field visible; sample "
            "different frames or lower the frame stride.")
    return H_by, inl_by
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -v`
Expected: PASS.

- [ ] **Step 5: Add the fail-loud and masking tests**

```python
def test_register_fails_loud_on_unregisterable_frame():
    import pytest
    from nfl_gsplat.errors import CalibrationError
    frames = {0: _textured_field(), 1: np.zeros((480, 640, 3), np.uint8)}
    with pytest.raises(CalibrationError, match="could not be registered"):
        em.register_to_reference(frames, ref_idx=0)


def test_keep_mask_zeroes_player_boxes_with_padding():
    m = em.keep_mask((100, 200, 3), [(50, 40, 70, 60)], pad=5)
    assert m[50, 60] == 0          # inside the box
    assert m[36, 46] == 0          # inside the pad
    assert m[10, 10] == 255        # untouched field
```

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -v` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/endzone_mosaic.py tests/test_endzone_mosaic.py
git commit -m "feat(calibration): register endzone frames to a reference frame"
```

---

## Task 2: Accumulate static field paint

**Files:**
- Modify: `nfl_gsplat/calibration/endzone_mosaic.py`
- Test: `tests/test_endzone_mosaic.py`

**Interfaces:**
- Consumes: `keep_mask`, `register_to_reference` (Task 1).
- Produces: `accumulate_field_paint(frames, H_by_frame, boxes_by_frame, *, ref_shape, white_lo=(0,0,165), white_hi=(180,70,255)) -> np.ndarray` — float32 votes-per-pixel image in the reference frame, normalised to 0..1 by the per-pixel coverage count (so edges of the mosaic are not penalised for being seen less often).

- [ ] **Step 1: Write the failing test**

```python
def test_accumulate_reinforces_static_paint_and_suppresses_movers():
    h, w = 300, 400
    frames, boxes, H_by = {}, {}, {}
    for i in range(8):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)   # green field (BGR)
        cv2.line(img, (0, 150), (w, 150), (250, 250, 250), 3)   # STATIC paint
        # a bright 'player' that moves every frame and is NOT boxed here
        cv2.rectangle(img, (20 + i * 40, 40), (60 + i * 40, 90), (255, 255, 255), -1)
        frames[i], boxes[i], H_by[i] = img, [], np.eye(3)
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    line_vote = acc[150, w // 2]
    mover_vote = acc[65, 40]
    assert line_vote > 0.9, f"static paint should reinforce, got {line_vote}"
    assert mover_vote < 0.4, f"mover should wash out, got {mover_vote}"


def test_accumulate_recovers_line_occluded_in_some_frames():
    """A SkyCam cable breaks the line in a few frames; accumulation must heal it."""
    h, w = 200, 300
    frames, boxes, H_by = {}, {}, {}
    for i in range(10):
        img = np.full((h, w, 3), (40, 90, 40), np.uint8)
        cv2.line(img, (0, 100), (w, 100), (250, 250, 250), 3)
        if i < 3:                                   # cable occludes mid-span
            cv2.line(img, (140, 90), (140, 110), (20, 20, 20), 9)
        frames[i], boxes[i], H_by[i] = img, [], np.eye(3)
    acc = em.accumulate_field_paint(frames, H_by, boxes, ref_shape=(h, w))
    assert acc[100, 140] > 0.6      # healed: unoccluded in 7/10 frames
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -k accumulate -v`
Expected: FAIL — `has no attribute 'accumulate_field_paint'`.

- [ ] **Step 3: Implement**

Append to `endzone_mosaic.py`:

```python
def _white_mask(img_bgr, lo, hi) -> np.ndarray:
    """Painted lines: bright and low-saturation. Cables are DARK, so they are
    excluded here by construction; their only effect is occlusion, which
    accumulating over many frames heals."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def accumulate_field_paint(frames, H_by_frame, boxes_by_frame, *, ref_shape,
                           white_lo=(0, 0, 165), white_hi=(180, 70, 255)):
    """Votes-per-pixel image of STATIC paint in the reference frame (0..1).

    Each frame's player-masked white mask is warped into the reference and
    summed, then divided by how often each pixel was actually observed. Static
    paint reinforces; movers and per-frame junk wash out."""
    h, w = ref_shape[:2]
    votes = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)
    for i, img in frames.items():
        H = H_by_frame.get(i)
        if H is None:
            continue
        paint = _white_mask(img, white_lo, white_hi)
        paint = cv2.bitwise_and(paint, keep_mask(img.shape, boxes_by_frame.get(i)))
        votes += cv2.warpPerspective(
            (paint > 0).astype(np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
        seen += cv2.warpPerspective(
            np.ones(img.shape[:2], np.float32), H, (w, h), flags=cv2.INTER_NEAREST)
    if not seen.any():
        raise CalibrationError(
            "endzone mosaic: no frames contributed coverage — check the "
            "homographies and that the videos decoded.")
    return np.divide(votes, seen, out=np.zeros_like(votes), where=seen > 0)
```

- [ ] **Step 4: Run to verify it passes**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -v` → all PASS.

- [ ] **Step 5: Lint + commit**

```bash
C:\venvs\nflgsplat\Scripts\python.exe -m ruff check nfl_gsplat/calibration/endzone_mosaic.py tests/test_endzone_mosaic.py
git add nfl_gsplat/calibration/endzone_mosaic.py tests/test_endzone_mosaic.py
git commit -m "feat(calibration): accumulate static field paint in the reference frame"
```

---

## Task 3: Fit the metric field model (labeling)

**This is the crux — it is what killed the field-markings approach. It must fail loud rather than guess.**

**Files:**
- Create: `nfl_gsplat/calibration/field_model_fit.py`
- Test: `tests/test_field_model_fit.py`

**Interfaces:**
- Consumes: accumulated votes image from Task 2; `field_landmarks` constants.
- Produces:
  - `detect_accumulated_lines(votes, *, vote_thresh=0.5, min_len_frac=0.10) -> list[tuple[float, float]]` — each yard-line candidate as `(a, b)` for the image line `x = a*y + b`, sorted by `b`.
  - `label_yard_lines(lines, *, yard_range_m, spacing_px_hint=None, tol_m=0.6) -> list[float]` — world X (metres) per detected line. Raises `CalibrationError` when two different label offsets fit within `tol_m` (ambiguous) or when spacing is not consistent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_model_fit.py`:

```python
import numpy as np
import pytest

from nfl_gsplat.calibration import field_model_fit as fmf
from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError


def test_label_yard_lines_assigns_world_x_from_prior():
    # 5 equally spaced lines; the sideline says the action sits near X=-20..0
    lines = [(0.0, 100.0 + k * 50.0) for k in range(5)]
    xs = fmf.label_yard_lines(lines, yard_range_m=(-20.0, 0.0))
    assert len(xs) == 5
    d = np.diff(xs)
    assert np.allclose(np.abs(d), YARD_LINE_SPACING_M, atol=1e-6)
    assert min(xs) >= -25.0 and max(xs) <= 5.0     # inside the prior window


def test_label_yard_lines_fails_loud_when_ambiguous():
    # A prior window far wider than the observed span admits several offsets
    lines = [(0.0, 100.0 + k * 50.0) for k in range(3)]
    with pytest.raises(CalibrationError, match="ambiguous"):
        fmf.label_yard_lines(lines, yard_range_m=(-50.0, 50.0))


def test_label_yard_lines_rejects_inconsistent_spacing():
    lines = [(0.0, 100.0), (0.0, 150.0), (0.0, 260.0)]      # 50, 110 - not uniform
    with pytest.raises(CalibrationError, match="spacing"):
        fmf.label_yard_lines(lines, yard_range_m=(-20.0, 0.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_field_model_fit.py -v`
Expected: FAIL — module `field_model_fit` not found.

- [ ] **Step 3: Implement**

Create `nfl_gsplat/calibration/field_model_fit.py`:

```python
"""Fit the metric NFL field model to accumulated endzone paint.

Labeling (deciding WHICH yard line each detected line is) is the step that
defeated single-frame field-marking calibration: the painted numbers face the
sideline, so from the endzone they are edge-on and unreadable. Two things make
it tractable here: the accumulated mosaic spans far more of the field than any
one frame, and the calibrated SIDELINE camera says which yard range is in view.

When the evidence admits more than one labeling, this module raises rather than
guessing — a wrong label offset is exactly how the earlier approach produced a
plausible-looking camera that fit nothing.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError


def detect_accumulated_lines(votes, *, vote_thresh: float = 0.5,
                             min_len_frac: float = 0.10):
    """Yard-line candidates from the votes image, as (a, b) with x = a*y + b."""
    mask = (votes >= vote_thresh).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=int(min_len_frac * max(h, w)),
                           maxLineGap=25)
    if segs is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        if abs(y2 - y1) < 1e-6:
            continue
        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1
        out.append((float(a), float(b)))
    return sorted(out, key=lambda ab: ab[1])


def label_yard_lines(lines, *, yard_range_m, tol_m: float = 0.6):
    """World X (metres) for each detected line, or raise if ambiguous.

    Lines are equally spaced by YARD_LINE_SPACING_M in world X, so their image
    order maps to a uniform world sequence; only the OFFSET (which line is
    which) is unknown. We enumerate offsets that place every line inside the
    sideline-derived ``yard_range_m`` window and require exactly one to fit."""
    if len(lines) < 2:
        raise CalibrationError(
            "endzone field fit: need >= 2 yard lines to label, got "
            f"{len(lines)} — accumulated paint too sparse.")
    bs = np.array([b for _a, b in lines], float)
    gaps = np.diff(bs)
    if gaps.min() <= 0:
        raise CalibrationError("endzone field fit: lines are not ordered.")
    # Uniform image spacing is only expected under mild perspective; require the
    # gaps to be self-consistent, else the detections are not a clean pencil.
    if gaps.max() / gaps.min() > 1.6:
        raise CalibrationError(
            "endzone field fit: inconsistent yard-line spacing "
            f"(gap ratio {gaps.max() / gaps.min():.2f}) — detections are not a "
            "clean set of yard lines; raise vote_thresh or re-check masking.")

    n = len(lines)
    lo, hi = float(min(yard_range_m)), float(max(yard_range_m))
    span = (n - 1) * YARD_LINE_SPACING_M
    # Candidate offsets: start X so that the whole run sits within the window
    # (widened by tol_m, since the prior is approximate).
    starts = []
    k = -200
    while True:
        x0 = k * YARD_LINE_SPACING_M
        if x0 > hi + tol_m:
            break
        if x0 >= lo - tol_m and x0 + span <= hi + tol_m:
            starts.append(x0)
        k += 1
    if not starts:
        raise CalibrationError(
            "endzone field fit: no yard-line labeling fits the sideline yard "
            f"range {yard_range_m} for {n} lines spanning {span:.1f} m — the "
            "prior and the detections disagree.")
    if len(starts) > 1:
        raise CalibrationError(
            f"endzone field fit: labeling ambiguous — {len(starts)} offsets fit "
            f"the yard range {yard_range_m} equally well. Narrow the range "
            "(more sideline evidence) or supply an absolute anchor.")
    x0 = starts[0]
    return [x0 + i * YARD_LINE_SPACING_M for i in range(n)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_field_model_fit.py -v` → PASS.

- [ ] **Step 5: Lint + commit**

```bash
C:\venvs\nflgsplat\Scripts\python.exe -m ruff check nfl_gsplat/calibration/field_model_fit.py tests/test_field_model_fit.py
git add nfl_gsplat/calibration/field_model_fit.py tests/test_field_model_fit.py
git commit -m "feat(calibration): label accumulated yard lines, fail loud on ambiguity"
```

---

## Task 4: Solve the reference camera and propagate

**Files:**
- Modify: `nfl_gsplat/calibration/endzone_mosaic.py`
- Test: `tests/test_endzone_mosaic.py`

**Interfaces:**
- Consumes: `endzone_multiplay.EndzonePrior` (has `.center_bounds`, `.center0`, `.focal_range`); `joint_solve.solve_fixed_center(corrs_by_frame, image_size, *, init_results, _frame_data_override, view_deg, center_bounds, audit_drop_px)`; `solve_pnp.CalibrationResult`; `utils.geometry.CameraIntrinsics, CameraPose`.
- Produces:
  - `solve_reference_camera(world_xyz, ref_uv, image_size, prior, *, audit_drop_px=4.0) -> CalibrationResult`
  - `propagate(H_by_frame, ref_cam, n_frames) -> list[CalibrationResult | None]` — frame `t`'s camera. Since `H_t` maps frame `t` INTO the reference, `K_t R_t = H_t^{-1} K_ref R_ref`; the centre is shared, so `t_t = -R_t C`.

- [ ] **Step 1: Write the failing test**

```python
def test_propagate_matches_a_directly_rendered_camera():
    """Build a reference camera, warp by a known H, and check propagate()
    reproduces the camera that actually took the warped image."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()
    ref = CalibrationResult(
        intrinsics=CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.0, num_correspondences=0,
        refined_with_ba=False)

    # a pure zoom about the principal point: frame t -> reference
    s = 1.15
    cx, cy = wh[0] / 2, wh[1] / 2
    H_t = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])

    out = em.propagate({7: H_t}, ref, n_frames=8)
    cam = out[7]
    assert cam is not None
    # a world point must land where H_t would have put it
    X = np.array([[-30.0, 6.0, 0.0]])
    uv_ref = project_points(X, K, R, -R @ C)
    uv_t = project_points(X, cam.intrinsics.K(), cam.pose.R, cam.pose.t)
    back = cv2.perspectiveTransform(uv_t.reshape(-1, 1, 2).astype(np.float32), H_t)
    assert np.abs(back.reshape(-1, 2) - uv_ref).max() < 1.0
    assert np.allclose(cam.pose.center_world(), C, atol=1e-6)   # shared centre
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -k propagate -v`
Expected: FAIL — `has no attribute 'propagate'`.

- [ ] **Step 3: Implement**

Append to `endzone_mosaic.py`:

```python
def solve_reference_camera(world_xyz, ref_uv, image_size, prior,
                           *, audit_drop_px: float = 4.0):
    """One camera for the reference frame from accumulated field correspondences.

    These are arbitrary field points, so we use solve_fixed_center's
    _frame_data_override path (never the named-landmark path)."""
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    world = np.asarray(world_xyz, np.float64).reshape(-1, 3)
    uv = np.asarray(ref_uv, np.float64).reshape(-1, 2)
    if len(world) < 6:
        raise CalibrationError(
            f"endzone mosaic: only {len(world)} field correspondences in the "
            "reference frame (need >= 6) — accumulated paint too sparse.")
    f0 = float(sum(prior.focal_range)) / 2.0
    anchor = CalibrationResult(
        intrinsics=CameraIntrinsics(f0, f0, 0.0, 0.0, 1, 1),
        pose=CameraPose(R=np.eye(3), t=-prior.center0),
        rms_px=0.0, num_correspondences=0, refined_with_ba=False)
    # Repeat the single reference view so the solver's anchor threshold is met.
    frame_data = {i: (world, uv) for i in range(3)}
    results, _mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=image_size,
        init_results=[anchor, anchor, anchor], _frame_data_override=frame_data,
        view_deg=0, center_bounds=prior.center_bounds,
        audit_drop_px=audit_drop_px)
    solved = [r for r in results if r is not None]
    if not solved:
        raise CalibrationError(
            "endzone mosaic: reference camera solve kept no frames — the "
            "field labeling or the accumulated lines are inconsistent.")
    return solved[0]


def propagate(H_by_frame, ref_cam, n_frames: int):
    """Per-frame cameras from the reference camera and each frame's homography.

    H_t maps frame t's pixels INTO the reference, so K_t R_t = H_t^-1 K_ref R_ref.
    The centre is shared (tripod), so t_t = -R_t C."""
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    C = np.asarray(ref_cam.pose.center_world(), np.float64)
    M_ref = ref_cam.intrinsics.K() @ ref_cam.pose.R
    w, h = ref_cam.intrinsics.width, ref_cam.intrinsics.height
    out: list = [None] * n_frames
    for t, H in H_by_frame.items():
        if not (0 <= t < n_frames):
            continue
        M = np.linalg.inv(H) @ M_ref                 # = K_t R_t
        # RQ-decompose M into upper-triangular K and rotation R.
        K, R = _rq3(M)
        if K[2, 2] == 0:
            continue
        K = K / K[2, 2]
        if np.linalg.det(R) < 0:                     # keep a right-handed frame
            K, R = K @ np.diag([-1.0, -1.0, 1.0]), np.diag([-1.0, -1.0, 1.0]) @ R
        fx, fy = float(abs(K[0, 0])), float(abs(K[1, 1]))
        out[t] = CalibrationResult(
            intrinsics=CameraIntrinsics(fx, fy, float(K[0, 2]), float(K[1, 2]), w, h),
            pose=CameraPose(R=R, t=-R @ C),
            rms_px=ref_cam.rms_px, num_correspondences=ref_cam.num_correspondences,
            refined_with_ba=False)
    return out


def _rq3(M):
    """RQ decomposition of a 3x3 matrix into (upper-triangular, rotation)."""
    P = np.flipud(np.eye(3))
    Q_, R_ = np.linalg.qr((P @ M).T)
    K = P @ R_.T @ P
    R = P @ Q_.T
    for i in range(3):                                # make K's diagonal positive
        if K[i, i] < 0:
            K[:, i] *= -1.0
            R[i, :] *= -1.0
    return K, R
```

- [ ] **Step 4: Run to verify it passes**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_endzone_mosaic.py -v` → all PASS.

- [ ] **Step 5: Lint + commit**

```bash
C:\venvs\nflgsplat\Scripts\python.exe -m ruff check nfl_gsplat/calibration/endzone_mosaic.py tests/test_endzone_mosaic.py
git add nfl_gsplat/calibration/endzone_mosaic.py tests/test_endzone_mosaic.py
git commit -m "feat(calibration): solve reference camera and propagate through homographies"
```

---

## Task 5: End-to-end driver + CLI

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py`
- Modify: `scripts/02_autocalibrate.py`
- Test: `tests/test_run_autocalib.py`

**Interfaces:**
- Consumes: everything above; `player_masks.boxes_provider_from_tracks`; `cameras_io.load_camera_track/write_camera_track`; `utils.video.ffprobe_meta/iter_frames`; `endzone_multiplay.EndzonePrior`; `run_autocalib.assemble_track_from_results`.
- Produces: `build_endzone_mosaic(*, play_dir, tracks_path, cameras_npz, endzone_video, fps, prior, stride=6, ref_frame=None, sideline_cam="sideline", endzone_cam="endzone") -> Path`.

- [ ] **Step 1: Write the failing wiring test**

Add to `tests/test_run_autocalib.py`. It must exercise the real driver with a synthetic video written to `tmp_path` (no real footage in CI):

```python
def test_build_endzone_mosaic_writes_endzone_track(tmp_path, monkeypatch):
    """Synthetic endzone clip of a static painted field viewed by a fixed camera
    that only zooms. The driver must register, accumulate, fit, solve and write
    an endzone track while preserving the sideline."""
    import cv2, numpy as np, pandas as pd
    from types import SimpleNamespace
    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    from nfl_gsplat.utils.geometry import CameraIntrinsics

    pdir = tmp_path / "play_x"; pdir.mkdir()
    wh = (640, 360)
    # sideline track (identity-ish) so the driver can read a yard prior
    K = CameraIntrinsics(900.0, 900.0, wh[0] / 2, wh[1] / 2, *wh).K()
    R = np.eye(3); t = np.array([0.0, 0.0, 30.0])
    sl = CameraTrack(K=np.repeat(K[None], 12, 0), R=np.repeat(R[None], 12, 0),
                     t=np.repeat(t[None], 12, 0), conf=np.ones(12),
                     width=wh[0], height=wh[1])
    write_camera_track(pdir / "cameras.npz", {"sideline": sl}, fps=30.0)

    # tracks.parquet: sideline players spanning a known yard window
    rows = []
    for fr in range(12):
        for k in range(6):
            rows.append({"frame": fr, "cam": "sideline", "track_id": k,
                         "player_uid": f"u{k}", "foot_u": 100 + 60 * k,
                         "foot_v": 300, "bbox_x1": 0, "bbox_y1": 0,
                         "bbox_x2": 1, "bbox_y2": 1, "conf": 1})
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c != "cam" else ""
    df.to_parquet(pdir / "tracks.parquet", index=False)

    # synthetic endzone video: painted lines, slight zoom per frame
    vid = pdir / "endzone.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, wh)
    base = np.full((wh[1], wh[0], 3), (40, 90, 40), np.uint8)
    for i in range(9):
        cv2.line(base, (0, 30 + i * 35), (wh[0], 30 + i * 35), (250, 250, 250), 2)
    for fr in range(12):
        s = 1.0 + 0.004 * fr
        M = cv2.getRotationMatrix2D((wh[0] / 2, wh[1] / 2), 0, s)
        vw.write(cv2.warpAffine(base, M, wh))
    vw.release()

    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=wh[0], height=wh[1],
                                                  num_frames=12, fps=30.0))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))
    out = build_endzone_mosaic(
        play_dir=pdir, tracks_path=pdir / "tracks.parquet",
        cameras_npz=pdir / "cameras.npz", endzone_video=vid, fps=30.0,
        prior=prior, stride=2)
    from nfl_gsplat.calibration.cameras_io import load_camera_track
    cams = load_camera_track(out)
    assert "sideline" in cams and "endzone" in cams
    assert (cams["endzone"].conf > 0).sum() >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_run_autocalib.py -k mosaic -v`
Expected: FAIL — `cannot import name 'build_endzone_mosaic'`.

- [ ] **Step 3: Implement the driver**

Add to `run_autocalib.py`:

```python
def build_endzone_mosaic(*, play_dir, tracks_path, cameras_npz, endzone_video,
                         fps, prior, stride: int = 6, ref_frame=None,
                         sideline_cam: str = "sideline",
                         endzone_cam: str = "endzone"):
    """Calibrate the endzone camera from an accumulated static-paint mosaic."""
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration import endzone_mosaic as em
    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.calibration.field_model_fit import (
        detect_accumulated_lines, label_yard_lines)
    from nfl_gsplat.calibration.field_landmarks import HALF_WIDTH_M
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    cams = load_camera_track(cameras_npz)
    if cams.get(sideline_cam) is None:
        raise SetupError(
            f"no {sideline_cam!r} camera in {cameras_npz} — run the sideline "
            "calibration first (02_autocalibrate --mode pretrained).")
    df = pd.read_parquet(tracks_path)

    meta = ffprobe_meta(str(endzone_video))
    image_size = (meta.width, meta.height)

    # sample frames (iter_frames yields RGB; OpenCV wants BGR)
    frames, boxes = {}, {}
    ez_boxes = df[df["cam"] == endzone_cam]
    for idx, rgb in iter_frames(endzone_video, stride=stride):
        frames[idx] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        g = ez_boxes[ez_boxes["frame"] == idx]
        boxes[idx] = list(zip(g["bbox_x1"], g["bbox_y1"], g["bbox_x2"], g["bbox_y2"]))
    if not frames:
        raise SetupError(f"no frames decoded from {endzone_video}")
    ref = ref_frame if ref_frame is not None else sorted(frames)[len(frames) // 2]

    H_by, _inl = em.register_to_reference(frames, ref_idx=ref)
    votes = em.accumulate_field_paint(
        frames, H_by, boxes, ref_shape=(meta.height, meta.width))

    lines = detect_accumulated_lines(votes)
    yard_range = _sideline_yard_range(df, cams[sideline_cam], cam=sideline_cam)
    xs = label_yard_lines(lines, yard_range_m=yard_range)

    # Each labelled line contributes two correspondences (its ends at +-HALF_WIDTH_M).
    world, uv = [], []
    for (a, b), X in zip(lines, xs):
        for Y in (-HALF_WIDTH_M, HALF_WIDTH_M):
            y_img = 0.25 * meta.height if Y < 0 else 0.75 * meta.height
            world.append([X, Y, 0.0])
            uv.append([a * y_img + b, y_img])
    ref_cam = em.solve_reference_camera(world, uv, image_size, prior)

    results = em.propagate(H_by, ref_cam, n_frames=meta.num_frames)
    cams[endzone_cam] = assemble_track_from_results(
        results, width=meta.width, height=meta.height, max_gap=stride * 3)
    return write_camera_track(Path(cameras_npz), cams, fps=fps)


def _sideline_yard_range(df, sideline_track, *, cam, pad_m: float = 8.0):
    """World-X window the sideline camera says is in play, padded."""
    import numpy as np

    from nfl_gsplat.calibration.endzone_identity import field_positions_by_uid

    field = field_positions_by_uid(df, sideline_track, cam=cam, smooth_window=1)
    xs = [p[0] for fmap in field.values() for p in fmap.values()]
    if not xs:
        raise SetupError(
            "cannot derive a yard range: no sideline field positions — "
            "run scripts/03c_identity_tracks.py so tracks carry player_uid.")
    return (float(np.min(xs)) - pad_m, float(np.max(xs)) + pad_m)
```

- [ ] **Step 4: Wire the CLI**

In `scripts/02_autocalibrate.py` add `mosaic_endzone = "mosaic-endzone"` to `CalibMode`, and in `main`:

```python
    if mode is CalibMode.mosaic_endzone:
        from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
        from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
        ep = meta.endzone_prior
        if ep is None:
            raise SetupError(
                f"{pd.meta_yaml}: --mode mosaic-endzone needs an 'endzone_prior:' "
                "block (x_range/y_range/z_range/focal_range).")
        out = build_endzone_mosaic(
            play_dir=pd.dir, tracks_path=pd.dir / "tracks.parquet",
            cameras_npz=pd.dir / "cameras.npz", endzone_video=pd.video("endzone"),
            fps=meta.fps,
            prior=EndzonePrior(tuple(ep["x_range"]), tuple(ep["y_range"]),
                               tuple(ep["z_range"]), tuple(ep["focal_range"])))
        _LOG.info(f"wrote endzone mosaic calibration → {out}")
```

- [ ] **Step 5: Run tests + lint**

Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest tests/test_run_autocalib.py -v` → PASS.
Run: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest -m "not gpu and not slow" -q` → all green.
Run: `C:\venvs\nflgsplat\Scripts\python.exe -m ruff check nfl_gsplat scripts tests` → only the pre-existing `B008`.

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py
git commit -m "feat(calibration): mosaic-endzone driver + CLI mode"
```

---

## Task 6: Real acceptance — the numeric gate

**Files:** none (verification only). Diagnostics to `C:\Users\sumedh\diag\`, never the repo.

- [ ] **Step 1: Back up the play's cameras.npz**

```bash
cp data/2025/week_04/SEA_at_AZ/play_002/cameras.npz /c/Users/sumedh/diag/play_002_cameras_backup.npz
```

- [ ] **Step 2: Ensure play_002 has a calibrated sideline**

play_002 currently has tracks (from `03c`) but no `cameras.npz`. Run the roboflow + sideline stages for it:

```bash
set PY=C:\venvs\nflgsplat\Scripts\python.exe
set ROBOFLOW_API_KEY=...
%PY% scripts/run_precompute_batch.py --stage roboflow --plays play_002
%PY% scripts/run_precompute_batch.py --stage sideline --plays play_002
```

- [ ] **Step 3: Run the mosaic calibration**

```bash
%PY% scripts/02_autocalibrate.py --play-dir data/2025/week_04/SEA_at_AZ/play_002 --mode mosaic-endzone
```
Expected: an `endzone` track written; centre inside the prior box (`X` in −150..−60, `|Y|` ≤ 15, `Z` 10..60).

- [ ] **Step 4: Measure per-frame reprojection — THE GATE**

Write a scratch script (in the scratchpad, not the repo) that, for each solved frame, projects the labelled yard-line world points through that frame's camera and measures the distance to the detected paint in that frame. Report median and 90th percentile.

**Pass = few-px (~1–3 px) median.** If it is 10s of px, the labeling is probably off by an offset — check `label_yard_lines`' chosen `x0` against the sideline yard range before touching anything else.

- [ ] **Step 5: Overlay check**

Render the field model through 3 frames spread across the play, save to `C:\Users\sumedh\diag\`, and confirm the drawn lines sit on the real painted lines.

- [ ] **Step 6: Report**

Report centre, focal range, solved-frame count, median/p90 reprojection, and whether the labeling was unambiguous. If `label_yard_lines` raised, that is a *result*, not a failure of the run — report the ambiguity and the candidate offsets.

---

## Self-Review

**Spec coverage:**
- Mosaic registration (spec stage 1) → Task 1. ✓ (with the documented direct-link simplification)
- Accumulate static paint, cable occlusion healed by accumulation (spec stage 2) → Task 2. ✓
- Field-model fit + labeling, fail loud on ambiguity (spec stage 2) → Task 3. ✓
- Reference-camera solve via `_frame_data_override` + `center_bounds` (spec) → Task 4. ✓
- Propagate per frame, native pixels, shared centre (spec stage 3) → Task 4. ✓
- Driver + `--mode mosaic-endzone`, sideline preserved (spec) → Task 5. ✓
- Error handling gates (unlinkable frame, sparse paint, ambiguous labeling, no usable frames, missing prior/sideline) → Tasks 1, 2, 3, 4, 5. ✓
- Testing: synthetic recovery, occlusion healing, ambiguity, propagation, real acceptance with a numeric gate → Tasks 1–4, 6. ✓
- **Gap accepted deliberately:** the spec's *global bundle* is not built (see the simplification note); Task 6 measures whether it is needed. The spec's *absolute anchor* (goal line/logo) is likewise not implemented — the sideline yard range is the sole disambiguator in v1, and `label_yard_lines` fails loud when that is insufficient, which is the safe behaviour. Add the anchor only if Task 6 shows ambiguity.

**Placeholder scan:** no TBD/TODO; every code step carries real code. Task 6 Step 4's measurement script is described rather than written out, because it is a throwaway diagnostic outside the repo — its pass criterion is stated numerically.

**Type consistency:** `H_by_frame: dict[int, np.ndarray]` maps frame→reference in Tasks 1, 2, 4, 5. `votes: np.ndarray` float 0..1 in Tasks 2, 3, 5. `lines: list[(a, b)]` with `x = a*y + b` in Tasks 3, 5. `EndzonePrior` fields `.center_bounds/.center0/.focal_range` used consistently in Tasks 4, 5. `CalibrationResult` construction matches its real fields (`intrinsics, pose, rms_px, num_correspondences, refined_with_ba`).

## Risks

- **Labeling is the crux** and v1 leans entirely on the sideline yard range. Expect `label_yard_lines` to raise on wide/uncertain priors — that is by design. Fallback: an absolute anchor (goal line/logo), or a one-time human-confirmed offset per game (cheap, since C is shared across the first half).
- **Correspondence construction in Task 5 assumes each yard line spans the field width**, taking its two endpoints at fixed image heights. If the accumulated lines are clipped by the frame, those endpoints are not at ±`HALF_WIDTH_M` and the solve will be biased — Task 6's reprojection number is what exposes this; the fix is to intersect the labelled lines with the detected sidelines instead.
- **Direct-link registration may thin out** on plays with large pan; the local-chain fallback covers it, and Task 6 reports if drift appears.
