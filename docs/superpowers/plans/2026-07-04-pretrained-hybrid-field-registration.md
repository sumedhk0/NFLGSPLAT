# Pretrained-Hybrid Field Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-frame labeled field correspondences with zero hand-labeling/training: a pretrained Roboflow model names the yard lines, repaired classical detection measures them, fusion feeds the existing PnP → `cameras.npz` path.

**Architecture:** A Windows-side precompute script caches Roboflow hosted-inference keypoints to `roboflow_kps.json`. Offline, per frame: slope-preserving classical yard-line detection + hash-row fits (existing) are fused with the cached named keypoints — each model keypoint votes the identity of its nearest classical line; identified line × hash-row intersections plus model number keypoints become `[(landmark_name, (u,v))]` for the existing `solve_pnp_from_correspondences` / `assemble_track_from_results` machinery.

**Tech Stack:** Python, OpenCV, numpy, `inference-sdk` (precompute only), pytest. No torch, no GPU anywhere in this cycle.

**Spec:** `docs/superpowers/specs/2026-07-04-pretrained-hybrid-field-registration-design.md`

## Global Constraints

- NEVER commit real NFL video/frames; `roboflow_kps.json` lives in `<play_dir>` (outside the repo), test fixtures are synthetic.
- Fail loud: `SetupError`/`CalibrationError` with a pointer; no silent fallbacks that change numerical results.
- World frame: +X toward home endzone; `away_*` yard lines have negative X, `home_*` positive, `mid_50` = 0 (`_yardline_x_m`).
- Image convention (validated on this footage): **left = +Y = image-top; right = −Y = image-bottom**. Upper hash row → `*_left_hash`; model class `"30"` (bare) = top number → `*_left_number`; `"30-bottom"` → `*_right_number`.
- Model sideline classes (`*-sl`) are hallucinated — always dropped, precompute-side.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Keep `pytest -m "not gpu and not slow"` green.

---

### Task 1: Yard-line name ↔ world-X inverse (mapping consistency core)

**Files:**
- Modify: `nfl_gsplat/calibration/field_landmarks.py` (add one function after `_yardline_x_m`)
- Test: `tests/test_field_landmarks_inverse.py` (new)

**Interfaces:**
- Consumes: `_yardline_x_m(name) -> float`, `YARD_LINE_SPACING_M`, `GOAL_LINE_X_M` (all existing in `field_landmarks.py`).
- Produces: `yardline_name_from_x_m(x_m: float, *, tol_m: float = 0.5) -> str` — inverse of `_yardline_x_m` for painted yard lines (goal..goal). Raises `ValueError` if `x_m` is farther than `tol_m` from every painted line. Task 4's neighbor-fill relies on this exact signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_field_landmarks_inverse.py
import pytest

from nfl_gsplat.calibration.field_landmarks import (
    _yardline_names, _yardline_x_m, yardline_name_from_x_m,
)


def test_round_trip_every_yardline():
    # THE consistency guarantee: name -> X -> name is the identity for all
    # 21 painted lines (away_goal..home_goal). A single mismatch here means
    # misidentified lines and silently wrong calibration.
    for name in _yardline_names():
        assert yardline_name_from_x_m(_yardline_x_m(name)) == name


def test_snaps_within_tolerance():
    x30_away = _yardline_x_m("away_30")            # -18.288
    assert yardline_name_from_x_m(x30_away + 0.4) == "away_30"
    assert yardline_name_from_x_m(x30_away - 0.4) == "away_30"


def test_rejects_between_lines():
    # halfway between away_30 and away_35 is 2.286 m from both — no snap
    x = 0.5 * (_yardline_x_m("away_30") + _yardline_x_m("away_35"))
    with pytest.raises(ValueError, match="no painted yard line"):
        yardline_name_from_x_m(x)


def test_rejects_beyond_goal_lines():
    with pytest.raises(ValueError, match="no painted yard line"):
        yardline_name_from_x_m(60.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_field_landmarks_inverse.py -v`
Expected: FAIL — `ImportError: cannot import name 'yardline_name_from_x_m'`

- [ ] **Step 3: Implement**

Add to `nfl_gsplat/calibration/field_landmarks.py`, directly after `_yardline_names()`:

```python
def yardline_name_from_x_m(x_m: float, *, tol_m: float = 0.5) -> str:
    """Inverse of :func:`_yardline_x_m`: nearest painted yard line's name.

    Snaps ``x_m`` to the closest painted line (away_goal..home_goal) if within
    ``tol_m`` meters; raises ValueError otherwise. Round-trip with
    ``_yardline_x_m`` is exact for every painted line (tested).
    """
    best_name, best_d = None, float("inf")
    for name in _yardline_names():
        d = abs(_yardline_x_m(name) - x_m)
        if d < best_d:
            best_name, best_d = name, d
    if best_d > tol_m:
        raise ValueError(
            f"no painted yard line within {tol_m} m of X={x_m:.3f} m "
            f"(nearest: {best_name} at {best_d:.3f} m)"
        )
    return best_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_field_landmarks_inverse.py -v`
Expected: 4 PASS

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — expected: all pass.

```bash
git add nfl_gsplat/calibration/field_landmarks.py tests/test_field_landmarks_inverse.py
git commit -m "feat(calibration): yardline_name_from_x_m inverse with round-trip guarantee

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Roboflow keypoint mapping + JSON cache I/O

**Files:**
- Create: `nfl_gsplat/calibration/roboflow_kps.py`
- Test: `tests/test_roboflow_kps.py` (new)

**Interfaces:**
- Consumes: `NFL_LANDMARKS` (existing mapping), `SetupError` from `nfl_gsplat.errors`.
- Produces (Tasks 3, 5, 6 rely on these exact signatures):
  - `to_nfl_name(model_class: str, territory: str) -> str | None` — Roboflow class → `NFL_LANDMARKS` key, `None` if unmappable. Never returns a `*_sideline` name (upstream drops `-sl`, this also maps them to `None` for defense in depth).
  - `yard_base(model_class: str, territory: str) -> str | None` — just the yard-line base (`"away_30"`, `"mid_50"`), for identity voting.
  - `ModelKeypoint = tuple[str, float, float, float]` — `(model_class, u, v, conf)`.
  - `write_kps_json(path, *, model_id: str, video_name: str, num_frames: int, kp_conf: float, frames: dict[int, list[ModelKeypoint]]) -> None`
  - `load_kps_json(path, *, expect_num_frames: int | None = None) -> dict[int, list[ModelKeypoint]]` — raises `SetupError` on missing file or frame-count mismatch.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_roboflow_kps.py
import json

import pytest

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.calibration.roboflow_kps import (
    load_kps_json, to_nfl_name, write_kps_json, yard_base,
)
from nfl_gsplat.errors import SetupError


# The full mapping table, both territories. Model classes come from
# football-field-key-points-mvmjf/2 (yard + optional top/bottom + hash/sl).
CASES = [
    # (model_class, territory, expected NFL name)
    ("30",             "away", "away_30_left_number"),   # bare = top number
    ("30-bottom",      "away", "away_30_right_number"),
    ("30-top-hash",    "away", "away_30_left_hash"),
    ("30-bottom-hash", "away", "away_30_right_hash"),
    ("20",             "home", "home_20_left_number"),
    ("50",             "home", "mid_50_left_number"),    # 50 is territory-less
    ("50-top-hash",    "away", "mid_50_left_hash"),
]


@pytest.mark.parametrize("cls,terr,expected", CASES)
def test_to_nfl_name_table(cls, terr, expected):
    name = to_nfl_name(cls, terr)
    assert name == expected
    assert name in NFL_LANDMARKS          # every mapped name must exist


@pytest.mark.parametrize("cls", ["30-top-sl", "30-bottom-sl"])
def test_sidelines_always_unmappable(cls):
    assert to_nfl_name(cls, "away") is None


@pytest.mark.parametrize("cls", ["FG-POST", "goalline", "endzone-corner", "abc"])
def test_unknown_classes_unmappable(cls):
    assert to_nfl_name(cls, "home") is None


def test_yard_base():
    assert yard_base("30-top-hash", "away") == "away_30"
    assert yard_base("50", "away") == "mid_50"
    assert yard_base("FG-POST", "away") is None


def test_json_round_trip(tmp_path):
    frames = {0: [("30", 671.0, 242.0, 0.86)], 7: []}
    p = tmp_path / "roboflow_kps.json"
    write_kps_json(p, model_id="m/2", video_name="v.mp4", num_frames=10,
                   kp_conf=0.5, frames=frames)
    loaded = load_kps_json(p, expect_num_frames=10)
    assert loaded[0] == [("30", 671.0, 242.0, 0.86)]
    assert loaded[7] == []
    assert 3 not in loaded                 # absent frame stays absent


def test_load_missing_file_fails_loud(tmp_path):
    with pytest.raises(SetupError, match="03_roboflow_precompute"):
        load_kps_json(tmp_path / "nope.json")


def test_load_frame_count_mismatch_fails_loud(tmp_path):
    p = tmp_path / "roboflow_kps.json"
    write_kps_json(p, model_id="m/2", video_name="v.mp4", num_frames=10,
                   kp_conf=0.5, frames={})
    with pytest.raises(SetupError, match="stale"):
        load_kps_json(p, expect_num_frames=999)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_roboflow_kps.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.roboflow_kps`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/roboflow_kps.py
"""Pretrained Roboflow field-keypoint cache: class-name mapping + JSON I/O.

The hosted model (football-field-key-points-mvmjf) emits classes like
'30', '30-bottom', '30-top-hash', '30-bottom-sl'. Mapping to NFL_LANDMARKS
names (validated on real All-22 eval, 2026-07-03):

    '30'              -> {terr}_30_left_number    (bare = TOP number; left=+Y=image-top)
    '30-bottom'       -> {terr}_30_right_number
    '30-top-hash'     -> {terr}_30_left_hash
    '30-bottom-hash'  -> {terr}_30_right_hash
    '*-sl'            -> None  (sidelines hallucinated off-frame; dropped)
    '50*'             -> mid_50_* (territory-less)

``territory`` ('home'|'away') resolves which side of the 50 the yard numbers
belong to — the model doesn't know; the play's meta does.
"""
from __future__ import annotations

import json
from pathlib import Path

from nfl_gsplat.errors import SetupError

ModelKeypoint = tuple[str, float, float, float]   # (model_class, u, v, conf)

_VARIANT_TABLE = {
    (): ("left", "number"),
    ("bottom",): ("right", "number"),
    ("top", "hash"): ("left", "hash"),
    ("bottom", "hash"): ("right", "hash"),
    # sideline variants intentionally absent -> None (hallucinated)
}


def yard_base(model_class: str, territory: str) -> str | None:
    """'30-top-hash' -> 'away_30' (or None for non-yard classes)."""
    if territory not in ("home", "away"):
        raise ValueError(f"territory must be home/away, got {territory!r}")
    yard = model_class.split("-")[0]
    if not yard.isdigit():
        return None
    yd = int(yard)
    if yd == 50:
        return "mid_50"
    if yd < 5 or yd > 45 or yd % 5 != 0:
        return None
    return f"{territory}_{yd}"


def to_nfl_name(model_class: str, territory: str) -> str | None:
    """Roboflow class -> NFL_LANDMARKS name, or None if unmappable."""
    base = yard_base(model_class, territory)
    if base is None:
        return None
    variant = tuple(model_class.split("-")[1:])
    if variant not in _VARIANT_TABLE:
        return None
    lr, kind = _VARIANT_TABLE[variant]
    return f"{base}_{lr}_{kind}"


def write_kps_json(path, *, model_id: str, video_name: str, num_frames: int,
                   kp_conf: float, frames: dict[int, list[ModelKeypoint]]) -> None:
    doc = {
        "model_id": model_id, "video": video_name, "num_frames": num_frames,
        "kp_conf": kp_conf,
        "frames": {str(i): [{"name": n, "u": u, "v": v, "conf": c}
                            for (n, u, v, c) in kps]
                   for i, kps in frames.items()},
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def load_kps_json(path, *, expect_num_frames: int | None = None
                  ) -> dict[int, list[ModelKeypoint]]:
    p = Path(path)
    if not p.exists():
        raise SetupError(
            f"roboflow keypoint cache not found: {p} — run "
            "scripts/03_roboflow_precompute.py on the play video first."
        )
    doc = json.loads(p.read_text())
    if expect_num_frames is not None and doc.get("num_frames") != expect_num_frames:
        raise SetupError(
            f"stale roboflow_kps.json: cached for {doc.get('num_frames')} frames, "
            f"video has {expect_num_frames} — re-run scripts/03_roboflow_precompute.py."
        )
    return {int(i): [(k["name"], float(k["u"]), float(k["v"]), float(k["conf"]))
                     for k in kps]
            for i, kps in doc["frames"].items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_roboflow_kps.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add nfl_gsplat/calibration/roboflow_kps.py tests/test_roboflow_kps.py
git commit -m "feat(calibration): roboflow keypoint class mapping + JSON cache I/O

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Precompute script (`03_roboflow_precompute.py`)

**Files:**
- Create: `scripts/03_roboflow_precompute.py`
- Create: `nfl_gsplat/calibration/roboflow_precompute.py` (testable core; script is a thin CLI)
- Test: `tests/test_roboflow_precompute.py` (new)

**Interfaces:**
- Consumes: `write_kps_json`, `ModelKeypoint` (Task 2).
- Produces: `run_precompute(frames_iter, *, infer_fn, model_id, video_name, num_frames, out_json, kp_conf=0.5) -> int` (returns #frames with ≥1 kept keypoint). `frames_iter` yields `(frame_idx, bgr_ndarray)`; `infer_fn(bgr) -> list[ModelKeypoint]` is injected (real: inference-sdk HTTP; tests: stub). Drops `-sl` classes and `conf < kp_conf` at write time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_roboflow_precompute.py
import numpy as np

from nfl_gsplat.calibration.roboflow_kps import load_kps_json
from nfl_gsplat.calibration.roboflow_precompute import run_precompute


def _frames(n):
    img = np.zeros((10, 10, 3), np.uint8)
    for i in range(n):
        yield i, img


def test_precompute_writes_filtered_cache(tmp_path):
    def infer_fn(bgr):
        return [("30", 100.0, 50.0, 0.9),        # kept
                ("30-top-sl", 5.0, 5.0, 0.9),    # sideline -> dropped
                ("20", 10.0, 10.0, 0.2)]         # low conf -> dropped
    out = tmp_path / "roboflow_kps.json"
    n_hit = run_precompute(_frames(3), infer_fn=infer_fn, model_id="m/2",
                           video_name="v.mp4", num_frames=3, out_json=out,
                           kp_conf=0.5)
    assert n_hit == 3
    loaded = load_kps_json(out, expect_num_frames=3)
    assert loaded[0] == [("30", 100.0, 50.0, 0.9)]
    assert len(loaded) == 3


def test_precompute_stride_leaves_frames_absent(tmp_path):
    # caller controls stride by what frames_iter yields; absent frames stay absent
    def infer_fn(bgr):
        return [("30", 1.0, 2.0, 0.9)]
    out = tmp_path / "kps.json"
    frames = ((i, np.zeros((4, 4, 3), np.uint8)) for i in (0, 5))
    run_precompute(frames, infer_fn=infer_fn, model_id="m/2", video_name="v.mp4",
                   num_frames=10, out_json=out, kp_conf=0.5)
    loaded = load_kps_json(out, expect_num_frames=10)
    assert set(loaded) == {0, 5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_roboflow_precompute.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.roboflow_precompute`

- [ ] **Step 3: Implement core**

```python
# nfl_gsplat/calibration/roboflow_precompute.py
"""Core of the roboflow precompute step (script-independent, testable).

The CLI (scripts/03_roboflow_precompute.py) wires the video reader and the
hosted-inference HTTP client; this module only filters and caches.
"""
from __future__ import annotations

from nfl_gsplat.calibration.roboflow_kps import ModelKeypoint, write_kps_json


def run_precompute(frames_iter, *, infer_fn, model_id: str, video_name: str,
                   num_frames: int, out_json, kp_conf: float = 0.5) -> int:
    """Run ``infer_fn`` over frames, filter, write the JSON cache.

    Drops sideline classes ('*-sl', hallucinated) and keypoints below
    ``kp_conf``. Returns the number of frames with >=1 kept keypoint.
    """
    frames: dict[int, list[ModelKeypoint]] = {}
    n_hit = 0
    for idx, bgr in frames_iter:
        kept = [(n, u, v, c) for (n, u, v, c) in infer_fn(bgr)
                if c >= kp_conf and not n.endswith("-sl")]
        frames[idx] = kept
        if kept:
            n_hit += 1
    write_kps_json(out_json, model_id=model_id, video_name=video_name,
                   num_frames=num_frames, kp_conf=kp_conf, frames=frames)
    return n_hit
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_roboflow_precompute.py -v`
Expected: 2 PASS

- [ ] **Step 5: Write the CLI script (no test; thin wiring, mirrors eval script)**

```python
# scripts/03_roboflow_precompute.py
"""Cache pretrained Roboflow field keypoints for a play video (run on Windows).

The ONLY step needing internet + ROBOFLOW_API_KEY. Everything downstream
(PACE or local) reads the JSON. One run per play, ever.

    set ROBOFLOW_API_KEY=...
    python scripts/03_roboflow_precompute.py "C:\\...\\sideline.mp4" ^
        --out "C:\\...\\play_dir\\roboflow_kps.json" ^
        --model-id football-field-key-points-mvmjf/2 [--stride 1] [--kp-conf 0.5]
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2

from nfl_gsplat.calibration.roboflow_precompute import run_precompute


def _video_frames(path, stride):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="<play_dir>/roboflow_kps.json")
    ap.add_argument("--model-id", default="football-field-key-points-mvmjf/2")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--api-url", default="https://detect.roboflow.com")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY or pass --api-key")

    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)

    def infer_fn(bgr):
        # hosted API takes a file path; write frame to a temp png
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            cv2.imwrite(tmp, bgr)
            res = client.infer(tmp, model_id=args.model_id)
        finally:
            os.unlink(tmp)
        r = res[0] if isinstance(res, list) else res
        out = []
        for pred in (r.get("predictions") or []):
            for kp in (pred.get("keypoints") or []):
                name = kp.get("class_name") or kp.get("class")
                if name is None or kp.get("x") is None:
                    continue
                out.append((str(name), float(kp["x"]), float(kp["y"]),
                            float(kp.get("confidence", 0.0))))
        return out

    cap = cv2.VideoCapture(args.video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_hit = run_precompute(
        _video_frames(args.video, args.stride), infer_fn=infer_fn,
        model_id=args.model_id, video_name=Path(args.video).name,
        num_frames=num_frames, out_json=args.out, kp_conf=args.kp_conf)
    print(f"cached keypoints for {n_hit} frames -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — expected: all pass.

```bash
git add nfl_gsplat/calibration/roboflow_precompute.py scripts/03_roboflow_precompute.py tests/test_roboflow_precompute.py
git commit -m "feat(calibration): roboflow precompute core + Windows CLI script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Slope-preserving `_merge_collinear`

**Files:**
- Modify: `nfl_gsplat/calibration/field_detect.py:68-82` (replace `_merge_collinear`)
- Test: `tests/test_field_detect.py` (add tests; file exists)

**Interfaces:**
- Consumes: `YardLineSeg` (existing).
- Produces: same signature `_merge_collinear(segs: list[YardLineSeg], x_tol: float = 18.0) -> list[YardLineSeg]`, but merged segments now PRESERVE slant: each output is the least-squares line `x = a·y + b` through member endpoints, spanning the members' y-range. Grouping compares x at the global mean-y of all input endpoints (not per-segment mean-x, which mis-groups slanted lines at different heights). All existing consumers (`detect_lines`, `line_x_at`) already handle slanted segments.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_field_detect.py`)

```python
def test_merge_collinear_preserves_slope():
    # Two pieces of the SAME slanted line (slope: x increases 100px over 400px of y).
    from nfl_gsplat.calibration.field_detect import _merge_collinear
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_identify import line_x_at
    a = YardLineSeg((400.0, 0.0), (450.0, 200.0))
    b = YardLineSeg((450.0, 200.0), (500.0, 400.0))
    merged = _merge_collinear([a, b])
    assert len(merged) == 1
    m = merged[0]
    # slope preserved: x at top ~400, x at bottom ~500 (old code collapsed both to ~450)
    assert abs(line_x_at(m, 0.0) - 400.0) < 2.0
    assert abs(line_x_at(m, 400.0) - 500.0) < 2.0
    # spans the union of member y-ranges
    ys = sorted([m.p0[1], m.p1[1]])
    assert ys[0] <= 1.0 and ys[1] >= 399.0


def test_merge_collinear_keeps_distinct_slanted_lines_apart():
    # Two parallel slanted lines ~60px apart must NOT merge.
    from nfl_gsplat.calibration.field_detect import _merge_collinear
    from nfl_gsplat.calibration.field_features import YardLineSeg
    a = YardLineSeg((400.0, 0.0), (500.0, 400.0))
    b = YardLineSeg((460.0, 0.0), (560.0, 400.0))
    assert len(_merge_collinear([a, b])) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_field_detect.py -v -k merge_collinear`
Expected: `test_merge_collinear_preserves_slope` FAILS (old code returns vertical segment at mean-x ≈ 450, so `line_x_at(m, 0.0)` ≈ 450, not 400). The keeps-apart test may pass already.

- [ ] **Step 3: Replace `_merge_collinear`**

```python
def _merge_collinear(segs: list[YardLineSeg], x_tol: float = 18.0) -> list[YardLineSeg]:
    """Merge near-vertical segments belonging to the same painted line,
    PRESERVING slant (real broadcast yard lines lean well off vertical; the
    old vertical-collapse merge displaced them by tens of px mid-frame).

    Grouping: segments sorted/compared by their x at the global mean-y of all
    endpoints (a common reference row — per-segment mean-x mis-groups slanted
    lines detected at different heights). Each group is refit by least squares
    as x = a*y + b through all member endpoints, spanning the members' y-range.
    """
    import numpy as np
    if not segs:
        return []
    all_y = [p[1] for s in segs for p in (s.p0, s.p1)]
    ref_y = float(np.mean(all_y))

    def x_at(s: YardLineSeg, y: float) -> float:
        (x0, y0), (x1, y1) = s.p0, s.p1
        if abs(y1 - y0) < 1e-6:
            return 0.5 * (x0 + x1)
        return x0 + (y - y0) / (y1 - y0) * (x1 - x0)

    segs = sorted(segs, key=lambda s: x_at(s, ref_y))
    groups: list[list[YardLineSeg]] = []
    for s in segs:
        if groups and abs(x_at(groups[-1][-1], ref_y) - x_at(s, ref_y)) < x_tol:
            groups[-1].append(s)
        else:
            groups.append([s])

    merged: list[YardLineSeg] = []
    for g in groups:
        pts = np.array([p for s in g for p in (s.p0, s.p1)], dtype=np.float64)
        A = np.stack([pts[:, 1], np.ones(len(pts))], axis=1)
        a, b = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        merged.append(YardLineSeg((float(a * y0 + b), y0), (float(a * y1 + b), y1)))
    return merged
```

- [ ] **Step 4: Run the file's tests, then the full suite**

Run: `python -m pytest tests/test_field_detect.py -v`
Expected: all PASS (existing synthetic tests use near-vertical lines — the LSQ fit reproduces verticals exactly).
Run: `python -m pytest -m "not gpu and not slow" -q`
Expected: all pass. If a downstream test assumed vertical merged output, fix THAT test only if it encoded the bug (vertical collapse); the new behavior is the spec.

- [ ] **Step 5: Commit**

```bash
git add nfl_gsplat/calibration/field_detect.py tests/test_field_detect.py
git commit -m "fix(calibration): _merge_collinear preserves yard-line slant

Old merge collapsed merged Hough segments to vertical at their mean-x,
displacing slanted broadcast yard lines by tens of px. Groups now compare
x at a common reference row and refit x = a*y + b by least squares.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Fusion — identity votes + labeled correspondences

**Files:**
- Create: `nfl_gsplat/calibration/fuse_pretrained.py`
- Test: `tests/test_fuse_pretrained.py` (new)

**Interfaces:**
- Consumes: `line_x_at(seg, y)` (`field_identify`), `fit_hash_rows(hashes, *, image_width)` (`field_identify`), `yardline_name_from_x_m` + `_yardline_x_m` (Task 1), `yard_base` + `to_nfl_name` + `ModelKeypoint` (Task 2), `YardLineSeg` (`field_features`).
- Produces (Task 6 relies on this exact signature):
  - `fuse_frame(yard_lines: list[YardLineSeg], hashes: list[tuple[float,float]], model_kps: list[ModelKeypoint], *, territory: str, image_size: tuple[int,int], max_assign_px: float = 60.0, min_margin: float = 2.0) -> list[tuple[str, tuple[float, float]]]` — labeled correspondences for `solve_pnp_from_correspondences`. Returns `[]` (frame gap) rather than guessing when identity is ambiguous.
  - `identify_lines(yard_lines, model_kps, *, territory, max_assign_px=60.0, min_margin=2.0) -> dict[int, str]` — line index → yard base name (exposed for tests).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fuse_pretrained.py
"""Fusion tests on a synthetic slanted view.

Geometry: yard lines rendered as slanted segments x = X_world*40 + 800 + 0.15*y
(40 px per meter, leaning right). Hash rows at v=300 (upper/left/+Y) and v=700
(lower/right/-Y). Model keypoints derived from the same geometry + noise.
"""
import numpy as np
import pytest

from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import _yardline_x_m
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame, identify_lines

W, H = 1920, 1080


def _seg_for(x_world):
    def u(y):
        return x_world * 40.0 + 800.0 + 0.15 * y
    return YardLineSeg((u(0.0), 0.0), (u(float(H)), float(H)))


def _u_at(x_world, y):
    return x_world * 40.0 + 800.0 + 0.15 * y


LINES = [_seg_for(_yardline_x_m(n)) for n in ("away_40", "away_35", "away_30")]


def _hashes():
    # dense ticks along both rows so fit_hash_rows locks on
    return ([(float(x), 300.0) for x in range(100, 1900, 40)]
            + [(float(x), 700.0) for x in range(100, 1900, 40)])


def test_identify_lines_votes_nearest_with_noise():
    # model kp for away_30's top number, 25px off the line -> still votes right line
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 25.0
    ident = identify_lines(LINES, [("30", u30, 200.0, 0.8)], territory="away")
    assert ident == {2: "away_30"}


def test_identify_lines_fills_unvoted_neighbors_consistently():
    # votes on away_40 and away_30; the middle line must become away_35 —
    # via world-X interpolation + snap + yardline_name_from_x_m round-trip.
    kps = [("40", _u_at(_yardline_x_m("away_40"), 100.0) + 10.0, 100.0, 0.8),
           ("30", _u_at(_yardline_x_m("away_30"), 500.0) - 15.0, 500.0, 0.8)]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {0: "away_40", 1: "away_35", 2: "away_30"}


def test_identify_lines_drops_frame_on_conflict():
    # two kps voting DIFFERENT yards for the same line -> ambiguous -> {}
    u = _u_at(_yardline_x_m("away_30"), 400.0)
    kps = [("30", u + 5.0, 400.0, 0.8), ("40", u - 5.0, 400.0, 0.8)]
    assert identify_lines(LINES, kps, territory="away") == {}


def test_identify_lines_rejects_far_assignment():
    # kp ~220px from the only line (line x at v=50 is ~76) -> no vote -> {}
    kps = [("30", 300.0, 50.0, 0.8)]
    ident = identify_lines([_seg_for(_yardline_x_m("away_30"))], kps,
                           territory="away")
    assert ident == {}


def test_fuse_frame_emits_intersections_and_numbers():
    # two voted lines (40 and 30) so the unvoted away_35 gets neighbor-filled
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 20.0
    u40 = _u_at(_yardline_x_m("away_40"), 100.0) - 10.0
    corrs = fuse_frame(LINES, _hashes(),
                       [("30", u30, 200.0, 0.8), ("40", u40, 100.0, 0.8)],
                       territory="away", image_size=(W, H))
    names = {n for (n, _uv) in corrs}
    # every identified line x both hash rows, named by the left=+Y=upper convention
    for base in ("away_40", "away_35", "away_30"):
        assert f"{base}_left_hash" in names       # upper row v=300
        assert f"{base}_right_hash" in names      # lower row v=700
    assert "away_30_left_number" in names          # the model number kp rides along
    # intersection precision: away_30 x upper row lands on the true line at v=300
    uv = dict(corrs)["away_30_left_hash"]
    assert abs(uv[0] - _u_at(_yardline_x_m("away_30"), 300.0)) < 1.0
    assert abs(uv[1] - 300.0) < 1.0


def test_fuse_frame_no_model_kps_returns_empty():
    assert fuse_frame(LINES, _hashes(), [], territory="away",
                      image_size=(W, H)) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fuse_pretrained.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.fuse_pretrained`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/fuse_pretrained.py
"""Fuse pretrained-model identity with classical geometry.

Model keypoints are coarse (±3-30 px) but carry yard identity; classical
slanted lines are precise but anonymous. Each model keypoint votes for the
classical line nearest in x at the keypoint's row; identified lines are
intersected with the RANSAC hash rows to produce precise labeled
correspondences. Ambiguity -> drop (frame gap), never guess.
"""
from __future__ import annotations

from nfl_gsplat.calibration.field_identify import fit_hash_rows, line_x_at
from nfl_gsplat.calibration.field_landmarks import (
    _yardline_x_m, yardline_name_from_x_m, YARD_LINE_SPACING_M,
)
from nfl_gsplat.calibration.roboflow_kps import to_nfl_name, yard_base

_SNAP_TOL_M = 0.35 * YARD_LINE_SPACING_M     # accept interpolated X within ~1.6 m


def identify_lines(yard_lines, model_kps, *, territory,
                   max_assign_px: float = 60.0, min_margin: float = 2.0):
    """Vote yard identity onto classical lines. Returns {line_idx: base_name}.

    Gates (fail toward fewer, correct lines):
      - a keypoint only votes if its nearest line is within ``max_assign_px``
        AND ``min_margin``x closer than the runner-up;
      - conflicting votes on one line -> that line (and, being unresolvable,
        the whole frame's identity) is dropped;
      - identified world-X must be strictly monotone in image x, else {}.
    Unvoted lines between/adjacent to voted ones get identity by world-X
    interpolation snapped via ``yardline_name_from_x_m`` (round-trip-safe).
    """
    if not yard_lines or not model_kps:
        return {}
    votes: dict[int, set[str]] = {}
    for (cls, u, v, _conf) in model_kps:
        base = yard_base(cls, territory)
        if base is None:
            continue
        dists = sorted((abs(line_x_at(s, v) - u), i)
                       for i, s in enumerate(yard_lines))
        d0, i0 = dists[0]
        if d0 > max_assign_px:
            continue
        if len(dists) > 1 and dists[1][0] < min_margin * d0:
            continue                                  # ambiguous between two lines
        votes.setdefault(i0, set()).add(base)

    ident: dict[int, str] = {}
    for i, bases in votes.items():
        if len(bases) != 1:
            return {}                                 # conflicting identity: drop frame
        ident[i] = next(iter(bases))
    if not ident:
        return {}

    # order all lines by x at a common row; voted world-X must be monotone
    ref_y = sum(p[1] for s in yard_lines for p in (s.p0, s.p1)) / (2 * len(yard_lines))
    order = sorted(range(len(yard_lines)), key=lambda i: line_x_at(yard_lines[i], ref_y))
    xs_img = [line_x_at(yard_lines[i], ref_y) for i in order]
    voted = [(k, ident[order[k]]) for k in range(len(order)) if order[k] in ident]
    Xw = [_yardline_x_m(b) for (_k, b) in voted]
    if len(voted) >= 2:
        inc = all(b > a for a, b in zip(Xw, Xw[1:]))
        dec = all(b < a for a, b in zip(Xw, Xw[1:]))
        if not (inc or dec):
            return {}                                 # inconsistent left-right ordering

    # fill unvoted lines by piecewise-linear world-X interpolation + snap
    if len(voted) >= 2:
        import numpy as np
        vk = [k for (k, _b) in voted]
        vX = np.array(Xw, float)
        vx = np.array([xs_img[k] for k in vk], float)
        if vx[0] > vx[-1]:
            vx, vX = vx[::-1], vX[::-1]
        for k in range(len(order)):
            if order[k] in ident:
                continue
            X_est = float(np.interp(xs_img[k], vx, vX))
            try:
                name = yardline_name_from_x_m(X_est, tol_m=_SNAP_TOL_M)
            except ValueError:
                continue                              # off-grid: leave unidentified
            if abs(_yardline_x_m(name) - X_est) <= _SNAP_TOL_M:
                ident[order[k]] = name
    return ident


def fuse_frame(yard_lines, hashes, model_kps, *, territory, image_size,
               max_assign_px: float = 60.0, min_margin: float = 2.0):
    """One frame -> labeled correspondences [(landmark_name, (u, v))]."""
    ident = identify_lines(yard_lines, model_kps, territory=territory,
                           max_assign_px=max_assign_px, min_margin=min_margin)
    if not ident:
        return []
    corrs: list[tuple[str, tuple[float, float]]] = []

    rows = fit_hash_rows(hashes, image_width=image_size[0])
    # rows are sorted upper-first; upper row = +Y = *_left_hash (convention
    # validated on real footage 2026-06). With only one row we cannot tell
    # upper from lower -> skip hash intersections (fail toward less, correct).
    if len(rows) == 2:
        for lr, row in (("left", rows[0]), ("right", rows[1])):
            for i, base in ident.items():
                seg = yard_lines[i]
                uv = _intersect(seg, row)
                if uv is not None:
                    corrs.append((f"{base}_{lr}_hash", uv))

    for (cls, u, v, _conf) in model_kps:
        name = to_nfl_name(cls, territory)
        if name is not None and name.endswith("_number"):
            corrs.append((name, (float(u), float(v))))
    return corrs


def _intersect(seg_a, seg_b):
    """Intersection of two YardLineSeg treated as infinite lines; None if parallel."""
    (x1, y1), (x2, y2) = seg_a.p0, seg_a.p1
    (x3, y3), (x4, y4) = seg_b.p0, seg_b.p1
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return (float((a * (x3 - x4) - (x1 - x2) * b) / d),
            float((a * (y3 - y4) - (y1 - y2) * b) / d))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fuse_pretrained.py -v`
Expected: all PASS. (`fit_hash_rows` on the dense synthetic ticks returns 2 rows; if the row fit needs more points, densify `_hashes()` in the test, not the thresholds.)

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — expected: all pass.

```bash
git add nfl_gsplat/calibration/fuse_pretrained.py tests/test_fuse_pretrained.py
git commit -m "feat(calibration): fuse pretrained identity with classical geometry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Orchestration + CLI (`build_autocalib_npz_pretrained`, `--mode pretrained`)

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (add `_register_sequence_pretrained` + `build_autocalib_npz_pretrained` after `build_autocalib_npz_learned`)
- Modify: `scripts/02_autocalibrate.py` (add `pretrained` mode)
- Test: `tests/test_run_autocalib.py` (append)

**Interfaces:**
- Consumes: `fuse_frame` (Task 5), `load_kps_json` (Task 2), `_register_corrs` + `assemble_track_from_results` + `write_camera_track` (existing in `run_autocalib.py`), `detect_field_features` (existing), `ffprobe_meta`/`iter_frames` (existing `nfl_gsplat.utils.video`).
- Produces:
  - `_register_sequence_pretrained(frames, *, kps_by_frame, territory, image_size, cfg=None, boxes_for=None) -> list[CalibrationResult | None]`
  - `build_autocalib_npz_pretrained(*, play_dir, videos, fps, kps_json, territory, cfg=None, masks_provider=None) -> Path` — mirrors `build_autocalib_npz_learned`.
  - CLI: `python scripts/02_autocalibrate.py --play-dir P --mode pretrained --roboflow-kps <json> --territory away`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_run_autocalib.py`)

```python
def test_pretrained_register_sequence_with_stub_fusion(monkeypatch):
    # Frames with cached model kps register; frames without -> None (gap).
    import numpy as np
    from nfl_gsplat.calibration import run_autocalib as ra

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        return DetectedFeatures(yard_lines=["L"], sidelines=[], hashes=[(1.0, 2.0)],
                                numbers=[], image_size=(1920, 1080))
    monkeypatch.setattr(ra, "detect_field_features", fake_detect, raising=False)

    def fake_fuse(yard_lines, hashes, model_kps, *, territory, image_size, **kw):
        return [(f"c{i}", (10.0 * i, 5.0)) for i in range(8)] if model_kps else []
    monkeypatch.setattr(ra, "fuse_frame", fake_fuse, raising=False)

    def fake_register_corrs(corrs, image_size, **kw):
        return object() if len(corrs) >= 6 else None
    monkeypatch.setattr(ra, "_register_corrs", fake_register_corrs)

    kps = {0: [("30", 1.0, 2.0, 0.9)], 2: [("30", 1.0, 2.0, 0.9)]}   # frame 1 absent
    results = ra._register_sequence_pretrained(
        ["f0", "f1", "f2"], kps_by_frame=kps, territory="away",
        image_size=(1920, 1080))
    assert results[0] is not None and results[2] is not None
    assert results[1] is None                       # no cached kps -> gap


def test_pretrained_none_frame_is_gap(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    results = ra._register_sequence_pretrained(
        [None], kps_by_frame={0: [("30", 1.0, 2.0, 0.9)]}, territory="away",
        image_size=(1920, 1080))
    assert results == [None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_autocalib.py -v -k pretrained`
Expected: FAIL — `AttributeError: ... has no attribute '_register_sequence_pretrained'`

- [ ] **Step 3: Implement** (add to `run_autocalib.py`; module-level imports of `fuse_frame` and `detect_field_features` so monkeypatching works)

Add near the top of `run_autocalib.py` with the other imports:

```python
from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame
```

(Remove the now-duplicate local import of `detect_field_features` inside `build_autocalib_npz`.)

Add after `build_autocalib_npz_learned`:

```python
def _register_sequence_pretrained(frames, *, kps_by_frame, territory, image_size,
                                  cfg=None, boxes_for=None):
    """Per frame: classical detect + cached model kps -> fuse -> PnP.
    Frames without cached keypoints (or unreadable) are gaps (None)."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig
    cfg = cfg or FieldDetectConfig()
    boxes_for = boxes_for or (lambda f: [])
    results = []
    for fidx, fr in enumerate(frames):
        kps = kps_by_frame.get(fidx, [])
        if fr is None or not kps:
            results.append(None)
            continue
        feats = detect_field_features(fr, cfg=cfg, player_boxes=boxes_for(fidx))
        corrs = fuse_frame(feats.yard_lines, feats.hashes, kps,
                           territory=territory, image_size=image_size)
        results.append(_register_corrs(corrs, image_size))
    return results


def build_autocalib_npz_pretrained(*, play_dir, videos, fps, kps_json, territory,
                                   cfg=None, masks_provider=None):
    """Pretrained-hybrid calibration: cached Roboflow identity + repaired
    classical geometry -> per-frame PnP -> cameras.npz. No GPU, no training."""
    from nfl_gsplat.calibration.roboflow_kps import load_kps_json
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    tracks = {}
    for cam, video in videos.items():
        meta = ffprobe_meta(video)
        kps_by_frame = load_kps_json(kps_json, expect_num_frames=meta.num_frames)
        boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
        results = [None] * meta.num_frames
        from nfl_gsplat.calibration.field_detect import FieldDetectConfig
        _cfg = cfg or FieldDetectConfig()
        for fidx, fr in iter_frames(video, start_frame=0):
            if not (0 <= fidx < meta.num_frames):
                continue
            kps = kps_by_frame.get(fidx, [])
            if not kps:
                continue
            feats = detect_field_features(fr, cfg=_cfg, player_boxes=boxes_for(fidx))
            corrs = fuse_frame(feats.yard_lines, feats.hashes, kps,
                               territory=territory,
                               image_size=(meta.width, meta.height))
            results[fidx] = _register_corrs(corrs, (meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width,
                                                  height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_autocalib.py -v`
Expected: all PASS (new + existing).

- [ ] **Step 5: Wire the CLI** (`scripts/02_autocalibrate.py`)

Add `pretrained = "pretrained"` to `CalibMode`; add options and branch:

```python
# added options in main(...):
roboflow_kps: Optional[Path] = typer.Option(None, "--roboflow-kps",
    help="Path to roboflow_kps.json (pretrained mode; from scripts/03_roboflow_precompute.py)."),
territory: str = typer.Option("away", "--territory",
    help="Which side of the 50 the visible yard numbers belong to (pretrained mode)."),
```

```python
    if mode is CalibMode.pretrained:
        if roboflow_kps is None:
            raise typer.BadParameter("--roboflow-kps is required in pretrained mode.")
        out = build_autocalib_npz_pretrained(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            kps_json=roboflow_kps, territory=territory,
        )
    elif mode is CalibMode.learned:
        ...  # unchanged
```

Update the import line to include `build_autocalib_npz_pretrained`.

- [ ] **Step 6: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — expected: all pass.

```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py
git commit -m "feat(calibration): pretrained-hybrid mode end to end (--mode pretrained)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Real-footage acceptance overlay (manual gate, no commit of imagery)

**Files:**
- Create: `scripts/diag_pretrained.py`

**Interfaces:**
- Consumes: `load_kps_json` (Task 2), `detect_field_features` (repaired, Task 4), `fuse_frame` (Task 5), `fit_plane_homography` or `cv2.findHomography`, `NFL_LANDMARKS`.
- Produces: per-frame overlay PNGs (cyan projected field grid + green fused correspondences) written OUTSIDE the repo, plus per-frame stats (n corrs, inliers, median residual px) on stdout. This is the acceptance test from the spec: grid must track the painted field on field-visible frames.

- [ ] **Step 1: Write the script**

```python
# scripts/diag_pretrained.py
"""Acceptance diagnostic for the pretrained-hybrid path (run locally).

    python scripts/diag_pretrained.py <frames_dir> <roboflow_kps.json> ^
        --territory away --out-dir C:\\Users\\<you>\\diag\\pretrained_overlays

Frames_dir holds frames named f%05d.png (as written by eval/precompute
sampling); indices are parsed from filenames to look up cached keypoints.
Overlays go OUTSIDE the repo. Grid on painted lines = ship it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.field_landmarks import (
    HALF_WIDTH_M, HASH_OFFSET_M, NFL_LANDMARKS, YARD_LINE_SPACING_M,
)
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame
from nfl_gsplat.calibration.roboflow_kps import load_kps_json


def _grid(img, Hm):
    out = img.copy()

    def to_img(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), Hm).reshape(2)
        return (int(round(p[0])), int(round(p[1])))

    for k in range(-10, 11):
        X = k * YARD_LINE_SPACING_M
        cv2.line(out, to_img(X, +HALF_WIDTH_M), to_img(X, -HALF_WIDTH_M),
                 (255, 200, 0), 1, cv2.LINE_AA)
    for Y in (+HASH_OFFSET_M, -HASH_OFFSET_M):
        cv2.line(out, to_img(-10 * YARD_LINE_SPACING_M, Y),
                 to_img(10 * YARD_LINE_SPACING_M, Y), (255, 120, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir")
    ap.add_argument("kps_json")
    ap.add_argument("--territory", default="away", choices=["home", "away"])
    ap.add_argument("--out-dir", required=True,
                    help="overlay output dir OUTSIDE the repo")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kps_by_frame = load_kps_json(args.kps_json)

    for fp in sorted(Path(args.frames_dir).glob("f*.png")):
        m = re.match(r"f(\d+)", fp.stem)
        if not m:
            continue
        fidx = int(m.group(1))
        kps = kps_by_frame.get(fidx, [])
        img = cv2.imread(str(fp))
        if img is None or not kps:
            print(f"{fp.name}: skipped (no image or no cached kps)")
            continue
        Hh, Ww = img.shape[:2]
        feats = detect_field_features(img)
        corrs = fuse_frame(feats.yard_lines, feats.hashes, kps,
                           territory=args.territory, image_size=(Ww, Hh))
        if len(corrs) < 4:
            print(f"{fp.name}: only {len(corrs)} correspondences — gap")
            continue
        world = np.array([NFL_LANDMARKS[n][:2] for (n, _uv) in corrs], np.float64)
        uv = np.array([p for (_n, p) in corrs], np.float64)
        Hm, mask = cv2.findHomography(world, uv, cv2.RANSAC, 5.0)
        if Hm is None:
            print(f"{fp.name}: homography failed")
            continue
        inl = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        res = np.linalg.norm(proj - uv, axis=1)
        print(f"{fp.name}: {len(corrs)} corrs, inliers {int(inl.sum())}/{len(corrs)}, "
              f"median {np.median(res[inl]):.2f}px")
        over = _grid(img, Hm)
        for (gu, gv) in uv:
            cv2.circle(over, (int(gu), int(gv)), 6, (0, 200, 0), 2)
        cv2.imwrite(str(out_dir / f"fused_{fp.name}"), over)
    print(f"\noverlays -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run against existing eval artifacts (user's machine has them)**

Run (Windows):
```
python scripts/03_roboflow_precompute.py "C:\Users\sumedh\diag\sideline.mp4" --out "C:\Users\sumedh\diag\roboflow_kps.json" --stride 1
python scripts/diag_pretrained.py kp_eval\frames "C:\Users\sumedh\diag\roboflow_kps.json" --territory away --out-dir "C:\Users\sumedh\diag\pretrained_overlays"
```
Expected: field-visible frames report ≥8 corrs, ≥6 inliers, median ≤2 px; overlays show the grid on the painted lines. THIS IS THE ACCEPTANCE GATE — a human must look at the overlays.

- [ ] **Step 3: Commit (script only — never the overlays/frames/JSON)**

```bash
git add scripts/diag_pretrained.py
git commit -m "feat(calibration): pretrained-hybrid acceptance diagnostic overlay

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
