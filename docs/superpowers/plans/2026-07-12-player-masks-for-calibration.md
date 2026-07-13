# Player Masks for Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mask players before field detection in pretrained calibration — precompute player boxes once per play, rotate them into the working view, feed them to the existing `masks_provider` seam — so the endzone camera's hash rows stop being corrupted and it calibrates.

**Architecture:** A local precompute script reuses `detect_and_track` (YOLOv8) to write `tracks.parquet`. `boxes_provider_from_tracks` turns that into the `masks_provider` the calibration pipeline already accepts. In `build_autocalib_npz_pretrained` the reject-on-rotation guard is replaced by real box rotation (`view_rotation.rotate_box`), so boxes mask the rotated endzone frame correctly. Sideline (deg=0) behavior is unchanged.

**Tech Stack:** numpy, pandas (parquet), OpenCV, existing calibration + tracking modules, ultralytics (precompute only, lazy import).

**Spec:** `docs/superpowers/specs/2026-07-12-player-masks-for-calibration-design.md`

## Global Constraints

- `cameras.npz` stays in TRUE original-pixel coordinates (unchanged; masking only affects detection inputs).
- Player boxes are stored/consumed in ORIGINAL-frame pixels; calibration rotates them by the per-camera `deg` (endzone → 90) before masking the rotated frame. Rotations ∈ {0,90,180,270}.
- Sideline (`deg == 0`) path must be byte-for-byte behavior-identical to current main when no masks are supplied, and additive (never regressive) when masks are supplied.
- Masking is an ENHANCEMENT: a missing boxes cache warns loudly and runs unmasked; it never half-applies and never silently changes which region is masked.
- Fail loud: `ultralytics` absent in the precompute → `SetupError` with the env pointer; invalid rotation → `ValueError` from `view_rotation._check`.
- `pytest -m "not gpu and not slow"` green; `ruff check nfl_gsplat tests scripts` clean. NEVER commit real NFL imagery/video; fixtures synthetic.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `rotate_box` — rotate a player box into the working (rotated) frame

**Files:**
- Modify: `nfl_gsplat/calibration/view_rotation.py` (add `rotate_box`)
- Test: `tests/test_view_rotation.py` (append)

**Interfaces:**
- Consumes: existing `rotate_uv(u, v, deg, orig_wh)`, `rotated_wh(deg, orig_wh)`, `_check(deg)` in the same module.
- Produces (Task 3 relies on this): `rotate_box(box, deg, orig_wh) -> tuple[float, float, float, float]` — maps an axis-aligned `(x1, y1, x2, y2)` box in the ORIGINAL (W,H) frame to the axis-aligned box in the rotated frame (min/max over the four rotated corners). `deg == 0` returns the box unchanged.

- [ ] **Step 1: Write the failing tests** (append to tests/test_view_rotation.py)

```python
@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_rotate_box_covers_rotated_corners(deg):
    from nfl_gsplat.calibration.view_rotation import rotate_box, rotate_uv
    box = (5.0, 3.0, 20.0, 12.0)          # x1,y1,x2,y2 in a (W=32,H=20) frame
    ow = (32, 20)
    rb = rotate_box(box, deg, ow)
    # every original corner, rotated, must lie inside the rotated box (± epsilon)
    corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    for (u, v) in corners:
        ru, rv = rotate_uv(u, v, deg, ow)
        assert rb[0] - 1e-6 <= ru <= rb[2] + 1e-6
        assert rb[1] - 1e-6 <= rv <= rb[3] + 1e-6
    assert rb[0] <= rb[2] and rb[1] <= rb[3]     # well-ordered


def test_rotate_box_zero_is_identity():
    from nfl_gsplat.calibration.view_rotation import rotate_box
    assert rotate_box((1.0, 2.0, 3.0, 4.0), 0, (32, 20)) == (1.0, 2.0, 3.0, 4.0)


def test_rotate_box_round_trip_90_270():
    from nfl_gsplat.calibration.view_rotation import rotate_box, rotated_wh
    box = (5.0, 3.0, 20.0, 12.0)
    ow = (32, 20)
    r90 = rotate_box(box, 90, ow)
    back = rotate_box(r90, 270, rotated_wh(90, ow))
    assert max(abs(a - b) for a, b in zip(back, box)) <= 1.0   # index-vs-center ≤1px
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_view_rotation.py -v -k rotate_box`
Expected: FAIL — `ImportError: cannot import name 'rotate_box'`

- [ ] **Step 3: Implement** (add to `view_rotation.py`, after `rotate_uv`)

```python
def rotate_box(box, deg: int, orig_wh) -> tuple[float, float, float, float]:
    """Map an axis-aligned (x1, y1, x2, y2) box in the ORIGINAL frame to the
    axis-aligned box in the rotated frame. 90/180/270 keep boxes axis-aligned,
    so the min/max over the four rotated corners is exact. Used to mask player
    boxes (stored in original pixels) on a rotated working frame."""
    _check(deg)
    if deg == 0:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    x1, y1, x2, y2 = box
    corners = [rotate_uv(x1, y1, deg, orig_wh), rotate_uv(x2, y1, deg, orig_wh),
               rotate_uv(x1, y2, deg, orig_wh), rotate_uv(x2, y2, deg, orig_wh)]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    return (float(min(us)), float(min(vs)), float(max(us)), float(max(vs)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_view_rotation.py -v -k rotate_box`
Expected: 3 PASS

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/view_rotation.py tests/test_view_rotation.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/view_rotation.py tests/test_view_rotation.py
git commit -m "feat(calibration): rotate_box maps player boxes into the rotated view

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `boxes_provider_from_tracks` — tracks.parquet → masks_provider

**Files:**
- Create: `nfl_gsplat/calibration/player_masks.py`
- Test: `tests/test_player_masks.py` (new)

**Interfaces:**
- Consumes: `TRACK_COLUMNS` (`nfl_gsplat/tracking/detect_track.py`: columns include `frame`, `cam`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`); pandas.
- Produces (Task 3 relies on this): `boxes_provider_from_tracks(tracks_path) -> Callable[[str], Callable[[int], list[tuple[float,float,float,float]]]]` — loads the parquet once; returns a `masks_provider(cam)` that returns a `boxes_for(frame_idx)` yielding `(x1,y1,x2,y2)` original-pixel boxes for that camera+frame (empty list if none). Raises `SetupError` if the file is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_masks.py
import pandas as pd
import pytest

from nfl_gsplat.calibration.player_masks import boxes_provider_from_tracks
from nfl_gsplat.errors import SetupError
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS


def _tracks(tmp_path):
    rows = [
        {"frame": 0, "cam": "sideline", "bbox_x1": 10.0, "bbox_y1": 20.0,
         "bbox_x2": 30.0, "bbox_y2": 60.0},
        {"frame": 0, "cam": "endzone", "bbox_x1": 1.0, "bbox_y1": 2.0,
         "bbox_x2": 3.0, "bbox_y2": 4.0},
        {"frame": 2, "cam": "sideline", "bbox_x1": 5.0, "bbox_y1": 6.0,
         "bbox_x2": 7.0, "bbox_y2": 8.0},
    ]
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c != "cam" else ""
    p = tmp_path / "tracks.parquet"
    df.to_parquet(p, index=False)
    return p


def test_provider_yields_boxes_per_cam_and_frame(tmp_path):
    prov = boxes_provider_from_tracks(_tracks(tmp_path))
    sl = prov("sideline")
    assert sl(0) == [(10.0, 20.0, 30.0, 60.0)]
    assert sl(2) == [(5.0, 6.0, 7.0, 8.0)]
    assert sl(1) == []                                  # no detection that frame
    ez = prov("endzone")
    assert ez(0) == [(1.0, 2.0, 3.0, 4.0)]
    assert ez(9) == []


def test_provider_unknown_cam_empty(tmp_path):
    prov = boxes_provider_from_tracks(_tracks(tmp_path))
    assert prov("skycam")(0) == []


def test_provider_missing_file_fails_loud(tmp_path):
    with pytest.raises(SetupError, match="tracks"):
        boxes_provider_from_tracks(tmp_path / "nope.parquet")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_player_masks.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.player_masks`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/player_masks.py
"""Turn a per-play tracks.parquet into the calibration masks_provider seam.

Player boxes (original-frame pixels) are zeroed out of the white mask before
line/hash detection, so players' bright uniforms don't create false field
markings (measured: unmasked players make fit_hash_rows fit diagonal-garbage
hash rows on the endzone view). Boxes are indexed by (cam, frame); calibration
rotates them per-camera before masking the rotated working frame.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from nfl_gsplat.errors import SetupError

Box = tuple[float, float, float, float]


def boxes_provider_from_tracks(
    tracks_path,
) -> Callable[[str], Callable[[int], list[Box]]]:
    """Load tracks.parquet once; return masks_provider(cam) -> boxes_for(frame)."""
    import pandas as pd

    p = Path(tracks_path)
    if not p.exists():
        raise SetupError(
            f"player tracks not found: {p} — run scripts/03b_detect_players.py "
            "on the play first (writes tracks.parquet)."
        )
    df = pd.read_parquet(p)
    by_cam: dict[str, dict[int, list[Box]]] = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples(index=False):
        by_cam[str(r.cam)][int(r.frame)].append(
            (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)))

    def masks_provider(cam: str) -> Callable[[int], list[Box]]:
        per_frame = by_cam.get(cam, {})
        return lambda fidx: per_frame.get(int(fidx), [])

    return masks_provider
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_player_masks.py -v`
Expected: all PASS

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/player_masks.py tests/test_player_masks.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/player_masks.py tests/test_player_masks.py
git commit -m "feat(calibration): boxes_provider_from_tracks (tracks.parquet -> masks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire masks + box rotation into pretrained calibration + CLI

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (`build_autocalib_npz_pretrained`, ~lines 447-456; import)
- Modify: `scripts/02_autocalibrate.py` (build masks_provider from a boxes cache)
- Test: `tests/test_run_autocalib.py` (append)

**Interfaces:**
- Consumes: `rotate_box` (Task 1), `boxes_provider_from_tracks` (Task 2), existing `_camera_rotation`, `masks_provider` param.
- Produces: `build_autocalib_npz_pretrained` masks the ROTATED frame with per-camera-rotated boxes; the `deg != 0` reject-guard is removed. CLI resolves `<play_dir>/tracks.parquet` (or `--player-boxes PATH`) into a masks_provider; missing → warn + no masking.

- [ ] **Step 1: Write the failing tests** (append to tests/test_run_autocalib.py)

```python
def test_pretrained_rotates_player_boxes_for_masking(monkeypatch):
    # boxes are original-pixel; on a rotated (endzone) camera they must reach
    # detect_field_features rotated into the working frame
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.view_rotation import rotate_box

    captured = {"boxes": []}

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter(enumerate(frames)))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {0: [("30", 10.0, 20.0, 0.9)],
                                                           1: [("30", 10.0, 20.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured["boxes"].append(player_boxes)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[],
                                numbers=[], image_size=frame.shape[:2][::-1])
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)
    monkeypatch.setattr(ra, "fuse_frame",
                        lambda *a, **k: [(f"c{i}", (float(i), 0.0)) for i in range(8)])
    monkeypatch.setattr(ra, "_solve_sweep", lambda *a, **k: [None, None], raising=False)
    monkeypatch.setattr(ra, "solve_fixed_center", lambda *a, **k: (["R0", "R1"], False))
    monkeypatch.setattr(ra, "derotate_result", lambda r, deg, wh: r)
    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: ("TRACK", width, height))
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    orig_box = (100.0, 200.0, 150.0, 400.0)
    masks = lambda cam: (lambda fidx: [orig_box])
    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"endzone": "e.mp4"}, fps=30.0,
        kps_json={"endzone": "e.json"}, territory="away",
        masks_provider=masks, rotations={"endzone": 90})
    assert captured["boxes"][0] == [rotate_box(orig_box, 90, (1920, 1080))]


def test_pretrained_sideline_boxes_unrotated(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    captured = {"boxes": []}

    class _Meta:
        num_frames, width, height = 1, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([(0, np.zeros((1080, 1920, 3), np.uint8))]))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {0: [("30", 1.0, 2.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured["boxes"].append(player_boxes)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[], numbers=[],
                                image_size=(1920, 1080))
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)
    monkeypatch.setattr(ra, "fuse_frame", lambda *a, **k: [("c", (1.0, 0.0))])
    monkeypatch.setattr(ra, "_solve_sweep", lambda *a, **k: [None], raising=False)
    monkeypatch.setattr(ra, "solve_fixed_center", lambda *a, **k: (["R0"], False))
    monkeypatch.setattr(ra, "derotate_result", lambda r, deg, wh: r)
    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: "TRACK")
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    box = (10.0, 20.0, 30.0, 40.0)
    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"sideline": "s.mp4"}, fps=30.0,
        kps_json={"sideline": "s.json"}, territory="away",
        masks_provider=lambda cam: (lambda f: [box]))    # rotations=None -> sideline deg 0
    assert captured["boxes"][0] == [box]                 # unrotated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_autocalib.py -v -k player_boxes`
Expected: FAIL — the current code raises `CalibrationError` (reject-guard) for the rotated case.

- [ ] **Step 3: Implement** — in `run_autocalib.py`:

Add `rotate_box` to the `view_rotation` import (line ~22):

```python
from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_box, rotate_image, rotate_uv, rotated_wh,
)
```

Replace the reject-guard block (the `if masks_provider is not None and deg != 0: raise ...` and the following `boxes_for = ...` line) with:

```python
            raw_boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
            if deg != 0:
                # Boxes are original-pixel; rotate them into the working frame so
                # they mask the rotated image the detector actually sees.
                def boxes_for(fidx, _bf=raw_boxes_for, _deg=deg, _owh=orig_wh):
                    return [rotate_box(b, _deg, _owh) for b in _bf(fidx)]
            else:
                boxes_for = raw_boxes_for
```

(The existing `detect_field_features(fr, cfg=_cfg, player_boxes=boxes_for(fidx))` call at line ~469 is unchanged; `fr` is already rotated, and `boxes_for` now yields rotated boxes.)

- [ ] **Step 4: Wire the CLI** (`scripts/02_autocalibrate.py`) — in the `pretrained` branch, after `kps_json` is resolved, build the masks provider:

```python
        from nfl_gsplat.calibration.player_masks import boxes_provider_from_tracks
        boxes_path = player_boxes if player_boxes is not None else (pd.dir / "tracks.parquet")
        if boxes_path.exists():
            masks_provider = boxes_provider_from_tracks(boxes_path)
        else:
            _LOG.warning("no player tracks at %s — running calibration UNMASKED "
                         "(endzone likely fails; run scripts/03b_detect_players.py)",
                         boxes_path)
            masks_provider = None
        out = build_autocalib_npz_pretrained(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            kps_json=kps_json, territory=territory,
            rotations=rotations or None, masks_provider=masks_provider,
        )
```

and add the option to `main(...)`:

```python
    player_boxes: Optional[Path] = typer.Option(None, "--player-boxes",
        help="Path to tracks.parquet for player masking (pretrained mode; "
             "default <play_dir>/tracks.parquet if present)."),
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_run_autocalib.py tests/test_view_rotation.py tests/test_player_masks.py -v`
Expected: all PASS (the old `test_pretrained_rotated_camera_rejects_masks_provider` test asserted the reject-guard — DELETE it; its behavior is replaced by box rotation. Note the deletion in the commit body).
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.
Run: `python -m ruff check nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py
git commit -m "feat(calibration): mask players with view-rotated boxes in pretrained mode

Replaces the reject-on-rotation guard with real box rotation: player boxes
(original pixels) are rotated by the camera's view deg before masking the
rotated frame. CLI builds masks_provider from <play_dir>/tracks.parquet
(missing -> warn + unmasked). Sideline (deg=0) unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `03b_detect_players.py` — local player-box precompute

**Files:**
- Create: `scripts/03b_detect_players.py`
- Modify: `nfl_gsplat/tracking/detect_track.py` (add a `--detect-only` per-frame path: `detect_only(video, cam, cfg)`)
- Test: `tests/test_detect_track.py` (append a stubbed test for `detect_only` frame→rows mapping)

**Interfaces:**
- Consumes: existing `detect_and_track`, `TrackingConfig`, `TRACK_COLUMNS`, `foot_point_from_bbox`, `empty_tracks`.
- Produces: `detect_only(video, cam, cfg, *, frame_source=None) -> pd.DataFrame` — per-frame YOLO detection (no BoT-SORT), `track_id = -1`, same `TRACK_COLUMNS`. `frame_source` is an injectable iterator of `(idx, bgr)` for tests (default: real video). CLI `scripts/03b_detect_players.py <play_dir> [--detect-only] [--weights yolov8n.pt]` writes `<play_dir>/tracks.parquet`.

- [ ] **Step 1: Write the failing test** (append to tests/test_detect_track.py; create the file if absent, mirroring existing tracking tests)

```python
def test_detect_only_maps_boxes_to_rows(monkeypatch):
    import numpy as np
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS, TrackingConfig, detect_only

    # a fake per-frame detector: frame 0 has one box, frame 1 none
    class _Box:
        def __init__(self, xyxy, conf):
            import numpy as _np
            self.xyxy = _np.array([xyxy], float)
            self.conf = _np.array([conf], float)

    def fake_predict(bgr):
        return _Box([10.0, 20.0, 30.0, 60.0], 0.9) if bgr.sum() == 0 else None

    frames = [(0, np.zeros((8, 8, 3), np.uint8)), (1, np.ones((8, 8, 3), np.uint8))]
    df = detect_only("v.mp4", "endzone", TrackingConfig(),
                     frame_source=iter(frames), _predict=fake_predict)
    assert list(df.columns) == TRACK_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert (row.frame, row.cam, row.track_id) == (0, "endzone", -1)
    assert (row.bbox_x1, row.bbox_y2) == (10.0, 60.0)
    assert row.foot_v == 60.0 and row.foot_u == 20.0     # bottom-center
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detect_track.py -v -k detect_only`
Expected: FAIL — `ImportError: cannot import name 'detect_only'`

- [ ] **Step 3: Implement `detect_only`** (add to `detect_track.py`)

```python
def detect_only(video, cam, cfg, *, frame_source=None, _predict=None):
    """Per-frame YOLO person detection (no BoT-SORT). track_id=-1. For masking,
    which needs boxes, not track continuity — and runs on CPU without a tracker.
    ``frame_source`` / ``_predict`` are test injection seams."""
    if _predict is None:                                    # pragma: no cover (gpu path)
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise SetupError(
                "ultralytics not installed — activate the `nfl_smplx` conda env. "
                "See SETUP.md §1.") from e
        model = YOLO(cfg.yolo_weights)

        def _predict(bgr):
            res = model.predict(bgr, classes=[cfg.person_class_id],
                                conf=cfg.min_detection_conf, verbose=False)[0]
            return res.boxes if (res.boxes is not None and len(res.boxes)) else None
    if frame_source is None:                                # pragma: no cover (gpu path)
        from nfl_gsplat.utils.video import iter_frames
        import cv2
        frame_source = ((i, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                        for i, fr in iter_frames(video, start_frame=0))
    rows: list[dict] = []
    for idx, bgr in frame_source:
        boxes = _predict(bgr)
        if boxes is None:
            continue
        xyxy = boxes.xyxy if hasattr(boxes, "xyxy") else boxes
        confs = getattr(boxes, "conf", [1.0] * len(xyxy))
        for b, c in zip(np.asarray(xyxy, float), np.asarray(confs, float)):
            u, v = foot_point_from_bbox(np.asarray(b, dtype=np.float64))
            rows.append({
                "frame": int(idx), "cam": cam, "track_id": -1,
                "global_player_id": -1,
                "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
                "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                "conf": float(c), "foot_u": float(u), "foot_v": float(v),
                "jersey_number_ocr": -1})
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return _coerce_dtypes(df)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_detect_track.py -v -k detect_only`
Expected: PASS

- [ ] **Step 5: Write the CLI script** (no automated test; thin wiring)

```python
# scripts/03b_detect_players.py
"""Detect players per camera for one play -> <play_dir>/tracks.parquet (local).

Feeds player MASKS to calibration (players' bright uniforms otherwise corrupt
hash-mark detection, esp. on the endzone view). --detect-only skips BoT-SORT
(faster on CPU; masking needs boxes, not track IDs).

    python scripts/03b_detect_players.py data/2025/week_04/SEA_at_AZ/play_001 \
        --detect-only --weights yolov8n.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.tracking.detect_track import (
    TrackingConfig, detect_and_track, detect_only, empty_tracks,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("--detect-only", action="store_true",
                    help="per-frame detection, no BoT-SORT (faster on CPU)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cameras", default="sideline,endzone")
    args = ap.parse_args()

    cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pd_ = PlayDir.from_dir(args.play_dir, cameras=cams)
    cfg = TrackingConfig(yolo_weights=args.weights, min_detection_conf=args.conf,
                         device=args.device)
    dfs = []
    for cam in pd_.cameras:
        video = pd_.video(cam)
        if not Path(video).exists():
            print(f"skip {cam}: no video at {video}")
            continue
        fn = detect_only if args.detect_only else detect_and_track
        dfs.append(fn(video, cam, cfg))
        print(f"{cam}: {len(dfs[-1])} detections")
    df = pd.concat(dfs, ignore_index=True) if dfs else empty_tracks()
    df.to_parquet(pd_.tracks, index=False)
    print(f"wrote {len(df)} detections -> {pd_.tracks}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Ruff + full suite + commit**

Run: `python -m ruff check scripts/03b_detect_players.py nfl_gsplat/tracking/detect_track.py tests/test_detect_track.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add scripts/03b_detect_players.py nfl_gsplat/tracking/detect_track.py tests/test_detect_track.py
git commit -m "feat(tracking): detect_only per-frame path + 03b player-box precompute

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Real-footage acceptance — precompute, calibrate, measure endzone

**Files:** none (manual acceptance; controller/human gate). Needs `ultralytics` installed locally.

- [ ] **Step 1: Install the detector + precompute boxes**

Run (Windows):
```
pip install ultralytics
python scripts/03b_detect_players.py "data\2025\week_04\SEA_at_AZ\play_001" --detect-only --weights yolov8n.pt --device cpu
```
Expected: `wrote N detections -> ...\tracks.parquet` (both cameras). If CPU is too slow on the full clip, that's acceptable for one play; note the runtime.

- [ ] **Step 2: Two-camera calibration WITH masks**

```
python scripts/02_autocalibrate.py --play-dir data\2025\week_04\SEA_at_AZ\play_001 --mode pretrained --territory away
```
(The CLI auto-finds `tracks.parquet`.)

- [ ] **Step 3: Judge the result — two acceptance outcomes**

Outcome A (success): `cameras.npz` written with `sideline_*` AND `endzone_*`.
Verify (adapt npz key names to `cameras_io`):
```
python -c "import numpy as np; d=np.load(r'data\2025\week_04\SEA_at_AZ\play_001\cameras.npz'); \
R=d['endzone_R']; t=d['endzone_t']; K=d['endzone_K']; c=d['endzone_conf']; \
C=np.einsum('nij,ni->nj', R, -t); ok=c>0; \    # center = -R^T t
print('endzone kept', int(ok.sum()), 'center', C[ok].mean(0).round(1), 'fx', K[ok,0,0].min().round(), K[ok,0,0].max().round())"
```
Assert endzone center in |X| 50–150, |Y| ≤ 40, Z 10–80; sideline unchanged
(center ≈ (−3.6, 80.5, 35.9), ~816 kept). Spot-check an endzone overlay:
project the field through the de-rotated `endzone_*` K,R,t onto an ORIGINAL
endzone frame (reuse the overlay pattern from `scripts/diag_pretrained.py`).

Outcome B (endzone still short): masking fixed the hash rows (verify: re-run
the scratchpad `draw_ez_overlay` and confirm hash rows are now ≈horizontal)
but the joint solve still fails its gate. Record the masked usable-frame count.
This is the measured trigger for the spec's contingent step — wire
`label_lines_by_consensus` as an endzone fusion mode — as a FOLLOW-UP cycle,
not part of this plan. Either way, Tasks 1–4 merge (they are correct,
tested infrastructure and fix the reject-guard).

- [ ] **Step 4: Record the outcome** in the branch's progress notes and (on
  merge) update the endzone memory with the masked measurement.
