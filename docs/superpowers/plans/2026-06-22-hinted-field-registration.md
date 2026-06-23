# Hinted Per-Frame Field Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-working number-OCR yardage step with a single human yardage hint per play per camera (in `meta.yaml`), plus real player-masked line detection + hash detection, so per-frame field registration actually produces `cameras.npz` on real footage.

**Architecture:** Auto-detect yard lines (player-masked) + hash ticks per frame. A per-camera `CalibHint` seeds absolute yardage at one reference frame; identity propagates across frames by line-continuity (forward+backward sweep). Per-frame PnP from hash×line / sideline×line correspondences. Self-validating via reprojection RMS. Reuses `solve_pnp`, `NFL_LANDMARKS`, `CameraTrack`, temporal smoothing, all consumers.

**Tech Stack:** Python 3.10, numpy, OpenCV (HoughLinesP, connectedComponents), scipy, typer, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-22-hinted-field-registration-design.md`

## Conventions
- Repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`. `python -m pytest …` locally.
- `NFL_LANDMARKS` names: `{side}_{yard}_{lr}_{row}` (e.g. `home_30_left_hash`, `mid_50_left_sideline`). `field_features.landmark_name(side, yd, lr, row)` builds them.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Phase 1 — Calibration hint in meta.yaml

### Task 1: `CalibHint` + meta parsing

**Files:**
- Modify: `nfl_gsplat/utils/meta.py`
- Test: `tests/test_meta.py` (extend)

- [ ] **Step 1: Add tests to `tests/test_meta.py`:**

```python
def test_load_meta_parses_calib_hints(tmp_path):
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text(
        'season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 59.94\n'
        "calib_hints:\n"
        "  sideline: {ref_frame: 0, ref_x: 866, yard: 30, side: away, increasing: right}\n"
        "  endzone:  {ref_frame: 5, ref_x: 540, yard: 50, side: mid, increasing: left}\n"
    )
    m = load_meta(p)
    assert set(m.calib_hints) == {"sideline", "endzone"}
    h = m.calib_hints["sideline"]
    assert (h.ref_frame, h.ref_x, h.yard, h.side, h.increasing) == (0, 866.0, 30, "away", "right")
    assert m.calib_hints["endzone"].side == "mid"


def test_calib_hints_default_empty(tmp_path):
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text('season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 30\n')
    assert load_meta(p).calib_hints == {}


def test_calib_hint_bad_side_raises(tmp_path):
    import pytest
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text(
        'season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 30\n'
        "calib_hints:\n  sideline: {ref_frame: 0, ref_x: 5, yard: 30, side: nope, increasing: right}\n"
    )
    with pytest.raises(SetupError, match="side"):
        load_meta(p)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_meta.py -q` → the 3 new tests FAIL (no `calib_hints`).

- [ ] **Step 3: Edit `nfl_gsplat/utils/meta.py`.** Add the dataclass + parsing. After the `PlayMeta` dataclass add:

```python
@dataclass(frozen=True)
class CalibHint:
    ref_frame: int
    ref_x: float
    yard: int
    side: str          # home | away | mid
    increasing: str    # left | right (image direction yards increase)
```
Add `calib_hints: dict[str, "CalibHint"] = field(default_factory=dict)` to `PlayMeta` (import `field` from dataclasses). In `load_meta`, after building the other fields, parse:

```python
    hints: dict[str, CalibHint] = {}
    raw_hints = raw.get("calib_hints") or {}
    for cam, h in raw_hints.items():
        side = str(h["side"])
        inc = str(h["increasing"])
        yard = int(h["yard"])
        if side not in ("home", "away", "mid"):
            raise SetupError(f"{path}: calib_hints.{cam}.side must be home/away/mid, got {side!r}.")
        if inc not in ("left", "right"):
            raise SetupError(f"{path}: calib_hints.{cam}.increasing must be left/right, got {inc!r}.")
        if side == "mid":
            yard = 50
        elif yard < 5 or yard > 45 or yard % 5 != 0:
            raise SetupError(f"{path}: calib_hints.{cam}.yard {yard} invalid (5..45 step 5, or mid=50).")
        hints[str(cam)] = CalibHint(
            ref_frame=int(h["ref_frame"]), ref_x=float(h["ref_x"]),
            yard=yard, side=side, increasing=inc,
        )
```
and pass `calib_hints=hints` into the returned `PlayMeta(...)`.

- [ ] **Step 4: Run** `python -m pytest tests/test_meta.py -q` → PASS (existing + 3 new).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/utils/meta.py tests/test_meta.py
git add nfl_gsplat/utils/meta.py tests/test_meta.py
git commit -m "meta.yaml: add per-camera calib_hints (CalibHint) parsing + validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — Hint-seeded identification

### Task 2: `assign_from_hint` + `seed_state_from_hint` (replace OCR seed)

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (replace OCR tests with hint tests)

Current `field_identify.py` has `_assign_from_numbers(lines_sorted, numbers)` (OCR) and `identify_correspondences(feats, prior)` (which calls `_assign_from_numbers` then falls back to `prior`). We swap the OCR seed for a hint seed; `identify_correspondences` becomes **pure propagation** (uses `prior` only), and a new `seed_state_from_hint` produces the initial `IdentityState` from a `CalibHint`.

- [ ] **Step 1: Replace the OCR tests in `tests/test_field_identify.py`** (remove `test_identify_assigns_yardage_from_numbers`, `test_fold_direction_when_decreasing`, and the `OCRNumber`-based fixtures; keep `test_no_number_and_no_prior_returns_empty` renamed) with:

```python
from __future__ import annotations

from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
from nfl_gsplat.calibration.field_identify import (
    IdentityState, identify_correspondences, seed_state_from_hint,
)
from nfl_gsplat.utils.meta import CalibHint


def _vline(x, H=1080):
    return YardLineSeg(p0=(float(x), 0.0), p1=(float(x), float(H)))


def _feats(xs, hashes=None, sidelines=None):
    return DetectedFeatures(
        yard_lines=[_vline(x) for x in xs],
        sidelines=sidelines or [YardLineSeg((0, 200), (1920, 220)),
                                YardLineSeg((0, 900), (1920, 950))],
        hashes=hashes or [],
        numbers=[],
        image_size=(1920, 1080),
    )


def test_seed_from_hint_labels_by_spacing_and_direction():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    labels = set(state.line_yardage.values())
    # x=800 is the away_30; increasing right => x=1200 is away_25, x=400 is away_35.
    assert ("away", 30) in labels
    assert ("away", 25) in labels and ("away", 35) in labels


def test_seed_crosses_50_to_home_when_increasing():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=45, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    labels = set(state.line_yardage.values())
    # away_45 at 800; next right (x=1200) crosses midfield -> mid_50.
    assert ("away", 45) in labels
    assert ("mid", 50) in labels


def test_identify_propagates_from_prior_with_hashes():
    feats = _feats([400, 800, 1200], hashes=[(400, 560), (800, 560), (1200, 560)])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, state = identify_correspondences(feats, state0)
    names = {c[0] for c in corrs}
    assert any(n.startswith("away_30") for n in names)
    # next frame shifted +10px, lines carry identity from prior
    shifted = _feats([410, 810, 1210], hashes=[(410, 560), (810, 560), (1210, 560)])
    corrs2, _ = identify_correspondences(shifted, state)
    assert any(n.startswith("away_30") for n in {c[0] for c in corrs2})


def test_identify_without_prior_returns_empty():
    feats = _feats([400])
    corrs, state = identify_correspondences(feats, None)
    assert corrs == [] and not state.line_yardage
```

- [ ] **Step 2: Run** → FAIL (`seed_state_from_hint` missing; OCR funcs referenced gone).

- [ ] **Step 3: Edit `nfl_gsplat/calibration/field_identify.py`.** Remove `_assign_from_numbers`. Add the hint seed + make `identify_correspondences` propagation-only. Keep `IdentityState`, `_line_x`, `_seg_intersection`, and the correspondence-emitting tail.

```python
def _yard_step(side: str, yard: int, step: int) -> tuple[str, int]:
    """Move ``step`` yard-LINES (×5 yd) from (side, yard) toward higher 'number'
    direction, folding across midfield. ``step`` may be negative."""
    # Signed position along the field in yard-line units from the away goal (0)
    # to home goal (20); away_5=1 ... away_45=9, mid_50=10, home_45=11 ... home_5=19.
    if side == "mid":
        pos = 10
    elif side == "away":
        pos = yard // 5
    else:  # home
        pos = 20 - yard // 5
    pos += step
    if pos < 1 or pos > 19:
        return ("", 0)            # off the field
    if pos == 10:
        return ("mid", 50)
    if pos < 10:
        return ("away", pos * 5)
    return ("home", (20 - pos) * 5)


def seed_state_from_hint(feats, hint) -> IdentityState:
    """Build the initial IdentityState for ``hint.ref_frame`` from a CalibHint.

    Snap ``ref_x`` to the nearest detected yard line, label it (side, yard), and
    label the rest by the constant index spacing. ``increasing`` says which image
    direction (left/right) the yard 'numbers' grow, which sets the per-index step.
    """
    lines = sorted(feats.yard_lines, key=_line_x)
    if not lines:
        return IdentityState()
    xs = [_line_x(s) for s in lines]
    seed_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - hint.ref_x))
    step_per_index = 1 if hint.increasing == "right" else -1
    out: dict[float, tuple[str, int]] = {}
    for i, s in enumerate(lines):
        side, yard = _yard_step(hint.side, hint.yard, step_per_index * (i - seed_idx))
        if side:
            out[_line_x(s)] = (side, yard)
    return IdentityState(line_yardage=out)


def identify_correspondences(feats, prior):
    """Propagate yard-line identity from ``prior`` to this frame's lines (by
    nearest image-x) and emit ``[(landmark_name, (u,v))]`` at hash/sideline
    intersections. Returns (correspondences, new IdentityState). With no prior
    (or empty), returns ([], empty) — identity must be seeded by a hint."""
    lines = sorted(feats.yard_lines, key=_line_x)
    if not lines or prior is None or not prior.line_yardage:
        return [], IdentityState()
    import numpy as np
    prior_xs = np.array(list(prior.line_yardage.keys()))
    prior_vals = list(prior.line_yardage.values())
    corrs: list[tuple[str, tuple[float, float]]] = []
    state_map: dict[float, tuple[str, int]] = {}
    for seg in lines:
        x = _line_x(seg)
        j = int(np.argmin(np.abs(prior_xs - x)))
        if abs(prior_xs[j] - x) > 60.0:          # no matching prior line nearby
            continue
        side, yd = prior_vals[j]
        state_map[x] = (side, yd)
        for sl in feats.sidelines:
            pt = _seg_intersection(seg, sl)
            if pt is None:
                continue
            lr = "left" if pt[1] < feats.image_size[1] / 2 else "right"
            corrs.append((landmark_name(side, yd, lr, "sideline"), pt))
        for hx, hy in feats.hashes:
            if abs(hx - x) < 25.0:
                lr = "left" if hy < feats.image_size[1] / 2 else "right"
                corrs.append((landmark_name(side, yd, lr, "hash"), (float(hx), float(hy))))
    seen: set[str] = set()
    deduped = []
    for name, uv in corrs:
        if name not in seen:
            seen.add(name); deduped.append((name, uv))
    return deduped, IdentityState(line_yardage=state_map)
```
Keep the existing `landmark_name` import + the left/right camera-side comment.

- [ ] **Step 4: Run** `python -m pytest tests/test_field_identify.py -q` → PASS (4 tests). If `_yard_step` fold logic fails a test, debug it against the contract (away_30 + right → neighbors away_25/away_35; away_45 + right → mid_50). Keep it pure.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: hint-seeded yardage + propagation-only correspondences (drop OCR)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Real detection (hashes + player masking)

### Task 3: real `_detect_hashes` + player-masked `detect_lines`

**Files:**
- Modify: `nfl_gsplat/calibration/field_detect.py`
- Test: `tests/test_field_detect.py` (extend)

- [ ] **Step 1: Add tests to `tests/test_field_detect.py`:**

```python
def test_detect_hashes_groups_two_rows():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_hashes
    img = np.full((720, 1280, 3), (40, 120, 40), np.uint8)
    for x in range(200, 1100, 90):                 # upper hash row at y=300
        cv2.rectangle(img, (x, 298), (x + 10, 306), (240, 240, 240), -1)
    for x in range(200, 1100, 90):                 # lower hash row at y=430
        cv2.rectangle(img, (x, 428), (x + 10, 436), (240, 240, 240), -1)
    pts = detect_hashes(img, FieldDetectConfig())
    ys = sorted(p[1] for p in pts)
    assert len(pts) >= 16
    # two distinct y bands
    assert max(ys) - min(ys) > 100
    assert any(abs(y - 302) < 15 for _, y in pts) and any(abs(y - 432) < 15 for _, y in pts)


def test_detect_lines_masks_player_box():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines
    img = np.full((720, 1280, 3), (40, 120, 40), np.uint8)
    cv2.line(img, (600, 60), (600, 660), (240, 240, 240), 4)     # one real yard line
    cv2.rectangle(img, (300, 200), (360, 480), (250, 250, 250), -1)  # white "jersey" blob
    # without mask the blob may add a spurious line; with the mask it's removed
    masked = detect_lines(img, FieldDetectConfig(), player_boxes=[(300, 200, 360, 480)])
    xs = [round(0.5 * (s.p0[0] + s.p1[0])) for s in masked]
    assert any(abs(x - 600) < 25 for x in xs)            # real line kept
    assert not any(abs(x - 330) < 25 for x in xs)        # jersey blob removed
```

- [ ] **Step 2: Run** → FAIL (`detect_hashes` missing; `detect_lines` has no `player_boxes`).

- [ ] **Step 3: Edit `nfl_gsplat/calibration/field_detect.py`.** Add hash config + masking. Extend `FieldDetectConfig` with hash params; update `detect_lines` to accept `player_boxes`; add `detect_hashes`; delete `_ocr_numbers`; update `detect_field_features` to take `player_boxes` and call `detect_hashes`.

```python
@dataclass(frozen=True)
class FieldDetectConfig:
    white_thresh: int = 180
    min_line_len_frac: float = 0.25
    max_line_gap_px: int = 30
    vertical_deg: float = 35.0
    hash_min_area: int = 8
    hash_max_area: int = 400
    hash_max_h_px: int = 22       # ticks are short
    hash_row_gap_px: int = 40     # min y-gap between the two rows


def _zero_boxes(mask, player_boxes):
    if not player_boxes:
        return mask
    out = mask.copy()
    for x1, y1, x2, y2 in player_boxes:
        out[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0
    return out


def detect_lines(img_bgr, cfg, player_boxes=None):
    """Near-vertical painted yard lines via HoughLinesP, with player boxes
    removed from the white mask first (kills jersey over-detection)."""
    import numpy as np
    H = img_bgr.shape[0]
    mask = _zero_boxes(_white_mask(img_bgr, cfg), player_boxes)
    min_len = int(cfg.min_line_len_frac * H)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                           minLineLength=min_len, maxLineGap=cfg.max_line_gap_px)
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in segs[:, 0, :]:
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if ang >= (90 - cfg.vertical_deg):
            out.append(YardLineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    return _merge_collinear(out)


def detect_hashes(img_bgr, cfg, player_boxes=None):
    """Detect hash ticks as small white components, returns their centroids.

    Small bright connected components (area + max-height bounded) that are NOT
    part of long lines. The two hash rows separate by y downstream; here we
    return all tick centroids (players masked out)."""
    import numpy as np
    mask = _zero_boxes(_white_mask(img_bgr, cfg), player_boxes)
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts: list[tuple[float, float]] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if cfg.hash_min_area <= area <= cfg.hash_max_area and h <= cfg.hash_max_h_px and w <= cfg.hash_max_h_px * 3:
            pts.append((float(cents[i][0]), float(cents[i][1])))
    return pts


def detect_field_features(img_bgr, *, cfg=None, player_boxes=None):
    from nfl_gsplat.calibration.field_features import DetectedFeatures
    cfg = cfg or FieldDetectConfig()
    H, W = img_bgr.shape[:2]
    return DetectedFeatures(
        yard_lines=detect_lines(img_bgr, cfg, player_boxes),
        sidelines=_detect_sidelines(img_bgr, cfg),
        hashes=detect_hashes(img_bgr, cfg, player_boxes),
        numbers=[],
        image_size=(W, H),
    )
```
Remove the old `_detect_hashes`/`_ocr_numbers` stubs and the `numbers` import usage. Keep `_white_mask`, `_merge_collinear`, `_detect_sidelines`.

- [ ] **Step 4: Run** `python -m pytest tests/test_field_detect.py -q` → PASS. Tune `hash_*` / mask params if the synthetic test needs it; keep the public API.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_detect.py tests/test_field_detect.py
git add nfl_gsplat/calibration/field_detect.py tests/test_field_detect.py
git commit -m "field_detect: real hash detection + player-masked lines (drop OCR seam)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 4 — Seed-at-ref + bidirectional sweep

### Task 4: `run_autocalib` hint seeding + sweep

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py`
- Test: `tests/test_run_autocalib.py` (extend)

- [ ] **Step 1: Add a sweep test to `tests/test_run_autocalib.py`** (monkeypatch detection so labels are deterministic; assert per-frame results + propagation):

```python
def test_sweep_seeds_and_propagates(monkeypatch):
    import numpy as np
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.utils.meta import CalibHint

    # 5 frames; 3 vertical lines that pan +5px/frame; constant hashes.
    def feats_for(fidx):
        base = 400 + 5 * fidx
        xs = [base, base + 400, base + 800]
        return DetectedFeatures(
            yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
            sidelines=[YardLineSeg((0, 100), (1920, 110)), YardLineSeg((0, 980), (1920, 990))],
            hashes=[(x, 540) for x in xs],
            numbers=[], image_size=(1920, 1080),
        )
    monkeypatch.setattr(ra, "_detect_for_frame", lambda img, cfg, boxes: None)  # unused via override
    hint = CalibHint(ref_frame=2, ref_x=800 + 5 * 2, yard=30, side="away", increasing="right")
    results = ra._register_sequence(
        feats_by_frame=[feats_for(i) for i in range(5)], hint=hint, image_size=(1920, 1080),
    )
    assert len(results) == 5
    assert all(r is not None for r in results)       # every frame registered + labeled
```

(If your `solve_pnp` needs ≥6 well-spread points and the 3-line synthetic gives too few, expand `feats_for` to 5 lines + both hash rows so each frame yields ≥6 correspondences; adjust the assertion accordingly. The point is: seeded at ref_frame, all frames register.)

- [ ] **Step 2: Run** → FAIL (`_register_sequence` missing).

- [ ] **Step 3: Edit `nfl_gsplat/calibration/run_autocalib.py`.** Add a pure `_register_sequence(feats_by_frame, hint, image_size)` (seed at `hint.ref_frame`, sweep fwd+back) and rewrite `build_autocalib_npz` to detect per frame (with player boxes) and call it.

```python
def _register_sequence(feats_by_frame, hint, image_size):
    """Seed identity at hint.ref_frame, propagate forward and backward, register
    each frame. Returns [CalibrationResult|None] aligned to feats_by_frame."""
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.register_frame import register_frame

    T = len(feats_by_frame)
    results = [None] * T
    ref = max(0, min(int(hint.ref_frame), T - 1))
    state0 = seed_state_from_hint(feats_by_frame[ref], hint)

    # ref frame
    corrs, state_ref = identify_correspondences(feats_by_frame[ref], state0)
    res, _ = register_frame(feats_by_frame[ref], state0, image_size)
    results[ref] = res

    prior = state_ref
    for f in range(ref + 1, T):                      # forward
        res, prior = _step(feats_by_frame[f], prior, image_size, results, f)
    prior = state_ref
    for f in range(ref - 1, -1, -1):                 # backward
        res, prior = _step(feats_by_frame[f], prior, image_size, results, f)
    return results


def _step(feats, prior, image_size, results, f):
    from nfl_gsplat.calibration.field_identify import identify_correspondences
    from nfl_gsplat.calibration.register_frame import register_frame
    _corrs, new_state = identify_correspondences(feats, prior)
    res, _ = register_frame(feats, prior, image_size)
    results[f] = res
    # carry the richer of the two states forward (new_state has this frame's lines)
    return res, (new_state if new_state.line_yardage else prior)


def build_autocalib_npz(*, play_dir, videos, fps, hints, cfg=None, masks_provider=None):
    """Detect+register every frame of each camera using its CalibHint → cameras.npz."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_field_features
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

    cfg = cfg or FieldDetectConfig()
    tracks = {}
    for cam, video in videos.items():
        if cam not in hints:
            raise SetupError(
                f"no calib_hints for camera {cam!r} in meta.yaml — add a one-line "
                "yardage hint (ref_frame/ref_x/yard/side/increasing). See SETUP.md §3."
            )
        meta = ffprobe_meta(video)
        boxes_for = masks_provider(cam) if masks_provider else (lambda f: [])
        feats_by_frame = [None] * meta.num_frames
        for fidx, frame in iter_frames(video, start_frame=0):
            if 0 <= fidx < meta.num_frames:
                feats_by_frame[fidx] = detect_field_features(
                    frame, cfg=cfg, player_boxes=boxes_for(fidx))
        results = _register_sequence(feats_by_frame, hints[cam], (meta.width, meta.height))
        tracks[cam] = assemble_track_from_results(results, width=meta.width, height=meta.height)
    return write_camera_track(Path(play_dir) / "cameras.npz", tracks, fps=fps)
```
Keep `assemble_track_from_results`, `_longest_gap_range`, the smoothing. (The test monkeypatches `_detect_for_frame` only as a guard; the real path uses `detect_field_features`. If `_register_sequence` is fully pure over `feats_by_frame`, the test calls it directly — no detection needed.)

- [ ] **Step 4: Run** `python -m pytest tests/test_run_autocalib.py -q` → PASS (existing gap tests + the sweep test). Expand the synthetic `feats_for` to yield ≥`min_landmarks` correspondences if needed.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/run_autocalib.py tests/test_run_autocalib.py
git add nfl_gsplat/calibration/run_autocalib.py tests/test_run_autocalib.py
git commit -m "run_autocalib: hint seed-at-ref + bidirectional propagation sweep

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 5 — Stage wiring + docs

### Task 5: `02_autocalibrate` hints + diag trim + docs

**Files:**
- Modify: `scripts/02_autocalibrate.py`, `scripts/diag_calib.py`
- Modify: `SETUP.md`, `INSTRUCTIONS.md`

- [ ] **Step 1: Edit `scripts/02_autocalibrate.py`** — pass `meta.calib_hints` and player masks. Replace the `build_autocalib_npz(...)` call:

```python
    out = build_autocalib_npz(
        play_dir=pd.dir, videos=videos, fps=meta.fps, hints=meta.calib_hints,
    )
```
(Masks: leave `masks_provider=None` for now — the YOLO/tracks.parquet wiring is the remaining real-footage step, finalized at bring-up; an empty mask still works, just with more line clutter. Add a `# TODO(bring-up): wire tracks.parquet player boxes` comment.) Confirm parse: `python -c "import ast; ast.parse(open('scripts/02_autocalibrate.py').read())"`.

- [ ] **Step 2: Trim `scripts/diag_calib.py`** — remove the OCR band/rotation block (numbers aren't used anymore); keep the frame dump + line-x print (that's how the user reads `ref_x`). Add a one-line print reminding the user the line x-positions are the candidates for `ref_x`. Confirm it parses.

- [ ] **Step 3: Update `SETUP.md` §3 + `INSTRUCTIONS.md`** — calibration is automatic with a one-time hint:
  1. Dump a clean frame: `python scripts/diag_calib.py --play-dir <dir> --frame <F> --out-dir ~/scratch/diag`, view it, and read the printed yard-line x-positions.
  2. Add `calib_hints` to that play's `meta.yaml` (per camera: `ref_frame`, `ref_x` = the x of a line you can identify, `yard`/`side` of that line, `increasing`).
  3. `python scripts/02_autocalibrate.py --play-dir <dir>` → `cameras.npz` (runs automatically in `04_process_play.sh`). A wrong hint fails loud with a frame range — flip `side`/`increasing` and re-run.
  Note the YOLO player-mask wiring for line de-cluttering is finalized at bring-up.

- [ ] **Step 4: Full suite + commit**

```bash
python -m pytest -m "not gpu and not slow and not real_video" -q   # all green
python -c "import ast; ast.parse(open('scripts/02_autocalibrate.py').read())"
python -m ruff check nfl_gsplat scripts tests
git add -A
git commit -m "Wire calib_hints into 02_autocalibrate; trim diag OCR; docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** hint in meta.yaml → Task 1; `assign_from_hint`/`seed_state_from_hint` + propagation-only identify (OCR removed) → Task 2; player-masked lines + real hash detection → Task 3; seed-at-ref bidirectional sweep + missing-hint SetupError → Task 4; stage wiring + diag trim + docs → Task 5. Self-validation via RMS = the existing `register_frame` None-gap + `assemble_track_from_results` fail-loud (unchanged, reused). Cross-camera consistency = shared `NFL_LANDMARKS` (inherent).
- **Type consistency:** `CalibHint(ref_frame, ref_x, yard, side, increasing)` used in Tasks 1,2,4. `seed_state_from_hint(feats, hint) -> IdentityState`, `identify_correspondences(feats, prior) -> (corrs, IdentityState)`, `detect_field_features(img, *, cfg, player_boxes)`, `detect_lines(img, cfg, player_boxes)`, `detect_hashes(img, cfg, player_boxes)`, `_register_sequence(feats_by_frame, hint, image_size)`, `build_autocalib_npz(*, play_dir, videos, fps, hints, ...)` consistent across tasks.
- **Placeholder scan:** the only deferred piece is the YOLO/tracks.parquet `masks_provider` wiring (Task 5), explicitly a bring-up TODO with a working empty-mask default — not a stub in the tested logic.

## Known follow-ups (bring-up, against real footage)
- Wire `masks_provider` to per-frame player boxes (from `tracks.parquet`, produced by the prior `detect_track` stage) to de-clutter line detection.
- Tune `FieldDetectConfig` (white threshold, hash area/height bands, min line length) on real frames via `diag_calib`.
- Confirm hash-row → left/right and the `increasing` direction against the camera (RMS gate flags a wrong hint).
