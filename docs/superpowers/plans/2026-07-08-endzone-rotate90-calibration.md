# Endzone Rotate-90 Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate endzone cameras by rotating their frames 90° into the sideline-shaped pipeline and exactly de-rotating the recovered cameras, plus behind-endzone multi-start candidates.

**Architecture:** New `nfl_gsplat/calibration/view_rotation.py` provides the pixel map (validated against cv2.rotate itself), image rotation, and exact de-rotation of `CalibrationResult`s (in-plane Rz composition; camera center provably invariant). `build_autocalib_npz_pretrained` applies per-camera rotation (name-based default: endzone → 90°, CLI override) to frames, cached keypoints, and image size; de-rotates results before assembly, so `cameras.npz` stays in original pixel coordinates. `joint_solve._candidate_centers` gains behind-endzone grid points.

**Tech Stack:** numpy, cv2 (`rotate`), existing pipeline; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-08-endzone-rotate90-calibration-design.md`

## Global Constraints

- Rotation values ∈ {0, 90, 180, 270}; default by camera name: `endzone` → 90, else 0; CLI `--rotate cam=deg` overrides (comma-separated for multiple).
- `cameras.npz` must be in TRUE original-pixel coordinates (de-rotation before assembly). Camera center is rotation-invariant — use that as a sanity check.
- Pixel-map convention must match `cv2.rotate` exactly (pixel indices, not continuous coords); the known half-pixel principal-point offset from index-vs-center conventions bounds de-rotation projection error at ≤ 1.0 px — tests use that tolerance, with a comment.
- Fail loud: unknown `--rotate` value → `typer.BadParameter`; per-camera CalibrationError from the pretrained build must NAME the camera.
- No frame buffering; no new GPU/torch deps; `pytest -m "not gpu and not slow"` green; `python -m ruff check` clean on touched files.
- NEVER commit real NFL imagery; test fixtures synthetic numpy only.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `view_rotation.py` — pixel map, image rotation, exact de-rotation

**Files:**
- Create: `nfl_gsplat/calibration/view_rotation.py`
- Test: `tests/test_view_rotation.py` (new)

**Interfaces:**
- Consumes: `CalibrationResult` (`solve_pnp.py`: `intrinsics, pose, rms_px, num_correspondences, refined_with_ba`), `CameraIntrinsics`, `CameraPose`, `project_points` (`utils/geometry.py`).
- Produces (Task 3 relies on these exact signatures):
  - `rotate_image(bgr: np.ndarray, deg: int) -> np.ndarray` — cv2.rotate wrapper, 0 = passthrough.
  - `rotate_uv(u: float, v: float, deg: int, orig_wh: tuple[int, int]) -> tuple[float, float]` — where the pixel at (u, v) of the ORIGINAL (W, H) image lands in the rotated image.
  - `rotated_wh(deg: int, orig_wh) -> tuple[int, int]` — (W, H) after rotation.
  - `derotate_result(result, deg: int, orig_wh) -> CalibrationResult` — solution computed in rotated-image coordinates → original-pixel coordinates.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_view_rotation.py
"""View-rotation utilities. The pixel map is validated against cv2.rotate
itself (single marked pixel), not against a hand-derived formula."""
import cv2
import numpy as np
import pytest

from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_image, rotate_uv, rotated_wh,
)
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

W, H = 32, 20                       # small asymmetric test image


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_rotate_uv_matches_cv2_rotate(deg):
    # mark one pixel, rotate the image with cv2, find where it went, and
    # demand rotate_uv predicts exactly that location
    img = np.zeros((H, W), np.uint8)
    u, v = 5, 3
    img[v, u] = 255
    rot = rotate_image(img, deg)
    ys, xs = np.nonzero(rot)
    assert len(xs) == 1
    pu, pv = rotate_uv(float(u), float(v), deg, (W, H))
    assert (round(pu), round(pv)) == (xs[0], ys[0])
    assert rot.shape[::-1] == rotated_wh(deg, (W, H))


def test_rotate_uv_round_trip_90_270():
    u2, v2 = rotate_uv(5.0, 3.0, 90, (W, H))
    u3, v3 = rotate_uv(u2, v2, 270, rotated_wh(90, (W, H)))
    assert (u3, v3) == (5.0, 3.0)


def test_rotate_image_invalid_deg_raises():
    with pytest.raises(ValueError, match="rotation"):
        rotate_image(np.zeros((4, 4), np.uint8), 45)


def _camera_for_rotated(deg, f=1400.0):
    """A camera solved IN the rotated frame of a (1920,1080) original, looking
    at the field plane (reuses the joint-solve test geometry)."""
    ow, oh = 1920, 1080
    rw, rh = rotated_wh(deg, (ow, oh))
    C = np.array([-90.0, 0.0, 35.0])
    target = np.array([-30.0, 0.0, 0.0])
    fwd = target - C; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    return CalibrationResult(
        intrinsics=CameraIntrinsics(f, f, rw / 2, rh / 2, rw, rh),
        pose=CameraPose(R=R, t=-R @ C), rms_px=0.5,
        num_correspondences=8, refined_with_ba=True), (ow, oh)


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_derotate_result_projection_equivalence(deg):
    # For any world point: projecting through the DE-ROTATED camera onto the
    # original image must equal mapping the rotated-camera projection back
    # through the pixel map. Tolerance 1.0 px: cv2.rotate maps pixel INDICES
    # (0..N-1) while intrinsics use the W/2 center convention — a constant
    # half-pixel offset, not an accumulating error.
    res_rot, orig_wh = _camera_for_rotated(deg)
    res = derotate_result(res_rot, deg, orig_wh)
    pts = np.array([[-30.0, 5.0, 0.0], [-40.0, -10.0, 0.0], [-20.0, 0.0, 0.0]])
    uv_rot = project_points(pts, res_rot.intrinsics.K(), res_rot.pose.R, res_rot.pose.t)
    uv_orig = project_points(pts, res.intrinsics.K(), res.pose.R, res.pose.t)
    inv = {90: 270, 180: 180, 270: 90}[deg]
    rw_h = rotated_wh(deg, orig_wh)
    mapped = np.array([rotate_uv(u, v, inv, rw_h) for (u, v) in uv_rot])
    assert np.abs(mapped - uv_orig).max() <= 1.0
    # camera center is rotation-invariant
    assert np.allclose(res.pose.center_world(), res_rot.pose.center_world(), atol=1e-9)
    # intrinsics rebuilt for the original dims
    assert (res.intrinsics.width, res.intrinsics.height) == orig_wh
    assert res.intrinsics.fx == res_rot.intrinsics.fx


def test_derotate_zero_is_identity():
    res_rot, orig_wh = _camera_for_rotated(0)
    res = derotate_result(res_rot, 0, orig_wh)
    assert np.allclose(res.pose.R, res_rot.pose.R)
    assert np.allclose(res.pose.t, res_rot.pose.t)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_view_rotation.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.view_rotation`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/view_rotation.py
"""Per-camera view rotation: run endzone footage through the sideline-shaped
pipeline by rotating frames 90 deg (yard lines become vertical, hash columns
become rows), then compose the known in-plane rotation back into the
recovered camera. Exact: fx = fy, so an in-plane rotation absorbs entirely
into R (camera center provably invariant: C = -R^T t is unchanged by
R -> Rz R, t -> Rz t)."""
from __future__ import annotations

import numpy as np

_VALID = (0, 90, 180, 270)


def _check(deg: int) -> None:
    if deg not in _VALID:
        raise ValueError(f"rotation must be one of {_VALID}, got {deg!r}")


def rotate_image(bgr: np.ndarray, deg: int) -> np.ndarray:
    _check(deg)
    if deg == 0:
        return bgr
    import cv2
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]
    return cv2.rotate(bgr, code)


def rotated_wh(deg: int, orig_wh) -> tuple[int, int]:
    _check(deg)
    w, h = orig_wh
    return (w, h) if deg in (0, 180) else (h, w)


def rotate_uv(u: float, v: float, deg: int, orig_wh) -> tuple[float, float]:
    """Where the pixel at (u, v) of the (W, H) original lands after
    rotate_image(deg). Matches cv2.rotate's pixel-index convention (tested
    against cv2 directly, not derived on faith)."""
    _check(deg)
    w, h = orig_wh
    if deg == 0:
        return (u, v)
    if deg == 90:                       # ROTATE_90_CLOCKWISE
        return (h - 1 - v, u)
    if deg == 180:
        return (w - 1 - u, h - 1 - v)
    return (v, w - 1 - u)               # 270 = counterclockwise


def _rz(deg: int) -> np.ndarray:
    """In-plane camera rotation composing the INVERSE pixel rotation, i.e.
    R_orig = _rz(deg) @ R_rotated. Sign fixed by the projection-equivalence
    test (test_derotate_result_projection_equivalence)."""
    th = {90: -np.pi / 2, 180: np.pi, 270: np.pi / 2}[deg]
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def derotate_result(result, deg: int, orig_wh):
    """CalibrationResult solved in rotated-image coordinates -> original
    pixel coordinates (same focal, original width/height, R and t composed
    with the in-plane rotation; camera center unchanged)."""
    _check(deg)
    if deg == 0:
        return result
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    rz = _rz(deg)
    R = rz @ result.pose.R
    t = rz @ result.pose.t
    w, h = orig_wh
    intr = CameraIntrinsics(result.intrinsics.fx, result.intrinsics.fy,
                            w / 2.0, h / 2.0, w, h)
    return CalibrationResult(intrinsics=intr, pose=CameraPose(R=R, t=t),
                             rms_px=result.rms_px,
                             num_correspondences=result.num_correspondences,
                             refined_with_ba=result.refined_with_ba)
```

NOTE on `_rz` signs: the derivation says 90° clockwise image rotation composes
as `Rz(−90°)` — but the projection-equivalence test is the authority. If the
test fails with the mapped/actual points related by a 180° or sign flip, swap
the 90/270 angles (or negate `s`) until the test passes for ALL THREE degs;
do not loosen the tolerance.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_view_rotation.py -v`
Expected: all PASS (fix `_rz` signs per the note if the equivalence test fails).

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/view_rotation.py tests/test_view_rotation.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/view_rotation.py tests/test_view_rotation.py
git commit -m "feat(calibration): view-rotation pixel map and exact camera de-rotation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Behind-endzone multi-start candidates

**Files:**
- Modify: `nfl_gsplat/calibration/joint_solve.py` (grid constants + `_candidate_centers`)
- Test: `tests/test_joint_solve.py` (append)

**Interfaces:**
- Consumes/Produces: `_candidate_centers(init_results) -> list[np.ndarray]` (existing; list grows).

- [ ] **Step 1: Write the failing test** (append to tests/test_joint_solve.py)

```python
def test_candidate_centers_include_endzone_positions():
    # endzone cameras sit behind an endzone (|X| 60-120, Y ~ 0) — the
    # sideline-only grid physically could not reach them (2026-07-07 failure)
    from nfl_gsplat.calibration.joint_solve import _candidate_centers
    cands = np.stack(_candidate_centers([None] * 5))
    endzoneish = (np.abs(cands[:, 0]) >= 60) & (np.abs(cands[:, 1]) <= 20)
    assert endzoneish.sum() >= 18            # both endzones x several Z/Y
    sideline = (np.abs(cands[:, 0]) <= 30) & (np.abs(cands[:, 1]) >= 45)
    assert sideline.sum() >= 54              # original grid retained
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_solve.py::test_candidate_centers_include_endzone_positions -v`
Expected: FAIL (endzoneish count 0).

- [ ] **Step 3: Implement** — in `joint_solve.py`, next to the existing grid constants:

```python
_GRID_EZ_X = (-120.0, -90.0, -60.0, 60.0, 90.0, 120.0)
_GRID_EZ_Y = (-20.0, 0.0, 20.0)
_GRID_EZ_Z = (15.0, 35.0, 65.0)
```

and in `_candidate_centers`, after the existing sideline grid loop:

```python
    for X in _GRID_EZ_X:                     # behind-endzone cameras
        for Y in _GRID_EZ_Y:
            for Z in _GRID_EZ_Z:
                cands.append(np.array([X, Y, Z]))
```

- [ ] **Step 4: Run the joint-solve test file**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: all PASS. Note the runtime — the candidate list roughly doubles
(54 → 108) and `_resolve_reflection` scores labelings × candidates with
short solves. The synthetic tests are protected by the anchor-first
early-accept path; if the file's runtime regresses past ~90 s, verify the
anchor candidate is still first in the list (it must remain `cands[0]`).

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/joint_solve.py tests/test_joint_solve.py
git commit -m "feat(calibration): behind-endzone multi-start camera candidates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Rotation seam in the pretrained build + CLI + camera-named errors

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (`build_autocalib_npz_pretrained` signature + body)
- Modify: `scripts/02_autocalibrate.py` (`--rotate` option, rotation map parsing)
- Test: `tests/test_run_autocalib.py` (append)

**Interfaces:**
- Consumes: Task 1's `rotate_image`, `rotate_uv`, `rotated_wh`, `derotate_result` (import at module level in run_autocalib.py so tests can monkeypatch `ra.rotate_image` etc.).
- Produces: `build_autocalib_npz_pretrained(*, play_dir, videos, fps, kps_json, territory, cfg=None, masks_provider=None, rotations: dict[str, int] | None = None)` — `rotations` maps camera → deg; missing cameras default by name (`endzone` → 90, else 0). CLI: `--rotate "endzone=90"` (comma-separated pairs) overrides.

- [ ] **Step 1: Write the failing tests** (append to tests/test_run_autocalib.py)

```python
def _pretrained_build_harness(monkeypatch, rotations=None, cam="endzone"):
    """Run build_autocalib_npz_pretrained with one fake camera and capture
    what reaches detection/fusion/joint-solve. Follows the monkeypatch
    pattern of test_build_pretrained_uses_joint_solve."""
    from nfl_gsplat.calibration import run_autocalib as ra

    captured = {}

    class _Meta:
        num_frames, width, height = 2, 1920, 1080

    frames = [np.full((1080, 1920, 3), i, np.uint8) for i in range(2)]
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter(enumerate(frames)))
    monkeypatch.setattr(
        "nfl_gsplat.calibration.roboflow_kps.load_kps_json",
        lambda p, expect_num_frames=None: {0: [("30", 10.0, 20.0, 0.9)],
                                           1: [("30", 11.0, 21.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured.setdefault("frame_shapes", []).append(frame.shape)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[],
                                numbers=[], image_size=frame.shape[:2][::-1])
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)

    def fake_fuse(yard_lines, hashes, model_kps, *, territory, image_size, **kw):
        captured.setdefault("fuse_kps", []).append(model_kps)
        captured["fuse_image_size"] = image_size
        return [(f"c{i}", (float(i), 0.0)) for i in range(8)]
    monkeypatch.setattr(ra, "fuse_frame", fake_fuse)

    monkeypatch.setattr(ra, "_solve_sweep",
                        lambda *a, **k: [None, None], raising=False)

    def fake_joint(corrs_by_frame, image_size, *, init_results, **kw):
        captured["joint_image_size"] = image_size
        return ["R0", "R1"], False
    monkeypatch.setattr(ra, "solve_fixed_center", fake_joint)

    def fake_derotate(result, deg, orig_wh):
        captured.setdefault("derotated", []).append((result, deg, orig_wh))
        return result
    monkeypatch.setattr(ra, "derotate_result", fake_derotate)

    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: ("TRACK", width, height))
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={cam: "v.mp4"}, fps=30.0,
        kps_json={cam: "k.json"}, territory="away", rotations=rotations)
    return captured


def test_pretrained_endzone_rotates_by_default(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations=None, cam="endzone")
    # frames reach detection rotated 90 deg: (1080,1920,3) -> (1920,1080,3)
    assert all(s == (1920, 1080, 3) for s in cap["frame_shapes"])
    assert cap["fuse_image_size"] == (1080, 1920)      # rotated (W,H)
    assert cap["joint_image_size"] == (1080, 1920)
    # cached kps rotated: (10,20) in 1920x1080 under 90CW -> (1080-1-20, 10)
    assert cap["fuse_kps"][0][0][1:3] == (1059.0, 10.0)
    # every joint result de-rotated back to the original dims
    assert [d for (_r, d, _wh) in cap["derotated"]] == [90, 90]
    assert all(wh == (1920, 1080) for (_r, _d, wh) in cap["derotated"])


def test_pretrained_sideline_not_rotated(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations=None, cam="sideline")
    assert all(s == (1080, 1920, 3) for s in cap["frame_shapes"])
    assert cap["fuse_image_size"] == (1920, 1080)
    assert cap["derotated"] == [] or all(d == 0 for (_r, d, _wh) in cap["derotated"])


def test_pretrained_rotations_override(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations={"endzone": 0}, cam="endzone")
    assert all(s == (1080, 1920, 3) for s in cap["frame_shapes"])


def test_pretrained_error_names_camera(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.errors import CalibrationError

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([]))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {})
    with pytest.raises(CalibrationError, match="endzone"):
        ra.build_autocalib_npz_pretrained(
            play_dir=".", videos={"endzone": "e.mp4"}, fps=30.0,
            kps_json={"endzone": "e.json"}, territory="away")
```

(Adapt the harness to how run_autocalib actually imports the patched names —
`detect_field_features`, `fuse_frame`, `solve_fixed_center`,
`assemble_track_from_results`, `write_camera_track` are already module-level
imports; keep patching them on `ra`. `derotate_result` must ALSO become a
module-level import for the fake to bind. `iter_frames` yields RGB frames in
production; the harness's uint8 arrays stand in fine.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_autocalib.py -v -k "rotat or names_camera"`
Expected: FAIL — `build_autocalib_npz_pretrained` has no `rotations` kwarg / no `derotate_result` attribute.

- [ ] **Step 3: Implement in run_autocalib.py**

Module-level import:

```python
from nfl_gsplat.calibration.view_rotation import (
    derotate_result, rotate_image, rotate_uv, rotated_wh,
)
```

Default-resolution helper (module level, above the build function):

```python
def _camera_rotation(cam: str, rotations) -> int:
    """View rotation for a camera: explicit map wins; else endzone -> 90
    (broadcast endzone views need the sideline-shaped pipeline rotated),
    everything else 0."""
    if rotations and cam in rotations:
        return int(rotations[cam])
    return 90 if cam == "endzone" else 0
```

In `build_autocalib_npz_pretrained(..., rotations=None)`, per camera:

```python
        deg = _camera_rotation(cam, rotations)
        orig_wh = (meta.width, meta.height)
        work_w, work_h = rotated_wh(deg, orig_wh)
```

- rotate each decoded frame before detection: `fr = rotate_image(fr, deg)`;
- rotate each cached keypoint once when loading:
  `kps_by_frame = {i: [(n, *rotate_uv(u, v, deg, orig_wh), c) for (n, u, v, c) in kps] for i, kps in kps_by_frame.items()}` (skip when `deg == 0`);
- use `(work_w, work_h)` everywhere the camera loop currently uses
  `(meta.width, meta.height)` for detection/fusion/sweep/joint image size
  (including `fit_hash_rows(..., image_width=work_w)`);
- after the joint solve: `joint = [derotate_result(r, deg, orig_wh) if r is not None else None for r in joint]`;
- assembly keeps the ORIGINAL dims: `assemble_track_from_results(joint, width=meta.width, height=meta.height, max_gap=30)`.
- Wrap the per-camera body so failures name the camera:

```python
        try:
            ...existing per-camera pipeline...
        except CalibrationError as e:
            raise CalibrationError(f"camera {cam!r}: {e}") from e
```

(Re-raise, never swallow — the run still dies loud, now attributably.)

- [ ] **Step 4: Wire the CLI** (`scripts/02_autocalibrate.py`)

```python
rotate: str = typer.Option("", "--rotate",
    help="Per-camera view rotation override, e.g. 'endzone=90' or "
         "'endzone=90,sideline=0'. Default: endzone->90, others->0."),
```

parse before the pretrained branch:

```python
    rotations = {}
    for pair in (p.strip() for p in rotate.split(",") if p.strip()):
        cam_name, _, deg_s = pair.partition("=")
        if deg_s not in ("0", "90", "180", "270"):
            raise typer.BadParameter(
                f"--rotate {pair!r}: rotation must be 0/90/180/270.")
        rotations[cam_name.strip()] = int(deg_s)
```

and pass `rotations=rotations or None` into `build_autocalib_npz_pretrained`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_run_autocalib.py tests/test_view_rotation.py -v`
Expected: all PASS (existing wiring tests must stay green — the sideline
path is deg=0 passthrough).
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.
Run: `python -m ruff check nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py
git commit -m "feat(calibration): per-camera view rotation seam in pretrained mode

Endzone frames + cached keypoints rotate 90 deg into the sideline-shaped
pipeline; recovered cameras are exactly de-rotated before assembly, so
cameras.npz stays in original pixel coordinates. Per-camera failures now
name the camera.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Rotated synthetic ground truth + real acceptance

**Files:**
- Test: `tests/test_joint_solve.py` (append)

**Interfaces:**
- Consumes: `_synthetic_frames`, `_init_results_from_truth`, `C_TRUE`, `W`, `H` (existing test helpers); Task 1's `rotate_uv`, `rotated_wh`, `derotate_result`; `solve_fixed_center`.

- [ ] **Step 1: Write the failing test**

```python
def test_rotated_view_solves_to_same_camera():
    # Simulate an endzone-style rotated view: rotate every synthetic uv
    # observation 90 deg (as the pipeline does to endzone frames), solve in
    # rotated coordinates, de-rotate — the recovered camera must match the
    # unrotated ground truth. End-to-end check of rotate_uv + solve +
    # derotate_result composing correctly.
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.calibration.view_rotation import (
        derotate_result, rotate_uv, rotated_wh,
    )
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30, noise_px=0.3)
    rw_h = rotated_wh(90, (W, H))
    fd_rot = {}
    for i in ids:
        world, uv = fd[i]
        uv_r = np.array([rotate_uv(u, v, 90, (W, H)) for (u, v) in uv])
        fd_rot[i] = (world, uv_r)
    results, mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=rw_h,
        init_results=[None] * 30,          # no anchors: grid multi-start only
        _frame_data_override=fd_rot)
    solved = [(i, r) for i, r in enumerate(results) if r is not None]
    assert len(solved) >= 25
    deros = [derotate_result(r, 90, (W, H)) for _i, r in solved]
    C_rec = deros[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5
    for (i, _r), d in zip(solved, deros):
        assert d.intrinsics.fx == pytest.approx(f_true[i], rel=0.02)
```

NOTE: with `init_results=[None]*30` there is no anchor candidate, so the
solve must find `C_TRUE = (-19, 1, 95)` from the grid. Neither the sideline
nor the endzone grid contains a point near (−19, 1, 95) — if the multi-start
cannot converge from the nearest grid point within the test's runtime, USE
the anchor path instead: pass
`init_results=_init_results_from_truth(ids, f_true, R_true, 30, jitter=1.0)`
(anchors are rotation-agnostic — they carry C, and per-frame pose init is
recomputed by look-at inside the solver). Prefer the anchor variant if the
no-anchor variant is flaky or slow; the point under test is the rotation
composition, not the multi-start.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_solve.py::test_rotated_view_solves_to_same_camera -v`
Expected: FAIL before Tasks 1-3 are in place (import error) or, after them,
potential geometry failure if `_rz` signs are wrong — this test is the
end-to-end guard.

- [ ] **Step 3: Make it pass, full suite, commit**

Run: `python -m pytest tests/test_joint_solve.py -v` then the full suite — all pass.

```bash
git add tests/test_joint_solve.py
git commit -m "test(calibration): rotated-view synthetic solves to the unrotated camera

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Real-footage acceptance (controller/human gate; needs no API key)**

```
python scripts/02_autocalibrate.py --play-dir data\2025\week_04\SEA_at_AZ\play_001 --mode pretrained --territory away
```
Expected outcomes, in order of preference:
1. `cameras.npz` written with `sideline_*` AND `endzone_*` arrays; endzone
   center behind an endzone (|X| ∈ 50–150, |Y| ≤ 40, Z ∈ 10–80); record the
   endzone kept-frame count; spot-check an endzone grid overlay (project the
   field through the de-rotated endzone K,R,t onto an ORIGINAL endzone frame).
2. Fail-loud naming camera 'endzone' at a joint-solve gate — record which
   gate and the counts; that measured result scopes the next cycle
   (endzone-specific identity anchors) and the branch still merges on the
   strength of Tasks 1-3's tests.
Sideline results must be unchanged either way (center ≈ (−3.6, 80.5, 35.9),
~816 kept frames).
