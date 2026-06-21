# Per-Frame Camera Calibration (Anchored Homography Tracking)

**Date:** 2026-06-21
**Status:** Approved (design); implementation plan pending

## Problem

The pipeline calibrates **one static `(K, R, t)` per camera per play** from a single
frame (`solve_pnp_from_annotations` → one `cameras.json`). Every 3D consumer —
two-view joint triangulation, cross-camera re-ID ground projection, ball 3D
triangulation, and field reconstruction — treats those parameters as fixed for the
whole play.

The available footage is **All-22, but not locked**: the cameras pan, tilt, and
zoom to follow the play. A calibration solved on one frame is therefore wrong for
every other frame (focal changes with zoom; orientation changes with pan/tilt), so
triangulation and re-ID would silently produce wrong 3D. We need **per-frame camera
parameters**.

## Decisions (locked)

1. **Method: keyframe anchors + homography tracking.** Hand-annotate landmarks on a
   few keyframes (PnP-solved anchors). Between anchors, track the field ground-plane
   homography frame-to-frame; decompose each frame's homography into `(K_t,R_t,t_t)`.
2. **Failure posture: fail loud.** If tracking confidence drops over a frame range,
   stop and tell the user to add a keyframe in that gap and re-run. Never emit
   silently-wrong per-frame calibration.
3. Internal CV defaults: reuse the existing YOLO person detector to mask players out
   of field-feature tracking; decompose homographies with the fixed-principal-point,
   unit-aspect, solve-focal intrinsic model the PnP solver already uses; track
   bidirectionally between consecutive anchors and blend to bound drift.

## Core idea

The field is planar (world `z = 0`). Each frame's view of the field is a homography
`H_t` mapping field-plane points to image pixels. We:

1. PnP-calibrate a few **keyframe anchors** → full `(K, R, t)` and, equivalently, the
   field→image homography at each anchor (for a `z=0` plane,
   `H = K · [r1 | r2 | t]`, the first two rotation columns + translation).
2. **Track `H_t`** for every frame between consecutive anchors via optical flow on
   static field features (with players masked out), RANSAC-fitting the inter-frame
   homography and composing onto the anchor homography. Track **forward** from the
   left anchor and **backward** from the right anchor; blend the two by distance so
   drift is bounded and the result snaps to both anchors.
3. **Decompose each `H_t`** back into `(K_t, R_t, t_t)` using fixed principal point +
   unit aspect, solving the focal — the same intrinsic model as `solve_pnp`.

Because every frame's camera is expressed in the **same field world coordinate
system** (defined by `NFL_LANDMARKS`), the two cameras remain mutually consistent
frame-by-frame, so per-frame two-view triangulation is valid.

## Data representation

- **`{play}/cameras.npz`** (replaces the single-pose `cameras.json`): per camera,
  per frame —
  - `{cam}_K` `[T, 3, 3]`, `{cam}_R` `[T, 3, 3]`, `{cam}_t` `[T, 3]`
  - `{cam}_conf` `[T]` (tracking confidence; anchors = 1.0)
  - scalars `width`, `height`, `fps`, and a `cams` string array listing camera names.
- **`{play}/{cam}_keyframes.json`**: `[{ "frame": int, "landmarks": [{"name","uv"}] }]`
  — the annotated anchors, so re-runs of the batch tracker never re-annotate.

`T` equals the clip's frame count (`ffprobe_meta(video).num_frames`); both cameras
share the same `T` (synced clips).

## Components (isolated, single-responsibility)

### `nfl_gsplat/calibration/annotate_gui.py` (extend)
Add a `frame_index` to the saved entries (already present) and let the calibration
script call `annotate(video, out, frame_index=k)` for **multiple** keyframes,
accumulating into `{cam}_keyframes.json` rather than a single landmarks file.

### `nfl_gsplat/calibration/solve_pnp.py` (reuse)
Unchanged. Each keyframe's landmarks → `CalibrationResult` (anchor `K,R,t`). A thin
helper converts an anchor `(K,R,t)` to its field→image homography
`H = K · [r1 | r2 | t]`.

### `nfl_gsplat/calibration/decompose_homography.py` (new, pure)
`homography_to_krt(H, width, height) -> (K, R, t)`:
- Fix `cx=width/2`, `cy=height/2`, unit aspect; solve focal `f` from the homography
  (the two field-plane axis columns are orthonormal under the true `K`, giving a
  constraint on `f`).
- Recover `[r1 | r2 | t] = K⁻¹ H`, normalize to a proper rotation
  (`r3 = r1 × r2`; re-orthonormalize via SVD), scale `t` consistently.
- `krt_to_homography(K, R, t) -> H` for round-trip tests.
CPU-only, fully unit-testable.

### `nfl_gsplat/calibration/track_homography.py` (new core)
`track_camera_sequence(video, anchors, masks_provider, *, cfg) -> CameraTrack`:
- `anchors`: `{frame: (K,R,t)}` from the keyframes.
- For each consecutive anchor pair `[a, b]`: select Good-Features-To-Track on the
  field region of frame `a` (excluding player-mask boxes and the non-field
  border/crowd), KLT-track to the next frame, RANSAC-fit the inter-frame homography,
  compose onto the running field→image homography. Track forward from `a` and
  backward from `b`; blend by normalized distance `w = (t-a)/(b-a)`.
- Per-frame **confidence** = f(RANSAC inlier ratio, feature count, residual). If a
  contiguous run falls below `cfg.min_conf`, raise `CalibrationError` naming the
  frame range and instructing the user to add a keyframe there.
- Decompose each blended `H_t` via `decompose_homography` → fill `CameraTrack`.
- The optical-flow / GoodFeatures / RANSAC calls are the OpenCV seam; the
  decomposition, blending, confidence gating, and gap detection are pure and tested
  with synthetic homography sequences.

### Player masking
A `masks_provider(frame_idx) -> list[bbox]` built from the existing YOLO person
detector (or, when available, the play's `tracks.parquet` boxes). Field features are
sampled only outside these boxes. This makes the batch tracker depend on detection
having run first (see Pipeline).

### `nfl_gsplat/calibration/cameras_io.py` (change)
- `CameraTrack`: holds the per-frame arrays + `.at(frame) -> (CameraIntrinsics, CameraPose)`.
- `load_camera_track(cameras_npz) -> {cam: CameraTrack}`; raises `SetupError` if
  missing. A `write_camera_track(path, tracks)` writer.
- Back-compat shim: a single-pose `cameras.npz` (all frames identical) is valid, so
  Phase-1 threading can land before tracking exists.

### Stage scripts
- **`scripts/02_calibrate_cameras.py`** (extend, interactive): `--play-dir` +
  repeatable `--keyframe FRAME` (or an interactive "pick frames" loop); for each
  camera and keyframe, annotate + solve, write `{cam}_keyframes.json`. Needs a
  display (OnDemand desktop). Does **not** produce `cameras.npz`.
- **`scripts/02b_track_calibration.py`** (new, batch, headless): reads keyframes +
  video + masks, runs `track_camera_sequence` per camera, writes `cameras.npz`.
  Fail-loud on low-confidence gaps.

## Pipeline & workflow

```
(interactive, display, once)  02_calibrate_cameras.py --annotate  → {cam}_keyframes.json
(batch, headless, GPU/CPU node) detection → 02b_track_calibration → cameras.npz
                                            → tracking/reID → pose triangulation → ball → field → render
```

`04_process_play.sh` gains a calibration-tracking step **after** detection (so player
masks exist) and **before** cross-cam re-ID. `02_calibrate_cameras.py` remains a
manual pre-step (the only display-gated part).

## Consumer threading

Each switches from one `(K,R,t)` to `track.at(t)`:
- `tracking/cross_cam_reid.py` — `project_to_plane_z0` per frame.
- `pose/triangulate.py` + `pose/run_pose.py` `extract_observations` — per-frame
  projection matrices for two-view triangulation.
- `ball/kalman_3d.py` — per-frame triangulation of the ball.
- `field/build_transforms.py` — emit one `transforms.json` entry per frame using that
  frame's pose (real parallax from camera motion improves the nerfstudio field).
- `compositing` / render — **unchanged** (uses a virtual-camera trajectory, not the
  broadcast cameras).

## Failure handling

- `< min_landmarks` at a keyframe → existing `CalibrationError` (annotate more).
- Low-confidence tracking run → `CalibrationError` with the frame range + "add a
  keyframe between X–Y and re-run `02b`".
- Missing `cameras.npz` at a consumer → `SetupError` pointing at the calibration
  steps.

## Testing (CPU)

- `decompose_homography`: `krt_to_homography → homography_to_krt` round-trip recovers
  `(K,R,t)` within tolerance for a range of focal/pan/tilt.
- `track_homography` core: synthetic ground-truth trajectory `(K_t,R_t,t_t)` over a
  known field-point cloud → project to per-frame pixels → feed a stub
  flow/homography estimator → assert recovered params ≈ truth; assert blending snaps
  to both anchors; assert an injected confidence gap raises with the right range.
- `cameras_io`: `write_camera_track`/`load_camera_track` round-trip; `.at(t)` returns
  the right slice; single-pose shim works.
- Consumer threading: per-frame projection used (re-ID/triangulation pick the
  frame-`t` matrix), verified on a synthetic 2-frame moving-camera fixture.
- Keep `pytest -m "not gpu and not slow"` green; ruff clean.

## Phasing (for the implementation plan)

1. **Data rep + threading**: `decompose_homography`, `cameras_io` `CameraTrack` +
   per-frame I/O with single-pose shim, thread all consumers + `build_transforms`,
   tests. (Pipeline still runs on a constant track.)
2. **Multi-keyframe annotation**: extend `annotate_gui` + `02_calibrate_cameras.py`,
   write `{cam}_keyframes.json`.
3. **Tracking core**: `track_homography.py` (flow/RANSAC + bidirectional blend +
   confidence/fail-loud) + masking, synthetic tests.
4. **Wire batch stage**: `02b_track_calibration.py` + `04_process_play.sh`
   integration.

## Out of scope

- Automatic per-frame field-line detection/registration (the deferred "hybrid"
  retry path).
- Lens distortion (assumed negligible for broadcast tele lenses, consistent with the
  current solver).
- Rolling-shutter correction.
- Re-rendering/altering the virtual-camera render path.
