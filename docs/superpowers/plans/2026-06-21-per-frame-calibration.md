# Per-Frame Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single static `(K,R,t)`-per-play calibration with per-frame camera parameters obtained by tracking the field ground-plane homography between hand-annotated keyframe anchors and decomposing each frame's homography into `(K_t,R_t,t_t)`.

**Architecture:** The field is planar (world z=0), so each frame's view is a homography. PnP-calibrate a few keyframe anchors; track the homography frame-to-frame (players masked) between anchors with a bidirectional blend; decompose each frame's homography to `(K,R,t)` in the shared field world frame. Store per-frame in `cameras.npz`; all 3D consumers read `track.at(frame)`. Fail loud when tracking loses confident lock.

**Tech Stack:** Python 3.10, numpy, scipy, OpenCV (`cv2`: calcOpticalFlowPyrLK, goodFeaturesToTrack, findHomography), typer, pytest. CPU-testable cores with OpenCV/video as isolated seams. Follow existing `nfl_gsplat` patterns (`SetupError`/`CalibrationError`, `utils.io`, `utils.geometry`).

**Reference spec:** `docs/superpowers/specs/2026-06-21-per-frame-calibration-design.md`

## Conventions

- Repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`. Run `python -m pytest …` locally (`conda run -n nfl_smplx …` on PACE).
- World/camera conventions are in `nfl_gsplat/utils/geometry.py`: `CameraIntrinsics(fx,fy,cx,cy,width,height)` with `.K()`; `CameraPose(R,t)` is world→camera (`x_cam = R@x_world + t`).
- A field→image homography for the z=0 plane is `H = K @ [r1 | r2 | t]` where `r1,r2` are the first two columns of `R`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Phase 1 — Homography math + per-frame data layer + threading

### Task 1: `decompose_homography.py` (pure)

**Files:**
- Create: `nfl_gsplat/calibration/decompose_homography.py`
- Test: `tests/test_decompose_homography.py`

- [ ] **Step 1: Write the failing test** — `tests/test_decompose_homography.py`:

```python
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    homography_to_krt, krt_to_homography,
)
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def _krt(fx, yaw_deg, pitch_deg, cam_height, W=1920, H=1080):
    # A plausible broadcast cam looking down at the z=0 field.
    intr = CameraIntrinsics(fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H)
    ry, rx = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Rz = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    # camera looks toward -Z world after a downward pitch; world->cam:
    R = Rx @ Rz @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    cam_center = np.array([0.0, -40.0, cam_height])
    t = -R @ cam_center
    return intr, CameraPose(R=R, t=t)


def test_krt_homography_roundtrip_recovers_params():
    intr, pose = _krt(fx=2600.0, yaw_deg=8.0, pitch_deg=22.0, cam_height=18.0)
    H = krt_to_homography(intr.K(), pose.R, pose.t)
    K2, R2, t2 = homography_to_krt(H, width=intr.width, height=intr.height)
    # Focal within 1%, and the recovered camera reprojects field points to the
    # same pixels (homography is scale-ambiguous, so compare projections).
    assert abs(K2[0, 0] - intr.fx) / intr.fx < 0.01
    field_pts = np.array([[0, 0, 0], [20, 10, 0], [-30, -15, 0], [45, 20, 0]], float)
    uv_ref = project_points(field_pts, intr.K(), pose.R, pose.t)
    uv_dec = project_points(field_pts, K2, R2, t2)
    assert np.allclose(uv_ref, uv_dec, atol=1.0)


def test_homography_to_krt_returns_proper_rotation():
    intr, pose = _krt(fx=3000.0, yaw_deg=-5.0, pitch_deg=30.0, cam_height=20.0)
    H = krt_to_homography(intr.K(), pose.R, pose.t)
    _, R2, _ = homography_to_krt(H, width=intr.width, height=intr.height)
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R2) - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_decompose_homography.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement** `nfl_gsplat/calibration/decompose_homography.py`:

```python
"""Homography <-> (K, R, t) for the planar (z=0) field.

A camera viewing the world z=0 plane satisfies, for a world point (X, Y, 0):
    s [u, v, 1]^T = K [r1 | r2 | t] [X, Y, 1]^T
so the field->image homography is H = K [r1 | r2 | t] (r1, r2 = first two
columns of R). Given H and the fixed-principal-point/unit-aspect intrinsic
model (same as solve_pnp), we recover the focal from the orthonormality of the
plane axes and rebuild a proper (K, R, t). Pure numpy; CPU-only.
"""
from __future__ import annotations

import numpy as np


def krt_to_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Field(z=0)->image homography H = K [r1 | r2 | t]."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    M = np.column_stack([R[:, 0], R[:, 1], t])     # 3x3 = [r1 | r2 | t]
    H = np.asarray(K, dtype=np.float64) @ M
    return H / H[2, 2]


def _solve_focal(H: np.ndarray, cx: float, cy: float) -> float:
    """Recover focal from the plane-axis orthonormality constraint.

    With K = diag(f, f, 1) about (cx, cy), let B = K^-1 H = [h1' | h2' | h3'].
    The true r1, r2 are orthonormal, giving h1'·h2' = 0 and |h1'| = |h2'|. Both
    reduce to an equation in 1/f^2; solve and average for robustness.
    """
    H = np.asarray(H, dtype=np.float64)
    # Translate so principal point is the origin (K becomes diag(f, f, 1)).
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    G = T @ H
    h1, h2 = G[:, 0], G[:, 1]
    # With K^-1 = diag(1/f, 1/f, 1): orthogonality h1·h2 = 0 over (x,y) plus
    # z-term => h1[0]h2[0] + h1[1]h2[1] = -f^2 h1[2]h2[2].
    denom_o = h1[2] * h2[2]
    num_o = -(h1[0] * h2[0] + h1[1] * h2[1])
    # Equal-norm: h1x^2+h1y^2 + f^2 h1z^2 = h2x^2+h2y^2 + f^2 h2z^2.
    denom_n = (h1[2] ** 2 - h2[2] ** 2)
    num_n = (h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)
    f2_candidates = []
    if abs(denom_o) > 1e-12 and num_o / denom_o > 0:
        f2_candidates.append(num_o / denom_o)
    if abs(denom_n) > 1e-12 and num_n / denom_n > 0:
        f2_candidates.append(num_n / denom_n)
    if not f2_candidates:
        raise ValueError("cannot recover focal from homography (degenerate view)")
    return float(np.sqrt(np.mean(f2_candidates)))


def homography_to_krt(
    H: np.ndarray, *, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a field(z=0)->image homography into (K, R, t).

    Fixes cx=width/2, cy=height/2, unit aspect; solves the focal; rebuilds an
    orthonormal R (r3 = r1 x r2, SVD-projected) and a consistently scaled t.
    """
    H = np.asarray(H, dtype=np.float64)
    cx, cy = width / 2.0, height / 2.0
    f = _solve_focal(H, cx, cy)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=np.float64)
    B = np.linalg.inv(K) @ H                       # [r1 | r2 | t] up to scale
    scale = 1.0 / np.linalg.norm(B[:, 0])
    r1 = B[:, 0] * scale
    r2 = B[:, 1] * scale
    t = B[:, 2] * scale
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    # Project to the nearest proper rotation.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # Ensure the camera looks toward the field (positive depth at field center).
    if (R @ np.array([0.0, 0.0, 0.0]) + t)[2] < 0:
        t = -t
        R = R @ np.diag([1.0, 1.0, 1.0])  # depth sign handled by t flip
    return K, R, t
```

- [ ] **Step 4: Run** — `python -m pytest tests/test_decompose_homography.py -q` → PASS. (If the depth-sign branch causes a projection mismatch, the round-trip test will catch it — verify both tests pass before committing.)

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/decompose_homography.py tests/test_decompose_homography.py
git add nfl_gsplat/calibration/decompose_homography.py tests/test_decompose_homography.py
git commit -m "Add planar homography <-> (K,R,t) decomposition

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `CameraTrack` + per-frame `cameras.npz` I/O

**Files:**
- Modify: `nfl_gsplat/calibration/cameras_io.py`
- Test: `tests/test_camera_track.py`

- [ ] **Step 1: Write the failing test** — `tests/test_camera_track.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.cameras_io import (
    CameraTrack, load_camera_track, write_camera_track,
)
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


def _track(T=5, W=1920, H=1080):
    K = np.stack([np.array([[2600.0 + i, 0, W / 2], [0, 2600.0 + i, H / 2], [0, 0, 1]])
                  for i in range(T)])
    R = np.stack([np.eye(3) for _ in range(T)])
    t = np.stack([np.array([0.0, 0.0, 20.0 + i]) for i in range(T)])
    conf = np.ones(T)
    return CameraTrack(K=K, R=R, t=t, conf=conf, width=W, height=H)


def test_camera_track_at_returns_frame_slice():
    tr = _track()
    intr, pose = tr.at(3)
    assert isinstance(intr, CameraIntrinsics) and isinstance(pose, CameraPose)
    assert intr.fx == 2603.0
    assert pose.t[2] == 23.0
    assert (intr.width, intr.height) == (1920, 1080)


def test_camera_track_at_clamps_out_of_range():
    tr = _track(T=5)
    # frames past the end clamp to the last (clips can differ by a frame).
    assert tr.at(99)[1].t[2] == 24.0


def test_write_load_roundtrip(tmp_path):
    tracks = {"sideline": _track(), "endzone": _track()}
    p = tmp_path / "cameras.npz"
    write_camera_track(p, tracks, fps=59.94)
    loaded = load_camera_track(p)
    assert set(loaded) == {"sideline", "endzone"}
    np.testing.assert_allclose(loaded["sideline"].K, tracks["sideline"].K)
    assert loaded["sideline"].at(0)[0].fx == 2600.0


def test_load_missing_raises(tmp_path):
    with pytest.raises(SetupError, match="cameras.npz"):
        load_camera_track(tmp_path / "nope.npz")
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_camera_track.py -q` → FAIL (ImportError).

- [ ] **Step 3: Replace `nfl_gsplat/calibration/cameras_io.py`** with the per-frame version (keep `_camera_from_entry` is dropped; old `load_cameras` removed — Task 5-8 migrate callers):

```python
"""Per-frame camera calibration I/O.

Calibration is per-frame (cameras pan/tilt/zoom during a play). A
:class:`CameraTrack` holds per-frame K/R/t for one camera; ``cameras.npz`` packs
all cameras of a play. ``.at(frame)`` returns the (CameraIntrinsics, CameraPose)
for a given frame (clamped to range, since the two synced clips may differ by a
frame). Produced by scripts/02b_track_calibration.py; consumed by every 3D stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


@dataclass(frozen=True)
class CameraTrack:
    K: np.ndarray        # [T, 3, 3]
    R: np.ndarray        # [T, 3, 3]
    t: np.ndarray        # [T, 3]
    conf: np.ndarray     # [T]
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return int(self.K.shape[0])

    def at(self, frame: int) -> tuple[CameraIntrinsics, CameraPose]:
        i = max(0, min(int(frame), self.num_frames - 1))
        K = self.K[i]
        intr = CameraIntrinsics(
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=int(self.width), height=int(self.height),
        )
        return intr, CameraPose(R=self.R[i].astype(np.float64), t=self.t[i].astype(np.float64))


def write_camera_track(path: Path | str, tracks: dict[str, CameraTrack], *, fps: float) -> Path:
    """Write all cameras to a single cameras.npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cams": np.array(list(tracks.keys())),
        "fps": np.array(float(fps)),
    }
    for cam, tr in tracks.items():
        arrays[f"{cam}_K"] = tr.K.astype(np.float64)
        arrays[f"{cam}_R"] = tr.R.astype(np.float64)
        arrays[f"{cam}_t"] = tr.t.astype(np.float64)
        arrays[f"{cam}_conf"] = tr.conf.astype(np.float64)
        arrays[f"{cam}_wh"] = np.array([tr.width, tr.height], dtype=np.int64)
    np.savez(path, **arrays)
    return path


def load_camera_track(path: Path | str) -> dict[str, CameraTrack]:
    """Load ``{cam: CameraTrack}`` from a cameras.npz. Raises SetupError if missing."""
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"cameras.npz not found at {path}. Calibrate keyframes "
            "(scripts/02_calibrate_cameras.py --play-dir <dir> --annotate) then run "
            "scripts/02b_track_calibration.py --play-dir <dir>. See SETUP.md §3."
        )
    d = np.load(path, allow_pickle=False)
    cams = [str(c) for c in d["cams"]]
    out: dict[str, CameraTrack] = {}
    for cam in cams:
        wh = d[f"{cam}_wh"]
        out[cam] = CameraTrack(
            K=d[f"{cam}_K"], R=d[f"{cam}_R"], t=d[f"{cam}_t"], conf=d[f"{cam}_conf"],
            width=int(wh[0]), height=int(wh[1]),
        )
    if not out:
        raise SetupError(f"{path}: no cameras found.")
    return out


def constant_track(intr: CameraIntrinsics, pose: CameraPose, num_frames: int) -> CameraTrack:
    """A CameraTrack with one pose repeated — the single-pose shim for testing
    and for static-camera footage."""
    K = np.repeat(intr.K()[None], num_frames, axis=0)
    R = np.repeat(np.asarray(pose.R, float)[None], num_frames, axis=0)
    t = np.repeat(np.asarray(pose.t, float).reshape(1, 3), num_frames, axis=0)
    return CameraTrack(K=K, R=R, t=t, conf=np.ones(num_frames),
                       width=intr.width, height=intr.height)
```

- [ ] **Step 4: Run** — `python -m pytest tests/test_camera_track.py -q` → PASS (4 tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/cameras_io.py tests/test_camera_track.py
git add nfl_gsplat/calibration/cameras_io.py tests/test_camera_track.py
git commit -m "cameras_io: per-frame CameraTrack + cameras.npz I/O (replaces load_cameras)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Thread per-frame cameras through the 3D consumers

**Files (read each, then edit the per-frame call sites):**
- Modify: `nfl_gsplat/tracking/cross_cam_reid.py`
- Modify: `nfl_gsplat/ball/kalman_3d.py`, `nfl_gsplat/ball/run_ball.py`
- Modify: `nfl_gsplat/pose/run_pose.py` (its `extract_observations` + triangulation calls)
- Modify: `nfl_gsplat/field/build_transforms.py`
- Test: extend `tests/test_triangulation.py` (or add `tests/test_percam_threading.py`)

The replacement recipe is identical everywhere: where code previously did
`cameras = load_cameras(path)` and used `cameras[cam] = (intr, pose)` for the
whole play, it now does `tracks = load_camera_track(path)` and, inside the
per-frame loop, `intr, pose = tracks[cam].at(frame)`. The pure geometry
functions (`project_to_plane_z0`, `triangulate_two_views`, `project_points`) are
unchanged — they already take `(K, R, t)` per call.

- [ ] **Step 1: Add a threading test** — `tests/test_percam_threading.py` proving a *moving* camera triangulates correctly only with per-frame params:

```python
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.utils.geometry import CameraPose, project_points, triangulate_two_views


def _two_cam_tracks(T=3):
    # Two cameras, each panning slightly per frame; known world point moves too.
    def cam(yaw0, dyaw, cx_off):
        Ks, Rs, ts = [], [], []
        for i in range(T):
            y = np.deg2rad(yaw0 + dyaw * i)
            R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
            K = np.array([[2000.0, 0, 960], [0, 2000.0, 540], [0, 0, 1]], float)
            Ks.append(K); Rs.append(R); ts.append(np.array([cx_off, 5.0, 30.0]))
        return CameraTrack(np.array(Ks), np.array(Rs), np.array(ts), np.ones(T), 1920, 1080)
    return {"a": cam(10, 2.0, -20.0), "b": cam(-10, -2.0, 20.0)}


def test_per_frame_triangulation_recovers_moving_point():
    tracks = _two_cam_tracks()
    truth = [np.array([1.0 * i, 2.0, 0.5]) for i in range(3)]
    for frame, X in enumerate(truth):
        ia, pa = tracks["a"].at(frame)
        ib, pb = tracks["b"].at(frame)
        uva = project_points(X[None], ia.K(), pa.R, pa.t)
        uvb = project_points(X[None], ib.K(), pb.R, pb.t)
        Pa = ia.K() @ pa.extrinsic_3x4()
        Pb = ib.K() @ pb.extrinsic_3x4()
        Xhat = triangulate_two_views(uva, uvb, Pa, Pb)[0]
        assert np.allclose(Xhat, X, atol=1e-3)
```

- [ ] **Step 2: Run** — `python -m pytest tests/test_percam_threading.py -q` → PASS already (it only exercises existing geometry + new CameraTrack). This test guards the contract the consumers must honor.

- [ ] **Step 3: Edit `cross_cam_reid.py`** — replace `from nfl_gsplat.calibration.cameras_io import load_cameras` with `load_camera_track`; in the function that projects foot points per `(cam, frame)`, change the camera lookup to `intr, pose = tracks[cam].at(int(frame))` and pass `intr.K(), pose.R, pose.t` into `project_to_plane_z0`. (Read the file to find the per-detection loop; the detection rows carry a `frame` column.)

- [ ] **Step 4: Edit `ball/kalman_3d.py` + `run_ball.py`** — change `build_and_write_ball_track`/`run_kalman` (and `_triangulate_ball_frame`) so the `cameras` argument is `dict[str, CameraTrack]` and each per-frame triangulation uses `cameras[a].at(frame)` / `cameras[b].at(frame)`. In `run_ball.py` replace `load_cameras(pdir.cameras_json)` with `load_camera_track(pdir.cameras_json)` (and update `pdir.cameras_json`→`pdir` path: see Task 9 for the `.npz` path rename).

- [ ] **Step 5: Edit `pose/run_pose.py`** — `extract_observations` triangulates joints per `(cam, frame)`; replace its `load_cameras` + per-cam `(intr,pose)` with `load_camera_track` + `tracks[cam].at(frame)` inside the per-frame loop before `triangulate_joints_two_view`.

- [ ] **Step 6: Edit `field/build_transforms.py`** — for each extracted frame index `k`, use `tracks[cam].at(k)` so each `transforms.json` entry carries that frame's own pose (camera motion = more parallax for nerfstudio).

- [ ] **Step 7: Run the affected suites** — `python -m pytest tests/test_triangulation.py tests/test_tracking.py tests/test_ball.py tests/test_field.py tests/test_percam_threading.py -q`. Fix until green. Then full: `python -m pytest -m "not gpu and not slow and not real_video" -q`.

- [ ] **Step 8: Lint + commit**

```bash
python -m ruff check nfl_gsplat/tracking/cross_cam_reid.py nfl_gsplat/ball nfl_gsplat/pose/run_pose.py nfl_gsplat/field/build_transforms.py tests/test_percam_threading.py
git add -A
git commit -m "Thread per-frame CameraTrack through reID/triangulation/ball/field

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 2 — Multi-keyframe annotation

### Task 4: Multi-keyframe annotation in the calibration script

**Files:**
- Modify: `nfl_gsplat/calibration/annotate_gui.py` (already supports `frame_index`)
- Modify: `scripts/02_calibrate_cameras.py`
- Create: `nfl_gsplat/calibration/keyframes.py` (pure load/save of keyframes JSON)
- Test: `tests/test_keyframes.py`

- [ ] **Step 1: Write the failing test** — `tests/test_keyframes.py`:

```python
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.keyframes import (
    Keyframe, load_keyframes, save_keyframes,
)


def test_keyframes_roundtrip(tmp_path):
    kfs = [
        Keyframe(frame=0, landmarks={"mid_50_left_hash": (1199.0, 654.0)}),
        Keyframe(frame=300, landmarks={"home_25_right_hash": (1101.0, 459.0)}),
    ]
    p = tmp_path / "sideline_keyframes.json"
    save_keyframes(p, kfs)
    got = load_keyframes(p)
    assert [k.frame for k in got] == [0, 300]
    assert got[0].landmarks["mid_50_left_hash"] == (1199.0, 654.0)


def test_load_keyframes_sorted_by_frame(tmp_path):
    p = tmp_path / "kf.json"
    save_keyframes(p, [Keyframe(5, {"a": (1.0, 2.0)}), Keyframe(1, {"b": (3.0, 4.0)})])
    assert [k.frame for k in load_keyframes(p)] == [1, 5]
```

- [ ] **Step 2: Run** — `python -m pytest tests/test_keyframes.py -q` → FAIL (ImportError).

- [ ] **Step 3: Create `nfl_gsplat/calibration/keyframes.py`:**

```python
"""Per-camera keyframe annotations for per-frame calibration.

A keyframe is a frame index + the landmark pixel clicks on that frame. Stored as
``{cam}_keyframes.json`` so the batch tracker (02b) re-uses them without
re-annotating. Anchors are solved by solve_pnp; the tracker fills the frames
between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.io import read_json, write_json


@dataclass(frozen=True)
class Keyframe:
    frame: int
    landmarks: dict[str, tuple[float, float]]   # name -> (u, v)


def save_keyframes(path: Path | str, keyframes: list[Keyframe]) -> Path:
    payload = [
        {"frame": int(k.frame),
         "landmarks": [{"name": n, "uv": [float(u), float(v)]}
                       for n, (u, v) in k.landmarks.items()]}
        for k in keyframes
    ]
    return write_json(path, payload)


def load_keyframes(path: Path | str) -> list[Keyframe]:
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"keyframes not found at {path}. Annotate first: "
            "scripts/02_calibrate_cameras.py --play-dir <dir> --annotate. See SETUP.md §3."
        )
    out: list[Keyframe] = []
    for entry in read_json(path):
        lms = {d["name"]: (float(d["uv"][0]), float(d["uv"][1])) for d in entry["landmarks"]}
        out.append(Keyframe(frame=int(entry["frame"]), landmarks=lms))
    return sorted(out, key=lambda k: k.frame)
```

- [ ] **Step 4: Run** — `python -m pytest tests/test_keyframes.py -q` → PASS.

- [ ] **Step 5: Update `scripts/02_calibrate_cameras.py`** to write `{cam}_keyframes.json` from one-or-more keyframes. Add a repeatable option `keyframe: list[int] = typer.Option([0], "--keyframe")`; for each camera, for each requested frame, call `run_annotator(video, tmp_json, frame_index=k)` (the GUI returns `[{name,uv,frame}]`), assemble into `Keyframe`s, `save_keyframes(pd.dir / f"{cam}_keyframes.json", ...)`. It no longer writes `cameras.json` (that's 02b's job). Keep it display-gated. Confirm it parses: `python -c "import ast; ast.parse(open('scripts/02_calibrate_cameras.py').read())"`.

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/keyframes.py scripts/02_calibrate_cameras.py tests/test_keyframes.py
git add -A
git commit -m "Multi-keyframe annotation: keyframes.py + 02_calibrate writes keyframes.json

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 3 — Homography tracking core

### Task 5: `track_homography.py` — pure tracking/blend/gap logic

**Files:**
- Create: `nfl_gsplat/calibration/track_homography.py`
- Test: `tests/test_track_homography.py`

The module separates **pure** logic (anchor→homography, bidirectional blend,
confidence-gap detection, decompose-to-track) from the **OpenCV seam**
(`_estimate_interframe_homographies`, which runs optical flow + RANSAC). Tests
cover the pure logic with synthetic homography sequences; the seam is
monkeypatched.

- [ ] **Step 1: Write the failing test** — `tests/test_track_homography.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration import track_homography as th
from nfl_gsplat.calibration.decompose_homography import krt_to_homography
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.geometry import CameraIntrinsics


def _anchor_H(fx, yaw_deg, W=1920, H=1080):
    intr = CameraIntrinsics(fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H)
    y = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
    t = np.array([0.0, 5.0, 25.0])
    return krt_to_homography(intr.K(), R, t)


def test_blend_between_anchors_snaps_to_endpoints():
    Ha, Hb = _anchor_H(2500, 0.0), _anchor_H(2700, 12.0)
    # Identity inter-frame steps => forward chain == Ha, backward chain == Hb.
    fwd = [Ha.copy() for _ in range(5)]
    bwd = [Hb.copy() for _ in range(5)]
    blended = th.blend_chains(fwd, bwd)
    assert np.allclose(blended[0] / blended[0][2, 2], Ha / Ha[2, 2], atol=1e-9)
    assert np.allclose(blended[-1] / blended[-1][2, 2], Hb / Hb[2, 2], atol=1e-9)


def test_confidence_gap_detection_raises_with_range():
    conf = np.array([1.0, 0.9, 0.2, 0.1, 0.15, 0.95, 1.0])
    with pytest.raises(CalibrationError, match="frames 2-4"):
        th.check_confidence(conf, min_conf=0.5, max_gap=0)


def test_check_confidence_passes_when_above_threshold():
    th.check_confidence(np.array([0.8, 0.7, 0.9]), min_conf=0.5, max_gap=0)


def test_assemble_track_decomposes_each_frame():
    Ha, Hb = _anchor_H(2500, 0.0), _anchor_H(2600, 6.0)
    Hs = [Ha, (Ha + Hb) / 2, Hb]
    conf = np.ones(3)
    tr = th.assemble_track(Hs, conf, width=1920, height=1080)
    assert tr.num_frames == 3
    assert tr.K.shape == (3, 3, 3)
    assert 2000 < tr.K[0, 0, 0] < 3200       # plausible focal
```

- [ ] **Step 2: Run** — `python -m pytest tests/test_track_homography.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement `nfl_gsplat/calibration/track_homography.py`:**

```python
"""Track the field ground-plane homography across a play (per-frame calibration).

Between consecutive PnP keyframe anchors we track the field->image homography
frame-to-frame on static field features (players masked), forward from the left
anchor and backward from the right anchor, then blend by normalized distance so
the chain snaps to both anchors and drift stays bounded. Each blended homography
is decomposed to (K, R, t). Confidence (RANSAC inlier ratio etc.) is gated:
a low-confidence run fails loud, asking for another keyframe.

Pure logic (blend / gap check / assemble) is unit-tested; the OpenCV optical-flow
+ RANSAC estimation is the `_estimate_interframe_homographies` seam.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.calibration.decompose_homography import homography_to_krt
from nfl_gsplat.errors import CalibrationError


@dataclass(frozen=True)
class TrackConfig:
    min_conf: float = 0.35      # per-frame confidence floor
    max_gap: int = 0            # allowed consecutive frames below floor
    max_features: int = 800
    ransac_thresh_px: float = 3.0


def _norm(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2]


def blend_chains(forward: list[np.ndarray], backward: list[np.ndarray]) -> list[np.ndarray]:
    """Blend a forward-tracked and backward-tracked homography chain.

    ``forward[i]`` is tracked from the left anchor (i=0 is the left anchor),
    ``backward[i]`` from the right anchor (last index is the right anchor).
    Weight w = i/(n-1): w=0 -> all forward (left anchor), w=1 -> all backward.
    Homographies are blended in normalized form (then re-normalized).
    """
    n = len(forward)
    assert n == len(backward)
    out: list[np.ndarray] = []
    for i in range(n):
        w = 0.0 if n == 1 else i / (n - 1)
        H = (1.0 - w) * _norm(forward[i]) + w * _norm(backward[i])
        out.append(_norm(H))
    return out


def check_confidence(conf: np.ndarray, *, min_conf: float, max_gap: int) -> None:
    """Raise CalibrationError naming the first run of > max_gap frames below floor."""
    below = np.asarray(conf) < min_conf
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) > max_gap:
                raise CalibrationError(
                    f"camera tracking lost lock on frames {i}-{j - 1} "
                    f"(confidence < {min_conf}). Add a keyframe in that range and "
                    f"re-run scripts/02b_track_calibration.py. See SETUP.md §3."
                )
            i = j
        else:
            i += 1


def assemble_track(
    homographies: list[np.ndarray], conf: np.ndarray, *, width: int, height: int,
) -> CameraTrack:
    """Decompose each frame's homography into (K, R, t) -> CameraTrack."""
    T = len(homographies)
    K = np.zeros((T, 3, 3)); R = np.zeros((T, 3, 3)); t = np.zeros((T, 3))
    for i, H in enumerate(homographies):
        Ki, Ri, ti = homography_to_krt(H, width=width, height=height)
        K[i], R[i], t[i] = Ki, Ri, ti
    return CameraTrack(K=K, R=R, t=t, conf=np.asarray(conf, float),
                       width=width, height=height)


def _estimate_interframe_homographies(
    video_path, frame_a: int, frame_b: int, masks_provider, cfg: TrackConfig,
) -> tuple[list[np.ndarray], np.ndarray]:
    """OpenCV seam: per-frame homography from frame_a to each frame in (a, b].

    Returns (Hrel list of 3x3 mapping frame_a -> frame_k, conf array). Uses
    goodFeaturesToTrack on the field region (outside masks_provider(k) boxes),
    calcOpticalFlowPyrLK to the next frame, findHomography(RANSAC). Monkeypatched
    in tests. Implemented against cv2 at GPU/real-video bring-up.
    """
    import cv2  # noqa: F401  (real implementation added at bring-up)

    raise NotImplementedError(
        "optical-flow homography estimation is finalized against real video at "
        "single-play bring-up; tests monkeypatch this seam."
    )


def track_camera_sequence(
    video_path,
    anchors: dict[int, np.ndarray],     # frame -> field->image homography (from solve_pnp)
    *,
    num_frames: int,
    width: int,
    height: int,
    masks_provider=lambda frame: [],
    cfg: TrackConfig = TrackConfig(),
) -> CameraTrack:
    """Build a per-frame CameraTrack from keyframe-anchor homographies.

    For each consecutive anchor pair, track forward from the left and backward
    from the right, blend, and gate confidence. Frames before the first / after
    the last anchor hold that anchor's homography (conf=1 at anchors).
    """
    anchor_frames = sorted(anchors)
    if not anchor_frames:
        raise CalibrationError("no keyframe anchors provided")
    Hs: list[np.ndarray] = [None] * num_frames       # type: ignore
    conf = np.zeros(num_frames)

    # Clamp ends to the nearest anchor.
    for f in range(0, anchor_frames[0]):
        Hs[f] = _norm(anchors[anchor_frames[0]]); conf[f] = 1.0
    for f in range(anchor_frames[-1], num_frames):
        Hs[f] = _norm(anchors[anchor_frames[-1]]); conf[f] = 1.0

    for a, b in zip(anchor_frames[:-1], anchor_frames[1:]):
        rel_fwd, c_fwd = _estimate_interframe_homographies(video_path, a, b, masks_provider, cfg)
        rel_bwd, c_bwd = _estimate_interframe_homographies(video_path, b, a, masks_provider, cfg)
        forward = [anchors[a]] + [Hrel @ anchors[a] for Hrel in rel_fwd]
        backward = list(reversed([anchors[b]] + [Hrel @ anchors[b] for Hrel in rel_bwd]))
        seg = blend_chains(forward, backward)
        seg_conf = np.minimum(np.r_[1.0, c_fwd], np.r_[list(reversed(c_bwd)), 1.0])
        for k, f in enumerate(range(a, b + 1)):
            Hs[f] = seg[k]; conf[f] = seg_conf[k]

    check_confidence(conf, min_conf=cfg.min_conf, max_gap=cfg.max_gap)
    return assemble_track(Hs, conf, width=width, height=height)
```

- [ ] **Step 4: Run** — `python -m pytest tests/test_track_homography.py -q` → PASS (the 4 pure tests; `_estimate_interframe_homographies` is not called by them).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/track_homography.py tests/test_track_homography.py
git add nfl_gsplat/calibration/track_homography.py tests/test_track_homography.py
git commit -m "Add homography-tracking core (blend, confidence gating, decompose)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end track test with a monkeypatched flow seam

**Files:**
- Test: extend `tests/test_track_homography.py`

- [ ] **Step 1: Add a synthetic full-sequence test** that monkeypatches `_estimate_interframe_homographies` to return the *true* relative homographies of a known panning trajectory, then asserts `track_camera_sequence` recovers per-frame `(K,R,t)` that reproject field points correctly:

```python
def test_track_camera_sequence_recovers_known_pan(monkeypatch):
    from nfl_gsplat.calibration.decompose_homography import krt_to_homography
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    W, H, T = 1920, 1080, 7
    def truth(i):
        intr = CameraIntrinsics(2500 + 20 * i, 2500 + 20 * i, W / 2, H / 2, W, H)
        y = np.deg2rad(2.0 * i)
        R = np.array([[np.cos(y), -np.sin(y), 0], [0, 0, -1], [np.sin(y), np.cos(y), 0]], float)
        return intr, CameraPose(R=R, t=np.array([0.0, 5.0, 25.0]))
    Htrue = [krt_to_homography(*[truth(i)[0].K(), truth(i)[1].R, truth(i)[1].t]) for i in range(T)]
    anchors = {0: Htrue[0], T - 1: Htrue[T - 1]}

    def fake_est(video, a, b, masks, cfg):
        step = 1 if b > a else -1
        idxs = list(range(a + step, b + step, step))
        rel = [Htrue[k] @ np.linalg.inv(Htrue[a]) for k in idxs]
        return rel, np.ones(len(idxs))
    monkeypatch.setattr(th, "_estimate_interframe_homographies", fake_est)

    tr = th.track_camera_sequence("v.mp4", anchors, num_frames=T, width=W, height=H)
    fld = np.array([[0, 0, 0], [25, 10, 0], [-20, -8, 0]], float)
    for i in range(T):
        it, pt = truth(i)
        ie, pe = tr.at(i)
        assert np.allclose(project_points(fld, it.K(), pt.R, pt.t),
                           project_points(fld, ie.K(), pe.R, pe.t), atol=2.0)
```

- [ ] **Step 2: Run** — `python -m pytest tests/test_track_homography.py -q` → PASS. If the mid-sequence reprojection error exceeds 2 px, the blend/normalization has a bug — fix `blend_chains`/`assemble_track` until it passes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_track_homography.py
git commit -m "Test: track_camera_sequence recovers a known pan end-to-end (mocked flow)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 4 — Batch stage + pipeline wiring

### Task 7: `02b_track_calibration.py` batch stage + anchor solving

**Files:**
- Create: `nfl_gsplat/calibration/run_tracking.py` (env-light core: keyframes → anchors → CameraTrack)
- Create: `scripts/02b_track_calibration.py` (thin CLI)
- Modify: `nfl_gsplat/paths.py` (`PlayDir.cameras_json` → add `cameras_npz`)
- Test: `tests/test_run_tracking.py`

- [ ] **Step 1: Add `cameras_npz` to `PlayDir`** in `nfl_gsplat/paths.py` (keep `cameras_json` for the keyframe-era annotations dir nothing else):

```python
    @property
    def cameras_npz(self) -> Path:
        return self.dir / "cameras.npz"

    def keyframes_json(self, cam: str) -> Path:
        return self.dir / f"{cam}_keyframes.json"
```
Add a test to `tests/test_config_paths.py`:
```python
def test_play_dir_camera_paths():
    from nfl_gsplat.paths import PlayDir
    pd = PlayDir(season="2024", week=1, matchup="NO_at_ATL", play_id="play_001")
    assert pd.cameras_npz.name == "cameras.npz"
    assert pd.keyframes_json("sideline").name == "sideline_keyframes.json"
```

- [ ] **Step 2: Write `tests/test_run_tracking.py`** (monkeypatch solve + track seams; assert it writes cameras.npz for both cams):

```python
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration import run_tracking as rt
from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track
from nfl_gsplat.calibration.keyframes import Keyframe, save_keyframes


def test_build_camera_npz_writes_both_cams(tmp_path, monkeypatch):
    for cam in ("sideline", "endzone"):
        save_keyframes(tmp_path / f"{cam}_keyframes.json",
                       [Keyframe(0, {"mid_50_left_hash": (10.0, 20.0)})])
    monkeypatch.setattr(rt, "_anchor_homographies",
                        lambda kfs, wh: {0: np.eye(3)})
    monkeypatch.setattr(rt, "_track",
                        lambda video, anchors, **kw: CameraTrack(
                            K=np.repeat(np.eye(3)[None], 4, 0),
                            R=np.repeat(np.eye(3)[None], 4, 0),
                            t=np.zeros((4, 3)), conf=np.ones(4),
                            width=1920, height=1080))
    monkeypatch.setattr(rt, "_video_dims_frames",
                        lambda video: (1920, 1080, 4))
    out = rt.build_camera_npz(
        play_dir=tmp_path,
        videos={"sideline": tmp_path / "s.mp4", "endzone": tmp_path / "e.mp4"},
        fps=30.0,
    )
    tracks = load_camera_track(out)
    assert set(tracks) == {"sideline", "endzone"}
    assert tracks["sideline"].num_frames == 4
```

- [ ] **Step 3: Implement `nfl_gsplat/calibration/run_tracking.py`:**

```python
"""Batch per-frame calibration: keyframes -> anchors -> tracked CameraTrack.

Headless (no display). For each camera: solve each keyframe's PnP -> anchor
field->image homography, then track_camera_sequence between anchors, and pack all
cameras into one cameras.npz. The PnP solve + video reads are isolated seams.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.decompose_homography import krt_to_homography
from nfl_gsplat.calibration.keyframes import load_keyframes
from nfl_gsplat.calibration.track_homography import TrackConfig, track_camera_sequence


def _video_dims_frames(video: Path) -> tuple[int, int, int]:
    from nfl_gsplat.utils.video import ffprobe_meta
    m = ffprobe_meta(video)
    return m.width, m.height, m.num_frames


def _anchor_homographies(keyframes, wh: tuple[int, int]) -> dict[int, np.ndarray]:
    """Solve each keyframe to a field->image homography via solve_pnp."""
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
    from nfl_gsplat.utils.io import write_json
    import tempfile

    width, height = wh
    out: dict[int, np.ndarray] = {}
    for kf in keyframes:
        entries = [{"name": n, "uv": [u, v], "frame": kf.frame}
                   for n, (u, v) in kf.landmarks.items()]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            tmp = Path(fh.name)
        write_json(tmp, entries)
        res = solve_pnp_from_annotations(tmp, image_size=(width, height))
        tmp.unlink(missing_ok=True)
        out[kf.frame] = krt_to_homography(res.intrinsics.K(), res.pose.R, res.pose.t)
    return out


def _track(video, anchors, *, num_frames, width, height, masks_provider, cfg):
    return track_camera_sequence(
        video, anchors, num_frames=num_frames, width=width, height=height,
        masks_provider=masks_provider, cfg=cfg,
    )


def build_camera_npz(
    *, play_dir: Path | str, videos: dict[str, Path], fps: float,
    masks_provider=lambda cam: (lambda frame: []),
    cfg: TrackConfig = TrackConfig(),
) -> Path:
    """Produce cameras.npz for all cameras from their keyframes + clips."""
    play_dir = Path(play_dir)
    tracks: dict[str, CameraTrack] = {}
    for cam, video in videos.items():
        keyframes = load_keyframes(play_dir / f"{cam}_keyframes.json")
        width, height, num_frames = _video_dims_frames(video)
        anchors = _anchor_homographies(keyframes, (width, height))
        tracks[cam] = _track(
            video, anchors, num_frames=num_frames, width=width, height=height,
            masks_provider=masks_provider(cam), cfg=cfg,
        )
    return write_camera_track(play_dir / "cameras.npz", tracks, fps=fps)
```

- [ ] **Step 4: Run** — `python -m pytest tests/test_run_tracking.py tests/test_config_paths.py -q` → PASS.

- [ ] **Step 5: Create `scripts/02b_track_calibration.py`:**

```python
"""Per-frame calibration (batch, headless): keyframes + clips -> cameras.npz.

    python scripts/02b_track_calibration.py --play-dir data/2025/week_04/SEA_at_AZ/play_001

Reads each camera's {cam}_keyframes.json (from 02_calibrate_cameras.py --annotate),
tracks the field homography across the clip (players masked via YOLO), and writes
cameras.npz. Fails loud if tracking loses lock (add a keyframe + re-run).
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.calibration.run_tracking import build_camera_npz
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _yolo_masks_provider(cam: str):
    """Return frame->bboxes using the play's tracks.parquet if present, else empty.
    (A direct YOLO pass is wired at bring-up; tracks.parquet already has boxes.)"""
    return lambda frame: []


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    pd = PlayDir.from_dir(play_dir)
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}
    out = build_camera_npz(play_dir=pd.dir, videos=videos, fps=meta.fps,
                           masks_provider=_yolo_masks_provider)
    _LOG.info(f"wrote per-frame calibration → {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/run_tracking.py scripts/02b_track_calibration.py nfl_gsplat/paths.py tests/test_run_tracking.py
python -c "import ast; ast.parse(open('scripts/02b_track_calibration.py').read())"
git add -A
git commit -m "Add 02b batch per-frame calibration stage (keyframes -> cameras.npz)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Wire into `04_process_play.sh` + docs

**Files:**
- Modify: `scripts/04_process_play.sh`
- Modify: `SETUP.md` (§3 calibration), `INSTRUCTIONS.md` (calibration step)

- [ ] **Step 1: Insert the tracking step into `04_process_play.sh`** after detection (so player masks / tracks exist) and before cross-cam re-ID. Add, in the `nfl_smplx` env block right after the `[1/6]` tracking detect step (or as its own step), keeping `bash -n` valid:

```bash
echo "=== [1b] per-frame calibration tracking → cameras.npz  (env: nfl_smplx) ==="
conda activate nfl_smplx
python scripts/02b_track_calibration.py --play-dir "$PLAY_DIR" $CFG
conda deactivate
```
Run `bash -n scripts/04_process_play.sh`.

- [ ] **Step 2: Update `SETUP.md` §3 and `INSTRUCTIONS.md`** calibration section: two steps now — (1) `02_calibrate_cameras.py --play-dir <dir> --annotate --keyframe 0 --keyframe 300 …` (interactive, OnDemand desktop) writes `{cam}_keyframes.json`; (2) `02b_track_calibration.py --play-dir <dir>` (headless) writes `cameras.npz`; if it fails loud naming a frame range, add a keyframe there and re-run step 2. Note cameras are now per-frame (handles PTZ All-22).

- [ ] **Step 3: Full suite + commit**

```bash
python -m pytest -m "not gpu and not slow and not real_video" -q   # expect all green
git add -A
git commit -m "Wire per-frame calibration into 04 driver + docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** homography↔KRT decompose → Task 1; per-frame `cameras.npz`/`CameraTrack` → Task 2; consumer threading (reID/triangulation/ball/field) → Task 3; multi-keyframe annotation + `keyframes.json` → Task 4; tracking core (blend/confidence/fail-loud/decompose) → Task 5–6; batch `02b` stage + anchor solve + `04` wiring → Task 7–8. The "fail loud" decision → `check_confidence` (Task 5). Player masking → `masks_provider` seam (Tasks 5/7, finalized at bring-up). All spec sections map to tasks.
- **Type consistency:** `CameraTrack(K,R,t,conf,width,height)` + `.at()` used identically in Tasks 2,3,5,6,7. `homography_to_krt(H,*,width,height)` / `krt_to_homography(K,R,t)` consistent across 1,5,7. `Keyframe(frame,landmarks)` consistent 4,7. `track_camera_sequence(video, anchors, *, num_frames, width, height, masks_provider, cfg)` consistent 5,6,7. `_estimate_interframe_homographies(video, a, b, masks_provider, cfg)` consistent 5,6.
- **Seams isolated & honest:** the only non-pure, bring-up-finalized pieces are `_estimate_interframe_homographies` (OpenCV optical flow) and `_yolo_masks_provider` (real masks). Both are explicitly seamed and monkeypatched in tests; everything else is CPU-tested.

## Known follow-ups (out of scope; finalize at single-play GPU/real-video bring-up)
- Real `_estimate_interframe_homographies` (goodFeaturesToTrack + calcOpticalFlowPyrLK + findHomography on masked field regions).
- Real `_yolo_masks_provider` (per-frame person boxes from the YOLO detector / tracks.parquet).
- Tune `TrackConfig` thresholds against actual All-22 footage.
