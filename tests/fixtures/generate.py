"""Generate a synthetic NFL fixture for unit + smoke tests.

Produces (under ``tests/fixtures/generated/``):
- ``cameras_gt.json``                  ground-truth (K, R, t) for both cameras
- ``{cam}_landmarks.json``             perfect pixel-exact annotations
- ``{cam}_frame.png``                  placeholder reference image with
                                       overlaid field lines (for visual sanity)
- ``players_gt.npz``                   3 player root positions + canonical joints
- ``ball_gt.npz``                      a parabolic ball trajectory

The fixture is CPU-only and uses no body-model weights: players are
represented by their 22 canonical body-joint positions (a T-pose template
baked into this file). Real SMPL-X bodies are optional and only needed for
the Phase 6 avatar tests, which are gated with ``pytest -m gpu``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HALF_LENGTH_M,
    HALF_WIDTH_M,
    NFL_LANDMARKS,
)
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points
from nfl_gsplat.utils.io import write_json, write_npz

# --- Fixture constants -------------------------------------------------------

FIXTURE_WIDTH = 1280
FIXTURE_HEIGHT = 720
FOV_Y_DEG = 40.0

# A minimal "T-pose" 22-joint template in SMPL-X body-joint order. Only the
# geometry is fixture-relevant; exact offsets don't need to match SMPL-X
# precisely for calibration/triangulation tests. Heights are in meters.
TEMPLATE_JOINTS_22 = np.array([
    [0.00, 0.00, 0.00],   #  0 pelvis
    [0.10, 0.00, 0.00],   #  1 left hip
    [-0.10, 0.00, 0.00],  #  2 right hip
    [0.00, 0.00, 0.08],   #  3 spine1
    [0.10, 0.00, -0.40],  #  4 left knee
    [-0.10, 0.00, -0.40], #  5 right knee
    [0.00, 0.00, 0.20],   #  6 spine2
    [0.10, 0.00, -0.80],  #  7 left ankle
    [-0.10, 0.00, -0.80], #  8 right ankle
    [0.00, 0.00, 0.32],   #  9 spine3
    [0.10, 0.08, -0.85],  # 10 left foot
    [-0.10, 0.08, -0.85], # 11 right foot
    [0.00, 0.00, 0.56],   # 12 neck
    [0.18, 0.00, 0.50],   # 13 left collar
    [-0.18, 0.00, 0.50],  # 14 right collar
    [0.00, 0.00, 0.72],   # 15 head
    [0.35, 0.00, 0.50],   # 16 left shoulder
    [-0.35, 0.00, 0.50],  # 17 right shoulder
    [0.62, 0.00, 0.50],   # 18 left elbow
    [-0.62, 0.00, 0.50],  # 19 right elbow
    [0.88, 0.00, 0.50],   # 20 left wrist
    [-0.88, 0.00, 0.50],  # 21 right wrist
], dtype=np.float64)

PLAYER_ROOTS = np.array([
    [0.0,  0.0,  0.92],      # midfield, at the centerline
    [8.0,  4.0,  0.92],      # home side, offense shifted right
    [-6.0, -3.0, 0.92],      # away side, defense
], dtype=np.float64)


# --- Camera rigging ---------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) \
        -> tuple[np.ndarray, np.ndarray]:
    """Compute (R, t) such that ``x_cam = R @ x_world + t`` with +Z_cam looking
    at ``target`` (OpenCV convention)."""
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12      # camera forward
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12      # camera right
    u = np.cross(r, f)                  # camera up (in world)
    # OpenCV: x=right, y=down, z=forward. Rows of R are world->cam basis.
    R = np.stack([r, -u, f], axis=0)
    t = -R @ eye
    return R, t


def _intrinsics_for_fov(width: int, height: int, fov_y_deg: float) -> CameraIntrinsics:
    fov_y = np.deg2rad(fov_y_deg)
    fy = 0.5 * height / np.tan(0.5 * fov_y)
    fx = fy                              # square pixels
    return CameraIntrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                            width=width, height=height)


def _sideline_camera() -> tuple[CameraIntrinsics, CameraPose]:
    eye = np.array([0.0, -40.0, 12.0])           # press box, sideline
    target = np.array([0.0, 0.0, 1.0])
    R, t = _look_at(eye, target)
    return _intrinsics_for_fov(FIXTURE_WIDTH, FIXTURE_HEIGHT, FOV_Y_DEG), CameraPose(R=R, t=t)


def _endzone_camera() -> tuple[CameraIntrinsics, CameraPose]:
    eye = np.array([-70.0, 0.0, 15.0])            # behind the away endzone
    target = np.array([0.0, 0.0, 1.0])
    R, t = _look_at(eye, target)
    return _intrinsics_for_fov(FIXTURE_WIDTH, FIXTURE_HEIGHT, FOV_Y_DEG), CameraPose(R=R, t=t)


# --- Image rendering --------------------------------------------------------

def _render_field_image(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                        width: int, height: int) -> np.ndarray:
    """Very simple visualization: field boundary + yard lines projected as
    polylines onto a green background. No shading, no players — this image
    exists so a human can eyeball a fixture, not for ML inference."""
    canvas = np.full((height, width, 3), (34, 110, 34), dtype=np.uint8)   # turf green

    # Field outline.
    corners = np.array([
        [+HALF_LENGTH_M, +HALF_WIDTH_M, 0.0],
        [+HALF_LENGTH_M, -HALF_WIDTH_M, 0.0],
        [-HALF_LENGTH_M, -HALF_WIDTH_M, 0.0],
        [-HALF_LENGTH_M, +HALF_WIDTH_M, 0.0],
    ])
    uv = project_points(corners, K, R, t).astype(np.int32)
    cv2.polylines(canvas, [uv.reshape(-1, 1, 2)], isClosed=True,
                  color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Yard lines.
    yl_xs = np.arange(-45.72, 45.72 + 0.01, 4.572)
    for x in yl_xs:
        segment = np.array([[x, +HALF_WIDTH_M, 0.0], [x, -HALF_WIDTH_M, 0.0]])
        uv = project_points(segment, K, R, t)
        if not np.isfinite(uv).all():
            continue
        p0, p1 = uv.astype(np.int32)
        cv2.line(canvas, tuple(p0), tuple(p1), (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


# --- Public entrypoint ------------------------------------------------------

def _visible_landmarks(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                       width: int, height: int) -> dict[str, tuple[float, float]]:
    """Return {name: (u, v)} for landmarks that project inside the frame."""
    names = list(NFL_LANDMARKS.keys())
    pts = np.stack([NFL_LANDMARKS[n] for n in names], axis=0)
    uv = project_points(pts, K, R, t)
    visible: dict[str, tuple[float, float]] = {}
    for name, (u, v) in zip(names, uv):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        if 0.0 <= u < width and 0.0 <= v < height:
            visible[name] = (float(u), float(v))
    return visible


def synthetic_bbox_for_player(
    root_xyz: np.ndarray,
    K: np.ndarray, R: np.ndarray, t: np.ndarray,
    *,
    body_height_m: float = 1.85,
    body_half_width_m: float = 0.40,
) -> tuple[float, float, float, float] | None:
    """Project a simple upright body-sized box standing at ``(root_xyz[0],
    root_xyz[1])`` on the ground into image space to produce ``(x1, y1, x2,
    y2)``. The box spans ``z ∈ [0, body_height_m]`` so that back-projecting
    the bbox bottom-center (the "foot point") to the Z=0 plane recovers the
    player's xy correctly — the property ``cross_cam_reid`` relies on.

    Returns ``None`` if behind-camera or any corner projects to NaN.
    """
    base_x, base_y = float(root_xyz[0]), float(root_xyz[1])
    corners = np.array([
        [base_x - body_half_width_m, base_y, 0.0],
        [base_x + body_half_width_m, base_y, 0.0],
        [base_x - body_half_width_m, base_y, body_height_m],
        [base_x + body_half_width_m, base_y, body_height_m],
    ])
    uv = project_points(corners, K, R, t)
    if not np.isfinite(uv).all():
        return None
    x1 = float(np.min(uv[:, 0]))
    x2 = float(np.max(uv[:, 0]))
    y1 = float(np.min(uv[:, 1]))
    y2 = float(np.max(uv[:, 1]))
    return x1, y1, x2, y2


def _ball_trajectory() -> np.ndarray:
    """Simple parabolic throw from ``(−10, 0, 2)`` to ``(+20, 5, 0)``, 2 s, 30 fps."""
    T = 60
    t = np.linspace(0.0, 2.0, T)
    x = np.linspace(-10.0, 20.0, T)
    y = np.linspace(0.0, 5.0, T)
    z = 2.0 + 8.0 * t - 0.5 * 9.81 * t * t
    z = np.clip(z, 0.0, None)
    return np.stack([x, y, z], axis=1)


def generate(out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intr_s, pose_s = _sideline_camera()
    intr_e, pose_e = _endzone_camera()

    cams = {
        "sideline": {
            "K": intr_s.K().tolist(),
            "R": pose_s.R.tolist(),
            "t": pose_s.t.tolist(),
            "width": intr_s.width,
            "height": intr_s.height,
        },
        "endzone": {
            "K": intr_e.K().tolist(),
            "R": pose_e.R.tolist(),
            "t": pose_e.t.tolist(),
            "width": intr_e.width,
            "height": intr_e.height,
        },
        "world": {
            "field_length_m": FIELD_LENGTH_M,
            "field_width_m": FIELD_WIDTH_M,
        },
    }
    write_json(out_dir / "cameras_gt.json", cams)

    # Per-camera landmark annotations (pixel-exact projections).
    for cam_name, (intr, pose) in [("sideline", (intr_s, pose_s)),
                                    ("endzone", (intr_e, pose_e))]:
        vis = _visible_landmarks(intr.K(), pose.R, pose.t, intr.width, intr.height)
        entries = [{"name": n, "uv": [u, v], "frame": 0} for n, (u, v) in vis.items()]
        write_json(out_dir / f"{cam_name}_landmarks.json", entries)

        img = _render_field_image(intr.K(), pose.R, pose.t, intr.width, intr.height)
        cv2.imwrite(str(out_dir / f"{cam_name}_frame.png"), img)

    # Synthetic players: 3 bodies in T-pose at fixed roots. Joints are in
    # world frame = ROOT + template offset.
    joints_world = PLAYER_ROOTS[:, None, :] + TEMPLATE_JOINTS_22[None, :, :]
    write_npz(out_dir / "players_gt.npz",
              roots=PLAYER_ROOTS, joints_world=joints_world.astype(np.float64))

    write_npz(out_dir / "ball_gt.npz", xyz=_ball_trajectory())

    return out_dir


if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "generated"
    path = generate(here)
    print(f"wrote synthetic fixture to {path}")
