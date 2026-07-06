"""Fixed-center joint solve (pretrained calibration, phase 3).

The camera is on a tripod: ONE shared center C for the whole play, per-frame
rotation r_t (Rodrigues) and focal f_t (fx=fy, principal point = image
center). Fit jointly to every fused correspondence with a robust loss —
per-frame PnP on planar telephoto views is multimodal (measured on real
footage: the same frame solves blind but fails with a neighbor's prior), and
a consistently mislabeled yard line is invisible per frame but irreconcilable
with the shared center.

Parameter vector layout: [C (3)] + per usable frame [r_t (3), f_t (1)],
frames in ascending index order.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.errors import CalibrationError

LAMBDA_F = 0.01          # px of residual per px of frame-to-frame focal change
LAMBDA_R = 50.0          # px of residual per radian of frame-to-frame rotation change
SMOOTH_MAX_GAP = 3       # only penalize usable-frame pairs this close in index
F_BOUNDS = (200.0, 40000.0)
MIN_INIT_FRAMES = 20
AUDIT_DROP_PX = 6.0
F_SCALE_PX = 3.0

FrameData = tuple[np.ndarray, np.ndarray]     # (world (N,3), uv (N,2))


def build_frame_data(corrs_by_frame, min_corrs: int = 4) -> dict[int, FrameData]:
    """Resolve landmark names to world points; keep frames with >= min_corrs."""
    out: dict[int, FrameData] = {}
    for fidx, corrs in corrs_by_frame.items():
        if not corrs or len(corrs) < min_corrs:
            continue
        world = np.stack([NFL_LANDMARKS[n] for (n, _uv) in corrs]).astype(np.float64)
        uv = np.array([p for (_n, p) in corrs], dtype=np.float64)
        out[int(fidx)] = (world, uv)
    return out


def pack_params(C, r_by_frame, f_by_frame, frame_ids) -> np.ndarray:
    parts = [np.asarray(C, np.float64).ravel()]
    for i in frame_ids:
        parts.append(np.asarray(r_by_frame[i], np.float64).ravel())
        parts.append(np.array([float(f_by_frame[i])]))
    return np.concatenate(parts)


def unpack_params(x, frame_ids):
    C = x[:3]
    r_by_frame, f_by_frame = {}, {}
    for k, i in enumerate(frame_ids):
        o = 3 + 4 * k
        r_by_frame[i] = x[o:o + 3]
        f_by_frame[i] = float(x[o + 3])
    return C, r_by_frame, f_by_frame


def _project(world, r, f, C, image_size):
    """Project (N,3) world points; camera center C, Rodrigues r, focal f."""
    import cv2
    R = cv2.Rodrigues(np.asarray(r, np.float64))[0]
    t = -R @ np.asarray(C, np.float64)
    x_cam = world @ R.T + t.reshape(1, 3)
    z = np.maximum(x_cam[:, 2], 1e-9)          # smooth guard: no NaN in residuals
    u = f * x_cam[:, 0] / z + image_size[0] / 2.0
    v = f * x_cam[:, 1] / z + image_size[1] / 2.0
    return np.stack([u, v], axis=1)


def residuals(x, frame_ids, frame_data, image_size, *,
              lambda_f: float = LAMBDA_F, lambda_r: float = LAMBDA_R,
              smooth_max_gap: int = SMOOTH_MAX_GAP) -> np.ndarray:
    C, r_by, f_by = unpack_params(x, frame_ids)
    parts = []
    for i in frame_ids:
        world, uv = frame_data[i]
        parts.append((_project(world, r_by[i], f_by[i], C, image_size) - uv).ravel())
    for a, b in zip(frame_ids, frame_ids[1:]):
        if b - a > smooth_max_gap:
            continue
        parts.append(np.array([lambda_f * (f_by[b] - f_by[a])]))
        parts.append(lambda_r * (r_by[b] - r_by[a]))
    return np.concatenate(parts)


def jac_sparsity(frame_ids, frame_data, *, smooth_max_gap: int = SMOOTH_MAX_GAP):
    """Sparsity pattern aligned with `residuals`: each reprojection row touches
    C (cols 0-2) and its frame's (r, f) block; smoothness rows touch two blocks."""
    from scipy.sparse import lil_matrix
    col = {i: 3 + 4 * k for k, i in enumerate(frame_ids)}
    n_rows = sum(2 * len(frame_data[i][1]) for i in frame_ids)
    pairs = [(a, b) for a, b in zip(frame_ids, frame_ids[1:])
             if b - a <= smooth_max_gap]
    n_rows += sum(1 + 3 for _ in pairs)
    S = lil_matrix((n_rows, 3 + 4 * len(frame_ids)), dtype=np.uint8)
    row = 0
    for i in frame_ids:
        n = 2 * len(frame_data[i][1])
        S[row:row + n, 0:3] = 1
        S[row:row + n, col[i]:col[i] + 4] = 1
        row += n
    for a, b in pairs:
        S[row, col[a] + 3] = 1; S[row, col[b] + 3] = 1        # focal pair
        row += 1
        for d in range(3):                                     # rotation pair
            S[row, col[a] + d] = 1; S[row, col[b] + d] = 1
            row += 1
    return S
