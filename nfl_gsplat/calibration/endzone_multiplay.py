"""Multi-play, shared-center endzone solve from identity correspondences.

The endzone camera is a fixed tripod: its center C is identical across all
first-half plays. We concatenate every play's frames into one frame set (offset
so indices never collide), share one C, and let each frame keep its own
rotation+focal -- exactly solve_fixed_center's model. The EndzonePrior bounds C
to the correct side so the Y-reflection is unreachable. Native pixels, view_deg=0."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.errors import CalibrationError


@dataclass(frozen=True)
class EndzonePrior:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    focal_range: tuple[float, float]

    @property
    def center_bounds(self):
        return (self.x_range, self.y_range, self.z_range)

    @property
    def center0(self) -> np.ndarray:
        return np.array([sum(self.x_range) / 2, sum(self.y_range) / 2,
                         sum(self.z_range) / 2], float)


def solve_endzone_identity(corrs_by_play, image_size, prior, *,
                           audit_drop_px=15.0, min_frames=10):
    """corrs_by_play[p] = {frame: (world(N,3), uv(N,2))}. Returns one
    [CalibrationResult|None] list per play, aligned to that play's frames."""
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    # Union frame_data with dense global indices; index_map recovers (play, local frame).
    frame_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    index_map: list[tuple[int, int]] = []   # global idx -> (play, local frame)
    lengths = []
    for p, corrs in enumerate(corrs_by_play):
        maxf = -1
        for fr in sorted(corrs):
            w, uv = corrs[fr]
            if len(w) >= 4:
                frame_data[len(index_map)] = (np.asarray(w, np.float64),
                                              np.asarray(uv, np.float64))
                index_map.append((p, int(fr)))
            maxf = max(maxf, int(fr))
        lengths.append(maxf + 1)
    if len(frame_data) < min_frames:
        raise CalibrationError(
            f"endzone identity solve: only {len(frame_data)} usable frames across "
            f"{len(corrs_by_play)} plays (need >= {min_frames}) — thin jersey OCR or "
            "too few plays; re-run identity precompute or add plays.")

    T = len(index_map)
    # Anchor solve_fixed_center's multi-start at the prior box center.
    anchor = CalibrationResult(
        intrinsics=CameraIntrinsics(sum(prior.focal_range) / 2, sum(prior.focal_range) / 2,
                                    0.0, 0.0, 1, 1),
        pose=CameraPose(R=np.eye(3), t=-prior.center0),
        rms_px=0.0, num_correspondences=0, refined_with_ba=False)
    init_results = [anchor if i < 3 else None for i in range(T)]

    results, _mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=image_size, init_results=init_results,
        _frame_data_override=frame_data, view_deg=0,
        center_bounds=prior.center_bounds, audit_drop_px=audit_drop_px)

    # Split the union results back per play using the dense index map.
    per_play = [[None] * L for L in lengths]
    for gidx, (p, fr) in enumerate(index_map):
        if gidx < len(results) and results[gidx] is not None:
            per_play[p][fr] = results[gidx]
    return per_play
