"""Register one frame: identified correspondences → per-frame (K, R, t)."""
from __future__ import annotations

from nfl_gsplat.calibration.field_identify import IdentityState, identify_correspondences
from nfl_gsplat.calibration.solve_pnp import CalibrationResult, solve_pnp_from_correspondences
from nfl_gsplat.errors import CalibrationError


def register_frame(
    feats, prior, image_size: tuple[int, int],
    *, max_reproj_px: float = 6.0, min_landmarks: int = 6,
) -> "tuple[CalibrationResult | None, IdentityState]":
    """Return (CalibrationResult|None, IdentityState). None when registration
    fails (too few correspondences or RMS over tolerance) — that frame is a gap."""
    corrs, state = identify_correspondences(feats, prior)
    if len(corrs) < min_landmarks:
        return None, state
    try:
        res = solve_pnp_from_correspondences(
            corrs, image_size=image_size, max_reproj_px=max_reproj_px,
            min_landmarks=min_landmarks,
        )
    except CalibrationError:
        return None, state
    return res, state
