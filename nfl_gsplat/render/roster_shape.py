"""The avatar stands as tall as the roster says.

WHY. The regressor's shape coefficients sit near neutral, and SMPL-X
neutral is 1.72 m; the roster median for these players is 1.85 m. Every
avatar was 8 % short (measured 2026-09-04: refit betas gave 1.70-1.72 m
for ids whose roster heights median 1.85). Roster height is known to the
inch and constrains nothing upstream, so it is the one shape fact worth
imposing.

WHAT. SMPL-X's first shape coefficient sets stature almost linearly,
9.8 cm per unit (measured on the neutral model from -2 to +3). Given a
player's betas and roster height, ``betas_for_height`` moves the first
coefficient so the model's crown-to-heel stature meets the height, in
two Newton steps, and leaves the other coefficients as they are.
"""
from __future__ import annotations

import numpy as np

STATURE_PER_BETA0_M: float = 0.098
TOL_M: float = 0.005


def stature_of(model, betas) -> float:
    """Crown-to-heel height of the neutral-pose model under ``betas``, metres."""
    import torch

    b = np.zeros(model.num_betas, np.float32)
    bb = np.asarray(betas, np.float32).reshape(-1)
    b[: min(len(bb), model.num_betas)] = bb[: model.num_betas]
    with torch.no_grad():
        v = model(betas=torch.tensor(b[None])).vertices[0].cpu().numpy()
    return float(v[:, 1].max() - v[:, 1].min())


def betas_for_height(model, betas, height_m: float, *, steps: int = 2) -> np.ndarray:
    """``betas`` with the first coefficient adjusted so the model stands
    ``height_m`` tall; the rest untouched."""
    out = np.array(betas, np.float64, copy=True).reshape(-1)
    for _ in range(steps):
        gap = float(height_m) - stature_of(model, out)
        if abs(gap) <= TOL_M:
            break
        out[0] += gap / STATURE_PER_BETA0_M
    return out


def heights_from_identity(merged) -> dict[int, float]:
    """``{pid: roster height in metres}`` for the ids identity gave a height."""
    out = {}
    for pid, p in merged.items():
        h = getattr(p, "height_m", None)
        if h and 1.4 < float(h) < 2.3:
            out[int(pid)] = float(h)
    return out
