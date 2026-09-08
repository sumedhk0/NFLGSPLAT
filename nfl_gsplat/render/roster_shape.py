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
# Weight (2026-09-08): the second coefficient sets girth almost without
# touching stature -- +-2 = +-27 % of the neutral mesh volume, +-3.5 cm of
# height -- and the neutral body (1.72 m, 0.0760 m^3) weighs 77 kg at
# DENSITY_KG_M3, a BMI of 26: so a roster weight is a target mesh volume.
# Without it every avatar had the regressor's near-neutral girth and a
# 140 kg tackle looked like a 85 kg corner (the user, v22).
DENSITY_KG_M3: float = 1010.0
TOL_VOLUME: float = 0.02          # relative
BETA1_RANGE = (-3.0, 5.0)


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


def volume_of(model, betas) -> float:
    """Mesh volume of the neutral-pose model under ``betas``, cubic metres."""
    import torch

    b = np.zeros(model.num_betas, np.float32)
    bb = np.asarray(betas, np.float32).reshape(-1)
    b[: min(len(bb), model.num_betas)] = bb[: model.num_betas]
    with torch.no_grad():
        v = model(betas=torch.tensor(b[None])).vertices[0].cpu().numpy().astype(np.float64)
    tri = v[np.asarray(model.faces, np.int64)]
    return float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


def betas_for_height_weight(model, betas, height_m: float, mass_kg: float, *, steps: int = 3) -> np.ndarray:
    """``betas`` with the first coefficient set for ``height_m`` and the second
    for ``mass_kg`` (mesh volume at DENSITY_KG_M3), alternating Newton steps;
    the rest untouched."""
    out = np.array(betas, np.float64, copy=True).reshape(-1)
    if len(out) < 2:
        out = np.concatenate([out, np.zeros(2 - len(out))])
    target = float(mass_kg) / DENSITY_KG_M3
    for _ in range(steps):
        out = betas_for_height(model, out, height_m, steps=2)
        v = volume_of(model, out)
        if abs(v - target) <= TOL_VOLUME * target:
            break
        v2 = volume_of(model, out + np.eye(len(out))[1] * 0.5)
        slope = (v2 - v) / 0.5
        if slope <= 1e-6:
            break
        out[1] = float(np.clip(out[1] + (target - v) / slope, *BETA1_RANGE))
    return betas_for_height(model, out, height_m, steps=2)


def weights_from_identity(merged) -> dict[int, float]:
    """``{pid: roster weight in kg}`` for the ids identity gave a weight."""
    out = {}
    for pid, p in merged.items():
        w = getattr(p, "weight_lb", None)
        if w and 60.0 < float(w) * 0.4536 < 200.0:
            out[int(pid)] = float(w) * 0.4536
    return out


def heights_from_identity(merged) -> dict[int, float]:
    """``{pid: roster height in metres}`` for the ids identity gave a height."""
    out = {}
    for pid, p in merged.items():
        h = getattr(p, "height_m", None)
        if h and 1.4 < float(h) < 2.3:
            out[int(pid)] = float(h)
    return out
