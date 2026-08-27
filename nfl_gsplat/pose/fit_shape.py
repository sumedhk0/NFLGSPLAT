"""Fit SMPL-X shape to a player's KNOWN stature.

Why this is needed at all
-------------------------
SMPLest-X regresses shape from the crop, and on broadcast All-22 the crops are
60-180 px tall. Measured on play_001: every one of 19 players came back with
betas ~0 and a stature of 1.62-1.63 m. The roster says those same 22 players are
1.70-1.98 m. That is not a loading bug -- the ``strict=False`` warning was
chased and cleared -- it is the model regressing to the mean because the pixels
do not carry the signal.

Height, though, is a published fact. So shape stops being something to estimate
per frame and becomes a per-player constant to look up, which is also what makes
it stable across frames: a player's body does not change between snaps, and
letting it wobble is a rendering artefact with no physical basis.

What this does and does not recover
-----------------------------------
Stature constrains the SKELETON -- bone lengths and overall scale. It says
nothing about girth: a lean and a heavy player of the same height have the same
joint positions. Weight is therefore taken as a second, weaker constraint, and
anything finer (limb thickness, musculature) needs image or multi-view evidence
rather than a scalar.

Only the first shape coefficient is solved. beta[0] is SMPL-X's dominant
size/scale direction, so one number is what a single scalar measurement can
honestly support; fitting all ten to one constraint would be free to move the
body in directions the measurement never observed.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Rest-pose stature is measured crown-to-sole on the template mesh. A player's
# listed height is measured the same way (barefoot), so no offset is applied --
# unlike the placement path, where the ANKLE JOINT sits ~8 cm above the turf.
_BETA0_BOUNDS = (-6.0, 6.0)


def template_stature(model, betas) -> float:
    """Rest-pose crown-to-sole height of the SMPL-X body for ``betas``."""
    import torch

    with torch.no_grad():
        out = model(betas=torch.as_tensor(
            np.asarray(betas, np.float32).reshape(1, -1)))
        verts = out.vertices.detach().cpu().numpy()[0]
    return float(verts[:, 1].max() - verts[:, 1].min())


def fit_beta0_to_height(model, target_height_m: float, *,
                        betas0=None, tol_m: float = 1e-3,
                        max_iter: int = 40) -> np.ndarray:
    """Solve beta[0] so the rest-pose stature matches ``target_height_m``.

    Bisection rather than gradient descent: stature is monotonic in beta[0] over
    the usable range, one dimensional, and bisection cannot overshoot into the
    non-physical shapes an unconstrained optimiser will happily find when handed
    a single scalar constraint.
    """
    if not 1.0 < float(target_height_m) < 2.6:
        raise SetupError(
            f"target height {target_height_m} m is not a plausible human "
            "stature -- check the units (roster heights are in INCHES).")

    betas = np.zeros(10, np.float32) if betas0 is None else np.asarray(
        betas0, np.float32).copy().reshape(-1)

    lo, hi = _BETA0_BOUNDS

    def stature_at(b0):
        trial = betas.copy()
        trial[0] = b0
        return template_stature(model, trial)

    s_lo, s_hi = stature_at(lo), stature_at(hi)
    if (target_height_m - s_lo) * (target_height_m - s_hi) > 0:
        raise SetupError(
            f"height {target_height_m:.2f} m is outside what beta[0] can reach "
            f"({min(s_lo, s_hi):.2f}-{max(s_lo, s_hi):.2f} m). Fitting more "
            "coefficients would be needed, and one scalar cannot constrain them.")

    ascending = s_hi > s_lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = stature_at(mid)
        if abs(s_mid - target_height_m) < tol_m:
            betas[0] = mid
            return betas
        if (s_mid < target_height_m) == ascending:
            lo = mid
        else:
            hi = mid
    betas[0] = 0.5 * (lo + hi)
    _LOG.warning("beta0 fit did not converge to %.3f m in %d iterations; "
                 "best stature %.3f m", target_height_m, max_iter,
                 stature_at(betas[0]))
    return betas
