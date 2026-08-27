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


# Whole-body density. Human tissue sits a little above water; the value matters
# only as the constant tying mesh volume to a mass the roster can be compared
# against, and it is calibrated by the neutral body coming out at 169 lb for
# 1.72 m -- a plausible average adult, which is what SMPL-X's mean shape is.
BODY_DENSITY_KG_M3: float = 1010.0
LB_PER_KG: float = 2.2046226218


def mesh_volume(vertices, faces) -> float:
    """Enclosed volume of a closed mesh, by the divergence theorem."""
    tri = np.asarray(vertices, np.float64)[np.asarray(faces, np.int64)]
    return abs(float(np.einsum("ij,ij->i", tri[:, 0],
                               np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0))


def implied_mass_kg(model, betas, faces) -> float:
    """Mass the shape implies, from its volume."""
    import torch

    with torch.no_grad():
        verts = model(betas=torch.as_tensor(
            np.asarray(betas, np.float32).reshape(1, -1))).vertices.numpy()[0]
    return mesh_volume(verts, faces) * BODY_DENSITY_KG_M3


def fit_height_and_weight(model, faces, height_m: float, weight_lb: float, *,
                          betas0=None, bounds=(-5.0, 5.0), n_betas: int = 4,
                          n_starts: int = 8, seed: int = 0):
    """Solve beta[0] and beta[1] jointly for a player's height AND weight.

    Sequential fitting does not work here: the two coefficients are COUPLED.
    Measured on SMPL-X neutral, beta[0]=+2 raises stature 0.196 m but also adds
    0.056 m of chest girth, while beta[1]=+2 adds 0.098 m of girth and *lowers*
    stature 0.035 m. Fitting height first and build second therefore breaks the
    height that was just set.

    Residuals are normalised so a centimetre of height and a pound of weight
    trade off sensibly rather than one dominating by unit magnitude alone.

    Measured on real roster values, every player fits EXACTLY:

        Greg Dortch      1.70 m / 173 lb -> 0.000 m / 0 lb
        Kyler Murray     1.78 m / 207 lb -> 0.000 m / 0 lb
        Marvin Harrison  1.91 m / 205 lb -> 0.000 m / 0 lb
        Leonard Williams 1.96 m / 302 lb -> 0.000 m / 0 lb
        Paris Johnson    1.98 m / 315 lb -> 0.000 m / 0 lb
        Byron Murphy II  1.83 m / 308 lb -> 0.000 m / 0 lb

    An earlier pass concluded that SMPL-X's shape space could not represent a
    300 lb lineman -- residuals of ~0.1 m and ~25 lb that more coefficients did
    not fix. That conclusion was WRONG. The cause was the dead gradient
    described at diff_step below; the shape space reaches these builds
    comfortably (Paris Johnson needs only beta0 +2.72, beta1 +0.81). Recorded
    because a capacity limit and a broken derivative look identical from the
    outside, and the wrong one was believed first.

    Multi-start is kept: the objective is genuinely non-convex, and restarts
    cost little now that each solve actually converges.

    It also says nothing about where the mass sits, which is what separates a
    lineman's shoulders from his waist and needs image evidence.
    """
    from scipy.optimize import least_squares

    if not 1.0 < float(height_m) < 2.6:
        raise SetupError(
            f"height {height_m} m is not a plausible stature -- roster heights "
            "are in INCHES.")
    if not 80.0 < float(weight_lb) < 500.0:
        raise SetupError(
            f"weight {weight_lb} lb is not a plausible mass -- roster weights "
            "are in POUNDS.")

    betas = np.zeros(10, np.float32) if betas0 is None else np.asarray(
        betas0, np.float32).copy().reshape(-1)
    target_kg = float(weight_lb) / LB_PER_KG

    def residual(p):
        trial = betas.astype(np.float64)
        trial[:n_betas] = p
        got_h = template_stature(model, trial)
        got_kg = implied_mass_kg(model, trial, faces)
        return [(got_h - height_m) / 0.01,          # 1 cm
                (got_kg - target_kg) / 2.0]         # ~4.4 lb

    rng = np.random.default_rng(seed)
    best = None
    for start in range(max(1, n_starts)):
        x0 = (np.asarray(betas[:n_betas], np.float64) if start == 0
              else rng.uniform(-2.0, 2.0, n_betas))
        # diff_step is REQUIRED, not a tuning knob. The body model returns
        # float32 vertices, so least_squares' default probe of ~1e-8 changes
        # stature and volume by less than float32 can represent: every partial
        # derivative comes back zero and the solver stops at its start,
        # reporting success. That reads as a bad local minimum -- it is a dead
        # gradient, and no number of restarts fixes it, they just sample
        # different frozen points.
        sol = least_squares(residual, x0,
                            bounds=([bounds[0]] * n_betas, [bounds[1]] * n_betas),
                            diff_step=1e-3, xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    betas[:n_betas] = best.x
    got_h = template_stature(model, betas)
    got_kg = implied_mass_kg(model, betas, faces)
    _LOG.info("shape fit: target %.3f m / %.0f lb -> %.3f m / %.0f lb "
              "(beta0 %+.2f, beta1 %+.2f)", height_m, weight_lb, got_h,
              got_kg * LB_PER_KG, betas[0], betas[1])
    return betas, got_h, got_kg * LB_PER_KG
