"""Tests for fitting SMPL-X shape to a known stature.

Uses a stand-in body model rather than the real SMPL-X files: those are
licence-gated and 100 MB+, so a test that needed them would be unrunnable on any
machine that had not completed a manual registration.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.pose.fit_shape import fit_beta0_to_height, template_stature


class _FakeModel:
    """Stature rises monotonically with beta[0], as it does in SMPL-X."""

    def __init__(self, base=1.72, gain=0.10):
        self.base, self.gain = base, gain

    def __call__(self, betas=None, **_):
        import torch

        b0 = float(np.asarray(betas).reshape(-1)[0])
        height = self.base + self.gain * b0
        verts = torch.tensor([[[0.0, -height / 2, 0.0],
                               [0.0, height / 2, 0.0]]], dtype=torch.float32)
        return type("Out", (), {"vertices": verts})()


def test_recovers_a_range_of_real_statures():
    """Roster heights for one play span 1.70-1.98 m; all must be reachable."""
    model = _FakeModel()
    for target in (1.702, 1.778, 1.905, 1.981):
        betas = fit_beta0_to_height(model, target)
        assert template_stature(model, betas) == pytest.approx(target, abs=2e-3)


def test_only_the_first_coefficient_moves():
    """One scalar measurement cannot honestly constrain ten coefficients; the
    rest must be left exactly where they came in."""
    model = _FakeModel()
    start = np.arange(10, dtype=np.float32) * 0.1
    got = fit_beta0_to_height(model, 1.90, betas0=start)
    assert not np.allclose(got[0], start[0])
    assert np.allclose(got[1:], start[1:])


def test_inches_mistaken_for_metres_is_rejected():
    """Roster heights are in INCHES. Passing 75 straight through would silently
    fit a seventy-five-metre body."""
    with pytest.raises(SetupError, match="INCHES"):
        fit_beta0_to_height(_FakeModel(), 75.0)


def test_unreachable_height_fails_loud():
    """Better to say beta[0] cannot get there than to return its best effort and
    let a visibly wrong body through."""
    model = _FakeModel(base=1.72, gain=0.001)     # barely responds to beta0
    with pytest.raises(SetupError, match="outside what"):
        fit_beta0_to_height(model, 2.40)


def test_fit_is_deterministic():
    model = _FakeModel()
    a = fit_beta0_to_height(model, 1.88)
    b = fit_beta0_to_height(model, 1.88)
    assert np.array_equal(a, b)


def test_height_and_weight_fit_uses_multiple_starts():
    """The objective is non-convex. A single solve from zeros lands in wildly
    different local minima depending only on how many coefficients are free --
    measured on real players, 10 betas gave a WORSE height than 6, and one swung
    from +0.151 m to -0.209 m between dimensionalities.
    """
    import numpy as np

    from nfl_gsplat.pose.fit_shape import fit_height_and_weight

    class _Model:
        """Height from beta0, volume from beta0 and beta1 -- coupled, as in
        SMPL-X, so the two cannot be fitted one after the other."""

        def __call__(self, betas=None, **_):
            import torch

            b = np.asarray(betas).reshape(-1)
            h = 1.72 + 0.10 * b[0]
            r = 0.16 + 0.03 * b[1] + 0.01 * b[0]
            # a box whose volume the divergence-theorem helper can integrate
            xs, zs = r, r
            corners = np.array([[-xs, -h / 2, -zs], [xs, -h / 2, -zs],
                                [xs, -h / 2, zs], [-xs, -h / 2, zs],
                                [-xs, h / 2, -zs], [xs, h / 2, -zs],
                                [xs, h / 2, zs], [-xs, h / 2, zs]], np.float32)
            return type("O", (), {"vertices": torch.tensor(corners[None])})()

    # A CLOSED box: all six faces, two triangles each. An open or
    # self-overlapping surface makes the divergence-theorem volume meaningless,
    # and the fit then chases a target it can never reach.
    faces = np.array([
        [0, 1, 2], [0, 2, 3],      # y = -h/2
        [4, 6, 5], [4, 7, 6],      # y = +h/2
        [3, 2, 6], [3, 6, 7],      # z = +r
        [0, 5, 1], [0, 4, 5],      # z = -r
        [0, 3, 7], [0, 7, 4],      # x = -r
        [1, 5, 6], [1, 6, 2],      # x = +r
    ])
    betas, got_h, got_lb = fit_height_and_weight(
        _Model(), faces, 1.90, 240.0, n_betas=2, n_starts=6)
    assert got_h == pytest.approx(1.90, abs=0.02)
    assert np.allclose(betas[2:], 0.0), "coefficients beyond n_betas moved"


def test_weight_in_kilograms_is_rejected():
    """Roster weights are POUNDS. Passing 95 (kg) would fit a child."""
    from nfl_gsplat.pose.fit_shape import fit_height_and_weight

    with pytest.raises(SetupError, match="POUNDS"):
        fit_height_and_weight(_FakeModel(), np.zeros((1, 3), int), 1.90, 45.0)
