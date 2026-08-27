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
