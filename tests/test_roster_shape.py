"""render.roster_shape: the avatar stands as tall as the roster says."""
import numpy as np
import pytest

from nfl_gsplat.render import roster_shape as rs


class _Model:
    """A stand-in with SMPL-X's measured stature law: 1.721 m + 0.098 m per beta0."""
    num_betas = 10

    def __call__(self, betas):
        import torch

        b0 = float(betas[0, 0])
        h = 1.721 + rs.STATURE_PER_BETA0_M * b0
        v = torch.zeros(2, 3)
        v[1, 1] = h
        return type("Out", (), {"vertices": v[None]})()


def test_betas_for_height_meets_the_roster_height():
    pytest.importorskip("torch")
    m = _Model()
    b = rs.betas_for_height(m, np.zeros(10), 1.85)
    assert abs(rs.stature_of(m, b) - 1.85) < rs.TOL_M
    assert b[0] > 1.0 and np.allclose(b[1:], 0.0)                   # only the first coefficient moves
    short = rs.betas_for_height(m, np.array([0.5] + [0.1] * 9), 1.60)
    assert abs(rs.stature_of(m, short) - 1.60) < rs.TOL_M and np.allclose(short[1:], 0.1)


def test_heights_from_identity_filters_junk():
    class P:
        def __init__(self, h):
            self.height_m = h
    merged = {1: P(1.85), 2: P(None), 3: P(0.0), 4: P(9.9), 5: P(1.70)}
    assert rs.heights_from_identity(merged) == {1: 1.85, 5: 1.70}
