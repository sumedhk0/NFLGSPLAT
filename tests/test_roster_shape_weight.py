"""roster_shape: height and weight both met (needs the SMPL-X model; skipped without it)."""
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(not Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists(), reason="SMPL-X model absent")
def test_a_tackle_and_a_corner_get_their_builds():
    smplx = pytest.importorskip("smplx")
    from nfl_gsplat.render.roster_shape import (DENSITY_KG_M3, betas_for_height_weight, stature_of, volume_of,
                                                weights_from_identity)

    model = smplx.create("data/body_models", model_type="smplx", gender="neutral", num_betas=10, use_pca=False,
                         batch_size=1)
    for h, kg in ((1.96, 145.0), (1.80, 86.0)):
        b = betas_for_height_weight(model, np.zeros(10), h, kg)
        assert abs(stature_of(model, b) - h) < 0.01
        assert abs(volume_of(model, b) * DENSITY_KG_M3 - kg) < 0.03 * kg

    class P:
        weight_lb = 320.0
    assert abs(weights_from_identity({1: P()})[1] - 145.2) < 0.2
