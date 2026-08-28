

# --- offline sequences need ZERO-PHASE smoothing ------------------------------
# The 1-euro filter is CAUSAL: it is built for real-time input, so on a recorded
# sequence it lags whatever it follows by an amount that grows with speed. This
# project has already paid for that once -- the same filter applied to camera
# poses moved the endzone reference frame's own camera, exact by construction,
# from 2.72 px of yard-line error to 99.69 px, purely as pan lag. A play's pose
# track is offline too, and a lagging body lands behind where the player was.

import numpy as np

from nfl_gsplat.pose.temporal_smooth import (OneEuroConfig, smooth_param_sequence,
                                             smooth_param_sequence_zero_phase)


def _ramp(n=120, speed=0.05, noise=0.004, seed=0):
    """A steadily moving signal -- the case where causal lag is worst."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    clean = speed * t
    return clean, clean + rng.normal(0.0, noise, n)


def test_causal_filter_lags_a_moving_signal():
    """Establishes the problem the zero-phase version exists to solve."""
    clean, noisy = _ramp()
    cfg = OneEuroConfig(min_cutoff=0.5, beta=0.0, fps=30.0)
    causal = smooth_param_sequence(noisy[:, None], cfg)[:, 0]
    # Well past the filter's warm-up, the causal output trails the truth.
    lag = float(np.mean(clean[40:] - causal[40:]))
    assert lag > 0.01, f"expected the causal filter to trail, got {lag:+.4f}"


def test_zero_phase_removes_the_lag():
    clean, noisy = _ramp()
    cfg = OneEuroConfig(min_cutoff=0.5, beta=0.0, fps=30.0)
    causal = smooth_param_sequence(noisy[:, None], cfg)[:, 0]
    zero = smooth_param_sequence_zero_phase(noisy[:, None], cfg)[:, 0]
    causal_bias = abs(float(np.mean(clean[40:-40] - causal[40:-40])))
    zero_bias = abs(float(np.mean(clean[40:-40] - zero[40:-40])))
    assert zero_bias < causal_bias / 5.0, (
        f"zero-phase bias {zero_bias:.5f} should be far below causal "
        f"{causal_bias:.5f}")


def test_zero_phase_still_removes_noise():
    """Removing lag must not mean removing the smoothing."""
    clean, noisy = _ramp()
    cfg = OneEuroConfig(min_cutoff=0.5, beta=0.0, fps=30.0)
    zero = smooth_param_sequence_zero_phase(noisy[:, None], cfg)[:, 0]
    raw_err = float(np.sqrt(np.mean((noisy[40:-40] - clean[40:-40]) ** 2)))
    out_err = float(np.sqrt(np.mean((zero[40:-40] - clean[40:-40]) ** 2)))
    assert out_err < raw_err


def test_zero_phase_leaves_a_constant_untouched():
    cfg = OneEuroConfig(min_cutoff=0.5, beta=0.0, fps=30.0)
    const = np.full((60, 3), 0.42)
    out = smooth_param_sequence_zero_phase(const, cfg)
    assert np.allclose(out, 0.42, atol=1e-9)


def test_zero_phase_carries_nans_through():
    """Gap interpolation stays upstream of smoothing, as for the causal path."""
    cfg = OneEuroConfig(fps=30.0)
    x = np.arange(30, dtype=float)[:, None]
    x[10] = np.nan
    out = smooth_param_sequence_zero_phase(x, cfg)
    assert np.isnan(out[10, 0])
    assert np.isfinite(out[[0, 5, 20, 29], 0]).all()
