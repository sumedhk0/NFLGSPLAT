"""keypoint_filter: a one-frame wrist jump is rejected, a real stride is not."""
import numpy as np
import pandas as pd

from nfl_gsplat.pose.keypoint_filter import reject_outliers


def _track(joint, xs, ys, conf=0.9, cam="sideline", pid=1):
    return [{"cam": cam, "global_player_id": pid, "joint": joint, "frame": f, "x": x, "y": y, "conf": conf}
            for f, (x, y) in enumerate(zip(xs, ys))]


def test_a_spike_is_rejected_and_a_stride_kept():
    n = 20
    wrist_x = np.linspace(100, 140, n)                    # a wrist moving 2 px a frame
    wrist_y = np.full(n, 300.0)
    wrist_x[10] += 45.0                                   # one frame 45 px off: a left/right swap
    rows = _track(9, wrist_x, wrist_y)
    stride_x = np.linspace(100, 300, n)                   # a fast foot: 10.5 px a frame, no spike
    rows += _track(15, stride_x, np.full(n, 400.0))
    df = pd.DataFrame(rows)
    out, n_rej = reject_outliers(df)
    assert n_rej == 1
    bad = out[(out.joint == 9) & (out.frame == 10)]
    assert float(bad.conf.iloc[0]) == 0.0
    assert (out[out.joint == 15].conf > 0).all()          # the stride is untouched
    assert (out[(out.joint == 9) & (out.frame != 10)].conf > 0).all()


def test_low_confidence_points_are_neither_used_nor_judged():
    n = 12
    rows = _track(10, np.full(n, 50.0), np.full(n, 50.0), conf=0.2)
    df = pd.DataFrame(rows)
    out, n_rej = reject_outliers(df)
    assert n_rej == 0 and np.allclose(out.conf, 0.2)


def test_left_right_flip_is_swapped_back_and_a_crossing_is_not():
    from nfl_gsplat.pose.keypoint_filter import fix_lr_flips

    n = 30
    f = np.arange(n)
    # legs crossing: the left foot moves right, the right foot moves left, through each other
    lx = 100.0 + 6.0 * f
    rx = 250.0 - 6.0 * f
    rows = _track(15, lx, np.full(n, 500.0)) + _track(16, rx, np.full(n, 500.0))
    rows += _track(13, lx, np.full(n, 420.0)) + _track(14, rx, np.full(n, 420.0))
    out, n_sw = fix_lr_flips(pd.DataFrame(rows))
    assert n_sw == 0
    # arms: a steady pose, then the labels flipped for frames 12-14
    ax = np.full(n, 300.0); bx = np.full(n, 380.0)
    rows = []
    for j_l, j_r, y in ((5, 6, 200.0), (7, 8, 260.0), (9, 10, 320.0)):
        xl, xr = ax.copy(), bx.copy()
        xl[12:15], xr[12:15] = bx[12:15], ax[12:15]
        rows += _track(j_l, xl, np.full(n, y)) + _track(j_r, xr, np.full(n, y))
    df = pd.DataFrame(rows)
    out, n_sw = fix_lr_flips(df)
    assert n_sw == 3
    fixed = out[(out.joint == 9)].sort_values("frame").x.to_numpy()
    assert np.allclose(fixed, 300.0)
    fixed_r = out[(out.joint == 10)].sort_values("frame").x.to_numpy()
    assert np.allclose(fixed_r, 380.0)
