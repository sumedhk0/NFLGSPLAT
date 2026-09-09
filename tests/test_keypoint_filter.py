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


def test_a_running_ankle_swing_is_kept_and_its_spike_rejected():
    n = 60
    f = np.arange(n)
    x = 400.0 + 12.0 * f + 80.0 * np.sin(2 * np.pi * f / 20.0)      # a sprinter's ankle: 12 px/frame plus an 80 px swing
    y = 700.0 + 15.0 * np.cos(2 * np.pi * f / 20.0)
    x[33] += 45.0                                                    # one frame off by a shoe's width
    rows = _track(15, x, y)
    out, n_rej = reject_outliers(pd.DataFrame(rows))
    assert n_rej == 1
    assert float(out[out.frame == 33].conf.iloc[0]) == 0.0
