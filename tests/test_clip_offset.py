"""pose.clip_offset: the offset is where the moving players' rays meet; a monotone curve is refused."""
import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.pose.clip_offset import ground_speeds, miss_by_offset, solve_offset


def look_at(centre, target):
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def track(centre, target, f, T):
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    R = look_at(centre, target)
    t = -R @ np.asarray(centre, float)
    return CameraTrack(K=np.repeat(K[None], T, 0), R=np.repeat(R[None], T, 0), t=np.repeat(t[None], T, 0),
                       conf=np.ones(T), width=1920, height=1080)


def project(tr, f, X):
    p = (tr.K[f] @ (tr.R[f] @ np.asarray(X, float).T + tr.t[f][:, None])).T
    return p[:, :2] / p[:, 2:3]


def test_offset_recovered_from_a_runner():
    T, fps, true_off = 120, 60.0, -7
    ta = track((-3.7, -101.6, 42.5), (-20.0, 0.0, 0.0), 9400.0, T)
    tb = track((88.0, 0.6, 21.0), (-20.0, 0.0, 0.0), 15000.0, T)
    rows, boxes = [], []
    for f in range(T):
        x, y = -30.0 + 0.12 * f, -6.0 + 0.05 * f              # 7.8 m/s along the field
        X = np.array([[x, y - 0.2, 0.95], [x, y + 0.2, 0.95], [x, y - 0.2, 1.45], [x, y + 0.2, 1.45]])
        for cam, tr, fo in (("sideline", ta, 0), ("endzone", tb, true_off)):
            fe = f + fo                                        # the endzone clip shows time f at frame f + off
            if not 0 <= fe < T:
                continue
            uv = project(tr, fe, X)
            for (u, v), j in zip(uv, (11, 12, 5, 6)):
                rows.append({"frame": fe, "cam": cam, "global_player_id": 1, "joint": j, "x": u, "y": v, "conf": 0.9})
        foot = project(ta, f, [[x, y, 0.0]])[0]
        boxes.append({"frame": f, "cam": "sideline", "track_id": 1, "bbox_x1": foot[0] - 20, "bbox_x2": foot[0] + 20,
                      "bbox_y1": foot[1] - 130, "bbox_y2": foot[1]})
    kdf, tdf = pd.DataFrame(rows), pd.DataFrame(boxes)
    moving = ground_speeds(tdf, ta, fps=fps, speed_min=3.0)
    assert len(moving) > 50
    curve = miss_by_offset(kdf, ta, tb, moving, range(-15, 6))
    assert solve_offset(curve, min_pairs=20) == true_off


def test_monotone_curve_is_refused():
    curve = {o: (0.5 - 0.01 * o, 100) for o in range(-10, 11)}
    with pytest.raises(CalibrationError):
        solve_offset(curve)
