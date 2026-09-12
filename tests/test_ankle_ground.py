"""render.play_timeline.ankle_ground: the ankle keypoints' rays meet the turf at ankle height where the
feet are; ground_positions takes that over the box point where a camera has it."""
import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.render.play_timeline import ankle_ground, ground_positions


def look_at(centre, target):
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    fwd = target - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def track(centre, target, f, T=3):
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    R = look_at(centre, target)
    t = -R @ np.asarray(centre, float)
    return CameraTrack(K=np.repeat(K[None], T, 0), R=np.repeat(R[None], T, 0), t=np.repeat(t[None], T, 0),
                       conf=np.ones(T), width=1920, height=1080)


def project(tr, f, X):
    p = (tr.K[f] @ (tr.R[f] @ np.asarray(X, float).T + tr.t[f][:, None])).T
    return p[:, :2] / p[:, 2:3]


def test_ankle_ground_lands_between_the_feet():
    tr = track((-3.7, -101.6, 42.5), (-20.0, 0.0, 0.0), 9400.0)
    feet = np.array([[-20.0, 1.0, 0.08], [-20.4, 1.2, 0.08]])
    uv = project(tr, 1, feet)
    kdf = pd.DataFrame([{"frame": 1, "cam": "sideline", "global_player_id": 7, "joint": j, "x": u, "y": v, "conf": 0.9}
                        for (u, v), j in zip(uv, (15, 16))]
                       + [{"frame": 1, "cam": "sideline", "global_player_id": 8, "joint": 15, "x": 5.0, "y": 5.0, "conf": 0.2}])
    g = ankle_ground(kdf, {"sideline": tr})
    assert set(g) == {("sideline", 1, 7)}
    assert np.allclose(g[("sideline", 1, 7)], feet[:, :2].mean(axis=0), atol=1e-3)


def test_ground_positions_prefers_the_ankles_over_the_box():
    tr = track((-3.7, -101.6, 42.5), (-20.0, 0.0, 0.0), 9400.0)
    # both ids, as the real table carries them: ground_positions keys by the PLAYER, and ankle_ground
    # keys its dict the same way, so a fixture with only the tracker's id tests the wrong contract
    df = pd.DataFrame([{"frame": 1, "cam": "sideline", "track_id": 7, "global_player_id": 7,
                        "bbox_x1": 900.0, "bbox_x2": 940.0, "bbox_y1": 400.0, "bbox_y2": 560.0},
                       {"frame": 1, "cam": "sideline", "track_id": 8, "global_player_id": 8,
                        "bbox_x1": 1100.0, "bbox_x2": 1140.0, "bbox_y1": 400.0, "bbox_y2": 560.0}])
    box_only = ground_positions(df, {"sideline": tr})
    ankles = {("sideline", 1, 7): np.array([-20.2, 1.1])}
    mixed = ground_positions(df, {"sideline": tr}, ankles=ankles)
    assert np.allclose(mixed[1][7], [-20.2, 1.1])
    assert np.allclose(mixed[1][8], box_only[1][8])
