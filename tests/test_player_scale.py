"""Players as a scale reference, where the paint has none."""
import numpy as np

from nfl_gsplat.calibration.player_scale import (
    EXPECTED_PLAYER_M,
    height_score,
    implied_heights,
)

W, H = 1920, 1080


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


def K_of(f):
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def project(K, R, t, pts):
    q = (np.asarray(pts, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


def scene(centre, focal, n=14, seed=0, height_m=EXPECTED_PLAYER_M):
    """Person boxes as a camera at ``centre`` would see standing players."""
    rng = np.random.default_rng(seed)
    K = K_of(focal)
    R, t = look_at(centre)
    ground = rng.uniform([-25, -18], [25, 18], size=(n, 2))
    feet = project(K, R, t, np.c_[ground, np.zeros(n)])
    heads = project(K, R, t, np.c_[ground, np.full(n, height_m)])
    boxes = np.column_stack([feet[:, 0] - 20, heads[:, 1],
                             feet[:, 0] + 20, feet[:, 1]])
    return K, R, t, boxes


def test_the_right_camera_recovers_real_player_height():
    K, R, t, boxes = scene(np.array([20.0, -90.0, 40.0]), 3000.0)
    h = implied_heights(K, R, t, boxes)
    assert len(h) >= 10
    assert abs(float(np.median(h)) - EXPECTED_PLAYER_M) < 0.05


def test_a_camera_at_the_wrong_scale_makes_everyone_the_wrong_size():
    """The check that the paint cannot make: how BIG is everything.

    A camera further away behind a longer lens reproduces the same lines and
    the same image, and gets the players wrong.
    """
    truth_centre = np.array([20.0, -90.0, 40.0])
    K, R, t, boxes = scene(truth_centre, 3000.0)
    good = height_score(K, R, t, boxes)

    # Same view, twice as far away with twice the focal length.
    far = truth_centre * 2.0
    K2 = K_of(6000.0)
    R2, t2 = look_at(far)
    bad = height_score(K2, R2, t2, boxes)

    assert good[0] < bad[0]
    assert abs(good[2] - EXPECTED_PLAYER_M) < 0.05


def test_it_declines_to_speak_on_too_few_detections():
    K, R, t, boxes = scene(np.array([20.0, -90.0, 40.0]), 3000.0, n=3)
    cost, used, _median = height_score(K, R, t, boxes)
    assert not np.isfinite(cost)
    assert used < 6
