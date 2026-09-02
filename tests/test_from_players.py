"""The players must be able to calibrate the view the paint cannot."""
import numpy as np
import pytest

from nfl_gsplat.calibration.from_players import (
    fit_from_seed,
    seed_homography,
    solve_second_view,
)
from nfl_gsplat.errors import CalibrationError

W, H = 1920, 1080


def look_at(centre, target=(0.0, 0.0, 0.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


def cam(centre, f):
    R, t = look_at(centre)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R, t


def project(c, pts):
    K, R, t = c
    q = (np.asarray(pts, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


# A sideline camera close to the one three independent samples agreed on, and an
# endzone camera behind the end zone where a real one sits.
SIDE = cam([30.0, -95.0, 47.0], 3000.0)
ENDZ = cam([-75.0, 0.0, 25.0], 1500.0)


def play(n_frames=6, n_players=20, seed=0):
    """Players milling about on the turf, seen by both cameras."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([-25, -18], [25, 18], size=(n_players, 2))
    feet_a, feet_b, truth = {}, {}, {}
    for i in range(n_frames):
        xy = start + rng.normal(0, 0.6, start.shape) + i * 0.4
        pts = np.c_[xy, np.zeros(len(xy))]
        f = 100 + 30 * i
        truth[f] = xy
        feet_a[f] = project(SIDE, pts)
        feet_b[f] = project(ENDZ, pts)
    return feet_a, feet_b, truth


def test_it_recovers_the_second_camera_it_was_never_told_about():
    feet_a, feet_b, _truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    K, R, t, n = solve_second_view(cams_a, feet_a, feet_b, W, H)
    assert n >= 12
    centre = -R.T @ t
    assert np.linalg.norm(centre - np.array([-75.0, 0.0, 25.0])) < 6.0


def test_the_recovered_camera_puts_the_players_where_the_first_view_does():
    """The point of the exercise: the two views must agree, not merely fit."""
    feet_a, feet_b, truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    K, R, t, _n = solve_second_view(cams_a, feet_a, feet_b, W, H)
    f = sorted(truth)[len(truth) // 2]
    seen = project((K, R, t), np.c_[truth[f], np.zeros(len(truth[f]))])
    gap = np.linalg.norm(seen - feet_b[f], axis=1)
    assert np.median(gap) < 5.0


def test_a_seed_pointing_the_wrong_way_reconciles_nothing():
    """A wrong seed must fail loudly, not converge to a confident wrong camera."""
    feet_a, feet_b, _truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    with pytest.raises(CalibrationError, match="could not calibrate"):
        # Seeded from the far end AND the wrong side of the field, with a lens
        # that sees almost nothing.
        solve_second_view(cams_a, feet_a, feet_b, W, H,
                          seeds=[(0.0, 300.0, 2.0, 5.0)])


def test_the_gate_shrinks_so_a_rough_seed_can_still_start():
    feet_a, feet_b, truth = play()
    world = {f: truth[f] for f in truth}
    # 20 m and 10 degrees off the truth: nothing matches at a tight gate.
    H0 = seed_homography((-95.0, 0.0, 40.0), 35.0, W, H)
    _H, n = fit_from_seed(H0, world, feet_b, gate_start=20.0, gate_final=10.0)
    _H2, n2 = fit_from_seed(H0, world, feet_b)
    assert n2 > n
