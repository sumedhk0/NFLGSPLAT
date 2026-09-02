"""The players must be able to calibrate the view the paint cannot."""
import numpy as np
import pytest

from nfl_gsplat.calibration.from_players import (
    fit_from_seed,
    focal_from_boxes,
    seed_homography,
    solve_second_view,
)
from nfl_gsplat.errors import CalibrationError

W, H = 1920, 1080
PLAYER_M = 1.85


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


def boxes_for(c, xy, height_m=PLAYER_M, half_w_px=15.0):
    """Person boxes as the detector would draw them: feet on the turf, head above."""
    feet = project(c, np.c_[xy, np.zeros(len(xy))])
    head = project(c, np.c_[xy, np.full(len(xy), height_m)])
    return np.column_stack([feet[:, 0] - half_w_px, head[:, 1],
                            feet[:, 0] + half_w_px, feet[:, 1]])


# The sideline camera the players verified on the real play, and an endzone
# camera behind the end zone where a real one sits.
SIDE = cam([29.5, -95.3, 47.1], 7500.0)
ENDZ_CENTRE = np.array([-75.0, 0.0, 25.0])
ENDZ = cam(ENDZ_CENTRE, 1500.0)


def play(n_frames=6, n_players=20, seed=0, endzone=ENDZ, head_m=PLAYER_M):
    """Players milling about on the turf, seen by both cameras."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([-4, -8], [16, 8], size=(n_players, 2))
    feet_a, boxes_b, truth = {}, {}, {}
    for i in range(n_frames):
        xy = start + rng.normal(0, 0.6, start.shape) + i * 0.4
        f = 100 + 30 * i
        truth[f] = xy
        feet_a[f] = project(SIDE, np.c_[xy, np.zeros(len(xy))])
        boxes_b[f] = boxes_for(endzone, xy, height_m=head_m)
    return feet_a, boxes_b, truth


def test_the_boxes_read_the_lens():
    """A 1.85 m player at range d is f*1.85/d px tall; invert it."""
    _feet_a, boxes_b, truth = play()
    d = np.linalg.norm(ENDZ_CENTRE - np.r_[truth[100].mean(0), 0.0])
    f = focal_from_boxes(boxes_b, d)
    assert abs(f - 1500.0) / 1500.0 < 0.15


def test_it_recovers_the_second_camera_it_was_never_told_about():
    feet_a, boxes_b, _truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    K, R, t, n = solve_second_view(cams_a, feet_a, boxes_b, W, H)
    assert n >= 12
    assert np.linalg.norm(-R.T @ t - ENDZ_CENTRE) < 6.0


def test_the_recovered_camera_puts_the_players_where_the_first_view_does():
    """The point of the exercise: the two views must agree, not merely fit."""
    feet_a, boxes_b, truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    K, R, t, _n = solve_second_view(cams_a, feet_a, boxes_b, W, H)
    f = sorted(truth)[len(truth) // 2]
    seen = project((K, R, t), np.c_[truth[f], np.zeros(len(truth[f]))])
    feet_b = np.column_stack([(boxes_b[f][:, 0] + boxes_b[f][:, 2]) / 2.0,
                              boxes_b[f][:, 3]])
    gap = np.linalg.norm(seen - feet_b, axis=1)
    assert np.median(gap) < 5.0


def test_a_camera_that_fits_the_feet_but_not_the_people_is_refused():
    """The measured failure, made synthetic.

    On the real play the first version converged to a 119 degree lens 5 m
    above the turf. It matched feet, so it had inliers, and it sat inside the
    mount box. What it could not do was make the players a believable
    height. Here the feet are consistent and the heads say 5 m: the fit is
    perfect and the answer must still be refused.
    """
    feet_a, boxes_b, _truth = play(head_m=5.0)
    cams_a = dict.fromkeys(feet_a, SIDE)
    with pytest.raises(CalibrationError, match="believable height"):
        solve_second_view(cams_a, feet_a, boxes_b, W, H)


def test_a_seed_pointing_the_wrong_way_reconciles_nothing():
    """A wrong seed must fail loudly, not converge to a confident wrong camera."""
    feet_a, boxes_b, _truth = play()
    cams_a = dict.fromkeys(feet_a, SIDE)
    with pytest.raises(CalibrationError, match="could not calibrate"):
        # From the far end AND the wrong side, with a lens that sees nothing.
        solve_second_view(cams_a, feet_a, boxes_b, W, H,
                          seeds=[((0.0, 300.0, 2.0), 20000.0)])


def test_the_gate_shrinks_so_a_rough_seed_can_still_start():
    _feet_a, boxes_b, truth = play()
    feet_b = {f: np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])
              for f, b in boxes_b.items()}
    world = dict(truth)
    # 20 m and a third of a lens off the truth: nothing matches at a tight gate.
    H0 = seed_homography((-95.0, 0.0, 40.0), 1000.0, W, H,
                         target=np.r_[truth[100].mean(0), 0.0])
    _H, n = fit_from_seed(H0, world, feet_b, gate_start=20.0, gate_final=10.0)
    _H2, n2 = fit_from_seed(H0, world, feet_b)
    assert n2 > n
