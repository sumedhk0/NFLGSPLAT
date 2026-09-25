"""The per-play action classifier: which way a man moves relative to his chest (pose.action_class)."""
import numpy as np
import pytest

from nfl_gsplat.pose import action_class as ac


def test_relative_class_forward_back_lateral_and_slow():
    v = np.array([3.0, 0.0])                                    # moving +x at 3 m/s
    assert ac.relative_class(0.0, v) == ac.FWD
    assert ac.relative_class(np.pi, v) == ac.BACK                # facing -x, moving +x: a backpedal
    assert ac.relative_class(np.pi / 2, v) == ac.LAT
    assert ac.relative_class(np.radians(40), v) == ac.FWD
    assert ac.relative_class(np.radians(140), v) == ac.BACK
    assert ac.relative_class(0.0, np.array([0.5, 0.0])) == ac.SLOW   # under V_MIN: no direction to judge


def test_track_motion_uses_the_clip_frame_rate():
    frames = list(range(0, 41, 2))                               # exported every 2 play frames
    xy = {f: np.array([f * 0.1, 0.0]) for f in frames}           # 0.1 m per play frame = 5.994 m/s at 59.94 fps
    speed, heading = ac.track_motion(xy, 20, half=4)
    assert abs(speed - 0.1 * ac.FPS) < 1e-6 and abs(heading) < 1e-9
    assert ac.track_motion(xy, 0, half=4) is None                # no frame 4 before


def test_features_do_not_see_the_facing():
    row = dict(speed=4.0, heading=np.pi, attack_sign=-1.0, offence=True, role="WR", t_snap=1.2, phase="pre",
               depth=3.0, ball_bearing=0.3, nearest_opp=2.5, nose=0.8, cam_bearing=-np.pi / 2, lr_sign=1.0)
    x = ac.feature_vector(row)
    assert x.shape == (len(ac.FEATURE_NAMES),) and np.all(np.isfinite(x))
    assert not any("facing" in n or "yaw" in n for n in ac.FEATURE_NAMES)
    # moving along the attack direction reads as downfield (+1)
    i = ac.FEATURE_NAMES.index("down_cos")
    assert abs(x[i] - 1.0) < 1e-9


def test_classifier_learns_a_separable_toy_problem_and_holds_out_by_id():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, len(ac.FEATURE_NAMES)))
    y = np.where(X[:, 0] > 0.5, ac.FWD, np.where(X[:, 0] < -0.5, ac.BACK, ac.LAT))
    ids = rng.integers(0, 20, size=n)
    acc = ac.grouped_cv_accuracy(X, y, ids, folds=4, seed=0)
    assert acc > 0.85
    model = ac.train(X, y, seed=0)
    p = ac.predict_proba(model, X[:5])
    assert p.shape == (5, len(ac.CLASSES)) and np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_expected_facing_for_confident_directional_classes_only():
    heading = 0.3
    assert ac.expected_facing(ac.FWD, heading) == pytest.approx(0.3)
    assert abs(((ac.expected_facing(ac.BACK, heading) - (0.3 + np.pi)) + np.pi) % (2 * np.pi) - np.pi) < 1e-9
    assert ac.expected_facing(ac.LAT, heading) is None           # sideways: which side is not known
