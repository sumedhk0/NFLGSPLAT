"""render.depth_snap: a one-view body slides along its own ray onto the other camera's detection."""
import numpy as np

from nfl_gsplat.render.depth_snap import snap_ground, snap_one


class _Track:
    """A camera at (0, -80) looking up the field, with a conf of 1 on every frame."""

    def __init__(self, n=5):
        self.R = [np.eye(3)] * n
        self.t = [np.array([0.0, 80.0, 0.0])] * n
        self.conf = np.ones(n)


def test_a_body_slides_along_its_ray_to_the_other_camera():
    centre = np.array([0.0, -80.0])
    xy = np.array([0.0, -10.0])                      # 70 m down the ray from the camera
    # the other camera says this man is a metre further away, on the same ray
    got, took = snap_one(xy, centre, {7: np.array([0.0, -9.0])})
    assert took == 7 and np.allclose(got, [0.0, -9.0], atol=1e-6)
    # a body well off the ray is not his
    got, took = snap_one(xy, centre, {8: np.array([3.0, -10.0])})
    assert took is None and np.allclose(got, xy)
    # two bodies on the ray, a metre apart in depth but both near it: refuse
    got, took = snap_one(xy, centre, {7: np.array([0.1, -9.0]), 8: np.array([0.2, -11.0])})
    assert took is None


def test_the_move_is_capped_and_the_lateral_gate_holds():
    centre = np.array([0.0, -80.0])
    xy = np.array([0.0, -10.0])
    assert snap_one(xy, centre, {7: np.array([0.0, -4.0])})[1] is None        # 6 m away: too far to move
    assert snap_one(xy, centre, {7: np.array([0.4, -9.0])})[1] == 7           # 0.4 m off the ray: fine


def test_only_the_same_team_can_be_this_man():
    tr = _Track()
    side = {0: {1: np.array([0.0, -10.0])}}
    other = {0: {5: np.array([0.0, -9.0])}}
    teams = {1: "KC", 5: "BAL"}
    out, n = snap_ground(side, other, tr, teams=teams)
    assert n == 0 and np.allclose(out[0][1], [0.0, -10.0])
    teams[5] = "KC"
    out, n = snap_ground(side, other, tr, teams=teams)
    assert n == 1 and np.allclose(out[0][1], [0.0, -9.0])


def test_the_other_camera_runs_on_its_own_clock():
    tr = _Track()
    side = {0: {1: np.array([0.0, -10.0])}}
    other = {-3: {1: np.array([0.0, -9.0])}}          # the other clip runs 3 frames behind
    out, n = snap_ground(side, other, tr, frame_shift=-3)
    assert n == 1 and np.allclose(out[0][1], [0.0, -9.0])
    out, n = snap_ground(side, other, tr, frame_shift=0)
    assert n == 0
