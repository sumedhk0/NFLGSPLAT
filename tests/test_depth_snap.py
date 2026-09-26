"""render.depth_snap: a one-view body slides along its own ray onto the other camera's detection."""
import numpy as np

from nfl_gsplat.render.depth_snap import snap_ground, snap_one, veto_outlier_snaps


def test_a_snap_that_disagrees_with_its_neighbours_is_undone():
    """Play 1's id 1 slid 2.0 m along its ray on one frame while its neighbours slid nothing: the
    wrong man on the ray. A run of consistent slides stays; the outlier in it, and the lone big one,
    are put back where the sideline had them."""
    tr = _Track(n=20)
    side = {f: {1: np.array([0.0, -10.0]), 2: np.array([5.0, -10.0])} for f in range(20)}
    other = {f: {1: np.array([0.0, -9.7])} for f in range(20)}            # id 1: 0.3 m further, every frame
    other[10][1] = np.array([0.0, -8.0])                                  # ... except one frame: 2.0 m
    other[15] = {1: np.array([0.0, -9.7]), 2: np.array([5.0, -8.0])}     # id 2: one lone 2 m snap
    out, n = snap_ground(side, other, tr)
    assert n == 19                                                        # 20 - the outlier - the loner
    assert np.allclose(out[9][1], [0.0, -9.7]) and np.allclose(out[11][1], [0.0, -9.7])
    assert np.allclose(out[10][1], [0.0, -10.0])                          # undone, not median-replaced
    assert np.allclose(out[15][2], [5.0, -10.0])
    # the veto itself, on the corrections: a lone slide under the limit is kept
    assert veto_outlier_snaps({3: {4: 0.8}}) == set()
    assert veto_outlier_snaps({3: {4: 1.5}}) == {(4, 3)}
    # the jump veto alone (the window off) still catches both: each is a 2 m jump from the frame before
    out, n = snap_ground(side, other, tr, veto_window=0)
    assert n == 19 and np.allclose(out[10][1], [0.0, -10.0])
    # and with both vetoes off the per-frame answer comes back
    out, n = snap_ground(side, other, tr, veto_window=0, jump_m=False)
    assert n == 21 and np.allclose(out[10][1], [0.0, -8.0])


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


def test_exclusive_snapping_gives_an_endzone_body_to_the_nearer_ray_only():
    """Two sideline bodies whose rays both pass an endzone body: with EXCLUSIVE the nearer ray
    snaps, the other keeps the sideline's own point; without it both slide onto the same man."""
    tr = _Track(n=6)
    side = {f: {1: np.array([0.0, -10.0]), 2: np.array([0.4, -10.0])} for f in range(6)}
    other = {f: {7: np.array([0.05, -9.0])} for f in range(6)}          # one endzone body, 1 m further up both rays
    both, n_both = snap_ground(side, other, tr, veto_window=0, exclusive=False)
    one, n_one = snap_ground(side, other, tr, veto_window=0, exclusive=True)
    assert n_both == 12 and n_one == 6
    assert np.allclose(both[3][1][1], -9.0, atol=0.05) and np.allclose(both[3][2][1], -9.0, atol=0.05)
    assert np.allclose(one[3][1][1], -9.0, atol=0.05) and np.allclose(one[3][2], [0.4, -10.0])


def test_veto_jumps_undoes_a_run_of_snaps_that_make_the_body_jump_but_keeps_a_continuous_one():
    from nfl_gsplat.render.depth_snap import veto_jumps

    # a body walking along y at 0.1 m/frame; from frame 5 the snap moves it 2 m deeper (the wrong man)
    side = {f: {28: np.array([0.0, 0.1 * f])} for f in range(10)}
    out = {f: {28: np.array([0.0, 0.1 * f + (2.0 if f >= 5 else 0.0)])} for f in range(10)}
    deltas = {28: {f: 2.0 for f in range(5, 10)}}
    undone = veto_jumps(out, side, deltas, set(), jump_m=0.6)
    assert undone == {(f, 28) for f in range(5, 10)}
    assert all(np.allclose(out[f][28], side[f][28]) for f in range(10))
    # a snap that moves a body 0.4 m stays (no jump), and a body first drawn on a snapped frame stays too
    out2 = {f: {7: np.array([0.0, 0.1 * f + (0.4 if f >= 5 else 0.0)]), 9: np.array([1.0, 3.0])} for f in range(5, 10)}
    side2 = {f: {7: np.array([0.0, 0.1 * f]), 9: np.array([1.0, 1.0])} for f in range(5, 10)}
    assert veto_jumps(out2, side2, {7: {f: 0.4 for f in range(5, 10)}, 9: {5: 2.0}}, set(), jump_m=0.6) == set()

def test_snap_own_id_resolves_two_bodies_on_one_ray():
    """Two teammates side by side across the field sit on the same sideline ray: the positional snap refuses (a
    runner-up within MARGIN_M), the id's own endzone point settles it; the flag off keeps the shipped refusal."""
    import numpy as np

    from nfl_gsplat.render import depth_snap as ds

    class _Track:
        conf = np.ones(10)

    centre = np.array([0.0, -100.0])
    ds_centre = ds.camera_ground_centre
    ds.camera_ground_centre = lambda track, f: centre
    try:
        side = {5: {37: np.array([0.0, -3.7]), 80: np.array([2.0, 0.0])}}
        other = {5: {37: np.array([0.05, -2.8]), 139: np.array([-0.05, -1.9])}}
        teams = {37: "KC", 139: "KC", 80: "KC"}
        off, _ = ds.snap_ground(side, other, _Track(), teams=teams, own_id=False, veto_window=0, jump_m=False)
        assert np.allclose(off[5][37], side[5][37])                        # ambiguous: left where the sideline put him
        on, _ = ds.snap_ground(side, other, _Track(), teams=teams, own_id=True, veto_window=0, jump_m=False)
        assert np.allclose(on[5][37], [0.0, -2.8], atol=0.06)               # onto his own endzone point along the ray
        assert np.allclose(on[5][80], side[5][80])                          # no endzone point of his own, no candidate
    finally:
        ds.camera_ground_centre = ds_centre


def test_spare_missed_keeps_a_missed_mans_endzone_point_out_of_other_mens_positional_snaps():
    """Play 1 at 556-612: Madubuike's sideline ray passes Ojabo's endzone point 1 m deeper; Ojabo's sideline tracks
    end at 527, so the loader draws that point as Ojabo AND the snap slid Madubuike onto it -- one man drawn twice.
    With spare, a man the sideline tracks at other times but misses on this frame keeps his endzone point; an
    endzone-only fragment (never on the sideline) stays a candidate, and on frames the sideline has him the rule is
    silent."""
    tr = _Track(n=6)
    side = {f: {4: np.array([0.0, -10.0])} for f in range(6)}
    for f in (0, 1):
        side[f][1] = np.array([3.0, -20.0])                              # id 1 on the sideline at 0-1 only
    other = {f: {1: np.array([0.05, -9.0])} for f in range(6)}          # id 1's endzone point on 4's ray
    kw = dict(veto_window=0, jump_m=False, own_id=False)
    off, n_off = snap_ground(side, other, tr, spare_missed=False, **kw)
    on, n_on = snap_ground(side, other, tr, spare_missed=True, **kw)
    assert all(np.allclose(off[f][4], [0.05, -9.0], atol=0.06) for f in range(6))
    assert all(np.allclose(on[f][4], [0.05, -9.0], atol=0.06) for f in (0, 1))   # he is on the sideline: silent
    assert all(np.allclose(on[f][4], [0.0, -10.0]) for f in range(2, 6))         # missed there: spared
    assert n_off == 6 and n_on == 2
    frag = {f: {9: np.array([0.05, -9.0])} for f in range(6)}                    # an endzone-only fragment
    got, n = snap_ground(side, frag, tr, spare_missed=True, **kw)
    assert n == 6 and all(np.allclose(got[f][4], [0.05, -9.0], atol=0.06) for f in range(6))
