"""play_timeline.anchor_feet: a body placed from a FEET point stands with its own posed feet there, not its pelvis."""
import numpy as np

from nfl_gsplat.render import timeline as tlm


def _state(pid, xy):
    return tlm.PlayerState(pid=pid, xy=np.asarray(xy, float), body_pose=np.zeros((21, 3)),
                           global_orient=np.zeros(3), betas=np.zeros(10), source="fused")


def _joints_with_offset(off_by_pid):
    """A joints_fn whose pelvis (joint 0) sits ``off`` (xy) from the ankle midpoint (joints 7, 8) for each state's id."""
    def fn(states):
        J = np.zeros((len(states), 9, 3))
        for i, s in enumerate(states):
            J[i, 0, :2] = off_by_pid.get(int(s.pid), (0.0, 0.0))
        return J
    return fn


def test_feet_point_states_move_by_the_posed_offset_and_refit_points_stay():
    """Play 1 (2026-09-25): Madubuike lunging at 444-474, his ground point at his trailing ankles, was drawn a torso width
    behind himself because the renderer puts the pelvis there. Id 4 is on feet points: every state moves by the posed
    pelvis - ankle offset. Id 80 is on the refit's pelvis points: nothing moves. Id 7 passes from refit to feet points
    at frame 20: the shift ramps with the share of feet points in the +-half window, no jump."""
    from nfl_gsplat.render.play_timeline import anchor_feet

    tl = tlm.Timeline(frames=list(range(40)))
    for f in range(40):
        tl.states[f] = [_state(4, (f * 0.1, 0.0)), _state(80, (5.0, 5.0)), _state(7, (10.0, 0.0))]
    pelvis_keys = {(f, 80) for f in range(40)} | {(f, 7) for f in range(20)}
    moved = anchor_feet(tl, pelvis_keys, _joints_with_offset({4: (0.6, 0.0), 80: (0.4, 0.0), 7: (0.0, 0.5)}), half=3)
    xy = {(f, s.pid): s.xy for f, ss in tl.states.items() for s in ss}
    assert np.allclose(xy[(10, 4)], [1.0 + 0.6, 0.0])            # feet points: the whole posed offset
    assert np.allclose(xy[(10, 80)], [5.0, 5.0])                 # the refit's pelvis: untouched
    assert np.allclose(xy[(5, 7)], [10.0, 0.0]) and np.allclose(xy[(30, 7)], [10.0, 0.5])
    ramp = [xy[(f, 7)][1] for f in range(15, 26)]
    assert all(b >= a for a, b in zip(ramp, ramp[1:])) and max(np.diff(ramp)) <= 0.5 / 7 + 1e-9   # a blend, not a jump
    assert len(moved) == 120


def test_anchor_feet_with_no_offset_changes_nothing():
    from nfl_gsplat.render.play_timeline import anchor_feet

    tl = tlm.Timeline(frames=[0, 1, 2])
    for f in range(3):
        tl.states[f] = [_state(4, (1.0, 2.0))]
    anchor_feet(tl, set(), _joints_with_offset({}), half=7)
    assert all(np.allclose(s.xy, [1.0, 2.0]) for ss in tl.states.values() for s in ss)
