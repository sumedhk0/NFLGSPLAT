import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render import tackle
from nfl_gsplat.render.timeline import upright_from_yaw, yaw_of


def test_fall_schedule_ends_after_the_down_frame_and_holds_to_the_last():
    s = tackle.fall_schedule(607, frames=8, last=615, settle=0)
    assert min(s) == 600 and max(s) == 615
    assert abs(s[600] - 1 / 8) < 1e-9 and s[607] == 1.0 and s[615] == 1.0
    assert 599 not in s
    assert tackle.fall_schedule(607, frames=4, settle=0) == {604: 0.25, 605: 0.5, 606: 0.75, 607: 1.0}
    # "down" is mid-fall: the body is flat FALL_SETTLE frames later (the film: down at 607, the pile flat by 610)
    s3 = tackle.fall_schedule(607, frames=8, last=615)
    assert s3[610] == 1.0 and s3[607] < 1.0 and min(s3) == 603 and abs(s3[603] - 1 / 8) < 1e-9


def test_fall_orient_pitches_the_body_face_down_along_its_facing():
    yaw = 0.3
    go = upright_from_yaw(yaw)
    up0 = Rotation.from_rotvec(go).apply([0.0, 1.0, 0.0])       # the model's up, world z when upright
    assert np.allclose(up0, [0, 0, 1], atol=1e-6)
    assert np.allclose(tackle.fall_orient(go, 0.0), go)
    up1 = Rotation.from_rotvec(tackle.fall_orient(go, 1.0)).apply([0.0, 1.0, 0.0])
    assert np.allclose(up1, [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-6)   # lying along the way he ran, face down
    # halfway the up is between the two, and the facing on the ground is unchanged until the body is flat
    up_half = Rotation.from_rotvec(tackle.fall_orient(go, 0.5)).apply([0.0, 1.0, 0.0])
    assert up_half[2] > 0.5 and np.hypot(up_half[0], up_half[1]) > 0.5
    assert abs(yaw_of(tackle.fall_orient(go, 0.5)) - yaw) < 1e-6


def test_fall_orient_toward_a_direction_and_the_tacklers_within_reach():
    from nfl_gsplat.render import timeline as tlm

    go = upright_from_yaw(0.0)                                   # facing +x
    up1 = Rotation.from_rotvec(tackle.fall_orient(go, 1.0, direction=[0.0, 2.0])).apply([0.0, 1.0, 0.0])
    assert np.allclose(up1, [0, 1, 0], atol=1e-6)                # falls onto the man at +y, not along his facing

    def st(pid, x, y):
        return tlm.PlayerState(pid=pid, xy=np.array([x, y]), body_pose=np.zeros((21, 3)), global_orient=go,
                               betas=np.zeros(10), source="sideline")
    states = [st(74, 0.0, 0.0), st(28, 0.5, 1.0), st(55, -1.0, 0.0), st(7, 3.0, 0.0), st(12, 0.3, 0.0)]
    teams = {74: "KC", 28: "BAL", 55: "BAL", 7: "BAL", 12: "KC"}
    got = tackle.tacklers(states, 74, teams, within_m=1.5)
    assert sorted(got) == [28, 55]                               # 7 is 3 m off, 12 is a teammate
    assert np.allclose(got[28], [-0.5, -1.0]) and np.allclose(got[55], [1.0, 0.0])
    assert tackle.tacklers(states, 99, teams) == {}


def test_tackled_body_pose_curls_the_legs_and_leaves_the_arms_holding_the_ball():
    bp = np.zeros((21, 3)); bp[15] = [0.1, -0.9, -0.7]; bp[16] = [0.1, 0.9, 0.7]; bp[19] = [0.2, 0, 0]
    out = tackle.tackled_body_pose(bp, 1.0)
    assert np.allclose(out[3], tackle.TACKLED_ROWS[3]) and np.allclose(out[0], tackle.TACKLED_ROWS[0])
    for row in (15, 16, 17, 18, 19, 20):
        assert np.allclose(out[row], bp[row])
    half = tackle.tackled_body_pose(bp, 0.5)
    assert 0 < half[3][0] < tackle.TACKLED_ROWS[3][0]
    assert np.allclose(tackle.tackled_body_pose(bp, 0.0), bp)


def test_lost_tacklers_are_the_other_team_bodies_lost_beside_the_carrier_before_the_down():
    from types import SimpleNamespace as S
    states = {}
    for f in range(580, 620):
        rows = [S(pid=74, xy=np.array([-32.0 - 0.05 * (f - 580), 1.0]))]              # the carrier, drawn to the end
        if f < 600:
            rows.append(S(pid=171, xy=np.array([-32.3, 1.4])))                          # lost at 599, half a metre off
        if f < 597:
            rows.append(S(pid=55, xy=np.array([-40.0, 8.0])))                           # lost at 596, far away
        if f < 604:
            rows.append(S(pid=87, xy=np.array([-32.4, 1.2])))                           # lost at 603 beside him, but a teammate
        if f < 590:
            rows.append(S(pid=21, xy=np.array([-32.2, 1.1])))                           # lost at 589: too early
        states[f] = rows
    team_of = {74: "KC", 171: "BAL", 55: "BAL", 87: "KC", 21: "BAL"}
    got = tackle.lost_tacklers(states, 74, team_of, 607, lost_frames=12, within_m=1.5)
    assert got == {171: 599}
    assert tackle.lost_tacklers(states, 74, team_of, 607, lost_frames=20, within_m=1.5) == {171: 599, 21: 589}
    assert tackle.lost_tacklers(states, 74, {}, 607) == {171: 599, 87: 603}                  # without teams, anyone close counts
