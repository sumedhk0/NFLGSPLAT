"""The plausibility rulers read what they claim to: a teleport is a step, running is not jitter, a gap is
not a step, and the census is blind to position (so it can only guard, never score)."""
import numpy as np

from nfl_gsplat.render.motion_rulers import (
    census_error,
    contiguous_steps,
    handover_steps,
    hinge_violations,
    joint_motion,
    positions_by_id,
    second_differences,
    summarize,
)


class S:
    def __init__(self, pid, xy):
        self.pid, self.xy = pid, np.asarray(xy, float)


def walk(pid, frames, v=(0.1, 0.0), start=(0.0, 0.0)):
    return {f: np.asarray(start) + np.asarray(v) * (f - frames[0]) for f in frames}


def test_a_runner_has_no_jitter_and_no_step():
    pos = {1: walk(1, range(100), v=(0.15, 0.05))}       # 9.5 m/s, a sprint
    steps = contiguous_steps(pos)
    assert len(steps) == 99 and max(s[0] for s in steps) < 0.2
    assert np.allclose(second_differences(pos)[1], 0.0, atol=1e-9)


def test_a_teleport_is_the_largest_step_and_names_the_frame():
    byf = walk(1, range(60))
    for f in range(30, 60):
        byf[f] = byf[f] + np.array([0.0, 6.0])         # the tail: six metres across at frame 30
    steps = contiguous_steps({1: byf})
    d, pid, f = steps[0]
    assert (pid, f) == (1, 29) and abs(d - np.hypot(0.1, 6.0)) < 1e-9


def test_a_gap_is_not_a_step():
    byf = walk(1, range(20))
    byf.update({f: np.array([50.0, 50.0]) for f in range(40, 60)})   # resumes far away after a gap
    steps = contiguous_steps({1: byf})
    assert max(s[0] for s in steps) < 0.2                # nothing was drawn between 19 and 40
    # and the second difference never straddles the gap either
    assert len(second_differences({1: byf})[1]) == 18 + 18


def test_a_square_wave_reads_as_jitter():
    byf = {f: np.array([0.0, 0.3 * (f % 2)]) for f in range(40)}   # toggles 0.3 m every frame
    j = second_differences({1: byf})[1]
    assert np.allclose(j, 0.6)


def test_census_counts_only_the_two_teams_and_is_blind_to_position():
    team = {1: "KC", 2: "KC", 3: "BAL", 4: None, 5: "REF"}
    frames = {f: [S(1, (0, 0)), S(2, (100, 100)), S(3, (0, 0)), S(4, (0, 0)), S(5, (0, 0))] for f in range(5)}
    err, means = census_error(frames, team, 0, 4)
    assert means == {"KC": 2.0, "BAL": 1.0} and err == (11 - 2) + (11 - 1)
    # a window with no frames is nan, not a crash
    assert np.isnan(census_error(frames, team, 100, 200)[0])


def test_joint_motion_is_zero_for_a_held_pose_and_reads_a_twitch():
    still = {f: np.zeros((22, 3)) for f in range(30)}
    twitch = {f: np.zeros((22, 3)) for f in range(30)}
    for f in range(30):
        twitch[f][20] = [0.0, 0.0, 0.1 * (f % 2)]        # one hand flicks 10 cm every frame
    jm = joint_motion({1: still, 2: twitch})
    assert np.allclose(jm[1][0], 0.0) and np.allclose(jm[1][1], 0.0)
    assert np.allclose(jm[2][0], 0.2) and np.allclose(jm[2][1], 0.1)


def test_summarize_reports_the_live_window_separately():
    byf = walk(1, range(0, 200))
    byf[150] = byf[150] + np.array([0.0, 5.0])           # one switch after the live window
    pos = {1: byf}
    frames = {f: [S(1, byf[f])] for f in byf}
    rep = summarize(pos, frames, {1: "KC"}, lo=50, hi=100)
    assert rep["steps"]["full_over_hard"] == 2 and rep["steps"]["live_over_step"] == 0
    assert rep["steps"]["worst"][0]["frame"] in (149, 150)
    assert all(50 <= w["frame"] <= 100 for w in rep["steps"]["worst_live"]) and len(rep["steps"]["worst_live"]) == 24
    assert rep["root_jitter"]["live"]["p90"] < 1e-9
    assert rep["census"]["live_teams"]["KC"] == 1.0 and "joints" not in rep


def test_a_handover_is_a_step_drawn_from_different_cameras():
    byf = walk(1, range(40))
    for f in range(20, 40):
        byf[f] = byf[f] + np.array([0.0, 0.5])            # the cameras disagree by half a metre
    views = {1: {f: ("endzone",) if f < 20 else ("sideline", "endzone") for f in range(40)}}
    steps = [s for s in contiguous_steps({1: byf}) if s[0] > 0.25]
    assert [(p, f) for _d, p, f in handover_steps(steps, views)] == [(1, 19)]
    # the same jump with the cameras unchanged is a switch, not a handover
    same = {1: {f: ("sideline", "endzone") for f in range(40)}}
    assert handover_steps(steps, same) == []
    frames = {f: [type("S", (), {"pid": 1, "xy": byf[f], "views": views[1][f]})()] for f in byf}
    rep = summarize({1: byf}, frames, {1: "KC"}, lo=0, hi=39)
    assert rep["steps"]["live_handover"] == 1 and rep["steps"]["worst"][0]["handover"] is True
    assert rep["steps"]["worst"][0]["views"] == [["endzone"], ["sideline", "endzone"]]


def test_hinge_violations_count_backwards_and_sideways_hinges():
    bp = np.zeros((10, 21, 3))
    bp[:5, 3, 0] = np.radians(-40)                    # L_knee backwards on 5 frames
    bp[:, 18, 1] = np.radians(60)                     # R_elbow flexed 60: fine
    bp[:2, 18, 0] = np.radians(50)                    # ... and sideways on 2 frames
    v = hinge_violations(bp)
    assert v["n"] == 10 and abs(v["hyperextended"] - 5 / 40) < 1e-9 and abs(v["off_axis"] - 2 / 40) < 1e-9
    assert np.isnan(hinge_violations(np.zeros((0, 21, 3)))["hyperextended"])


def test_positions_by_id_keys_by_int():
    frames = {np.int64(3): [S(np.int64(7), (1, 2))]}
    pos = positions_by_id(frames)
    assert list(pos) == [7] and list(pos[7]) == [3] and pos[7][3].tolist() == [1.0, 2.0]


def test_jerk_steps_flag_a_hop_but_not_a_sprinter():
    """A man running at 0.30 m/frame (9 m/s) takes long, EQUAL steps: the absolute step ruler counts
    every one, the jerk ruler none. One step of 0.45 m among 0.30 m strides is a hop; so is a 0.25 m
    step among a standing man's 0.02 m shuffles, which the absolute ruler never sees."""
    from nfl_gsplat.render import motion_rulers as mr

    sprint = {f: np.array([0.30 * f, 0.0]) for f in range(100, 130)}
    hop = {f: np.array([0.30 * f + (0.20 if f > 115 else 0.0), 0.0]) for f in range(100, 130)}
    shuffle = {f: np.array([0.02 * f + (0.25 if f > 120 else 0.0), 1.0]) for f in range(100, 130)}
    assert mr.jerk_steps({1: sprint}) == []
    assert sum(1 for s in mr.contiguous_steps({1: sprint}) if s[0] > 0.25) == 29
    j = mr.jerk_steps({2: hop, 3: shuffle})
    assert {(p, f) for _e, _s, p, f in j} == {(2, 115), (3, 120)}
    ex = {(p, f): e for e, _s, p, f in j}
    assert abs(ex[(2, 115)] - 0.20) < 1e-9 and abs(ex[(3, 120)] - 0.25) < 1e-9
    # a step across a gap is not a step, and an id with fewer than two neighbouring steps is not judged
    gappy = {100: np.array([0.0, 0.0]), 101: np.array([0.0, 0.0]), 105: np.array([9.0, 0.0])}
    assert mr.jerk_steps({4: gappy}) == []


def test_skating_ruler_tells_a_planted_foot_from_a_gliding_one():
    """A body moving 0.12 m/frame: with the left ankle held in the world for 7 of every 10 frames
    (pelvis-relative it drifts back at the body's speed) the slower ankle plants on the triples whose
    both ends fall in the hold, about half; with both ankles fixed relative to the pelvis nothing
    plants and the ratio is 1."""
    from nfl_gsplat.render import motion_rulers as mr

    T = 60
    pos = {f: np.array([0.12 * f, 0.0]) for f in range(T)}     # clear of the 0.1 moving threshold
    glide = {f: np.zeros((22, 3)) for f in range(T)}
    for f in range(T):
        glide[f][7] = [0.1, 0.0, 0.0]; glide[f][8] = [-0.1, 0.0, 0.0]
    sk = mr.skating({1: pos}, {1: glide})
    assert abs(sk["ratio_p50"] - 1.0) < 1e-9 and sk["planted"] == 0.0 and sk["n"] == T - 2
    plant = {f: np.zeros((22, 3)) for f in range(T)}
    for f in range(T):
        k = f % 10
        world_l = np.array([0.12 * (f - k) + 0.3, 0.0]) if k < 7 else np.array([0.12 * f + 0.3, 0.0])   # held 7 frames, then swung
        plant[f][7, :2] = world_l - pos[f]
        plant[f][8] = [-0.1, 0.0, 0.0]
    sk2 = mr.skating({1: pos}, {1: plant})
    assert 0.4 < sk2["planted"] < 0.6 and sk2["ratio_p50"] < 0.9
    assert sk2["worst"][0][0] == 1

