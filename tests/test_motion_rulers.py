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
