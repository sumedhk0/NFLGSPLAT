import numpy as np

from nfl_gsplat.render import gaze


def body(x, y, facing_deg, z0=0.0):
    """A 22-joint stand-in: pelvis at (x, y), shoulders across the facing, the head above the neck."""
    J = np.zeros((22, 3))
    J[:, 0], J[:, 1], J[:, 2] = x, y, 1.0 + z0
    a = np.radians(facing_deg)
    right = np.array([np.sin(a), -np.cos(a), 0.0])          # the man's right, facing (cos a, sin a)
    J[16] = [x, y, 1.5 + z0] - 0.2 * right                  # left shoulder
    J[17] = [x, y, 1.5 + z0] + 0.2 * right                  # right shoulder
    J[15] = [x, y, 1.75 + z0]                               # head
    J[0] = [x, y, 1.0 + z0]
    return J


def heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


EVENTS = {"snap": 10, "release": 40, "catch": 60, "passer": 80, "receiver": 74}
TEAMS = {80: "KC", 74: "KC", 12: "KC", 2: "BAL"}
LOS = -24.0


def scene(frames, overrides=None):
    """KC attacks -x (lined up at x > LOS); a safety (2) backpedals toward -x facing +x; the tackle (12) sets back."""
    bodies, ball = {}, {}
    for f in frames:
        t = f - frames[0]
        b = {80: body(-19.0 + 0.02 * t, 0.0, 90.0),            # the passer drops back, chest to the far sideline
             74: body(-22.0 - 0.1 * t, 5.0, 180.0),            # the receiver runs downfield from his side of the line
             12: body(-23.0 + 0.02 * t, -3.0, 180.0),          # the tackle faces the defence, moving back
             2: body(-40.0 - 0.05 * t, 1.0, 0.0)}              # the safety faces the quarterback, moving back
        b.update({p: fn(f) for p, fn in (overrides or {}).items()})
        bodies[f] = b
        ball[f] = (-19.0 + 0.02 * t, 0.3, 1.4) if f < EVENTS["release"] else (-30.0, 4.0, 3.0)
    return bodies, ball


def test_a_backpedalling_safety_looks_at_the_ball_not_where_he_moves():
    bodies, ball = scene(range(0, 70, 2))
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    for f in (14, 24, 34):                                    # the safety moves toward -x; the ball is at +x
        assert abs(heading(look[2][f])) < 10.0


def test_a_tackle_in_his_pass_set_looks_where_his_chest_points():
    bodies, ball = scene(range(0, 70, 2))
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    for f in (14, 30, 50):
        assert abs(abs(heading(look[12][f])) - 180.0) < 1.0       # no turn, even with the ball in flight


def test_the_passer_turns_his_head_downfield_within_the_necks_range(monkeypatch):
    """Chest to the far sideline (+90), downfield is -x (180): the head turns the most the neck allows (+75)."""
    monkeypatch.setattr(gaze, "READ_FRAMES", 8)                # the read of the receiver: frames 32-38 here
    monkeypatch.setattr(gaze, "SMOOTH_SIGMA", 0.0)             # exact angles for the check
    bodies, ball = scene(range(0, 70, 2))
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    assert abs(heading(look[80][20]) - (90.0 + gaze.MAX_TURN_DEG)) < 1.0
    # in the last READ_FRAMES before the release he looks at his receiver (at -x, +y from him)
    d = bodies[36][74][15] - bodies[36][80][15]
    want = np.degrees(gaze.turn_toward(np.radians(90.0), np.arctan2(d[1], d[0])))
    assert abs(heading(look[80][36]) - want) < 0.5 and want < 90.0 + gaze.MAX_TURN_DEG - 1.0


def test_the_ball_in_flight_pulls_the_look_up_and_toward_it():
    bodies, ball = scene(range(0, 70, 2))
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    v = look[2][50]                                           # the safety at x ~ -42.5 watches the ball at (-30, 4, 3)
    assert v[2] > 0.0 and abs(heading(v) - np.degrees(np.arctan2(3.0, 12.5))) < 5.0


def test_the_turn_is_clamped_to_the_necks_range_and_nothing_behind_the_shoulder_is_watched():
    assert np.isclose(np.degrees(gaze.turn_toward(0.0, np.radians(110.0))), gaze.MAX_TURN_DEG)
    assert np.isclose(np.degrees(gaze.turn_toward(0.0, np.radians(-110.0))), -gaze.MAX_TURN_DEG)
    assert np.isclose(np.degrees(gaze.turn_toward(np.radians(170.0), np.radians(-170.0))), -170.0)   # across +-180
    assert np.isclose(gaze.turn_toward(0.3, np.radians(170.0)), 0.3)          # behind him: he keeps his chest's look


def test_the_passer_faking_with_his_back_to_the_defence_looks_where_his_chest_points(monkeypatch):
    """Play-action on play 1 (420-440): chest to his own end zone, downfield behind him -- no over-the-shoulder look."""
    monkeypatch.setattr(gaze, "SMOOTH_SIGMA", 0.0)
    bodies, ball = scene(range(0, 70, 2), {80: lambda f: body(-19.0, 0.0, 5.0)})
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    assert abs(heading(look[80][20]) - 5.0) < 0.5


def test_a_one_sample_torso_flicker_is_damped():
    bodies, ball = scene(range(0, 70, 2), {12: lambda f: body(-23.0, -3.0, 90.0 if f == 30 else 180.0)})
    look, _ = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    assert abs(abs(heading(look[12][30])) - 180.0) < 60.0     # the 90-degree flicker is smoothed well below 90


def test_the_eye_sits_ahead_of_and_above_the_head_joint():
    bodies, ball = scene(range(0, 70, 2))
    look, eye = gaze.look_table(bodies, ball, EVENTS, los_x=LOS, teams=TEAMS)
    J = bodies[20][12]
    e = np.asarray(eye[12][20])
    assert np.isclose(e[2], J[15][2] + gaze.EYE_UP)
    assert np.isclose(np.hypot(e[0] - J[15][0], e[1] - J[15][1]), gaze.EYE_FWD)
