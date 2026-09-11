"""identity.roles: the formation before the snap says the role; unnamed ids take the role's build."""
import numpy as np

from nfl_gsplat.identity.roles import (POSITION_BUILDS, apply_role_builds, assign_roles, line_of_scrimmage,
                                       snap_frame)


def formation():
    """Play 1's pre-snap layout in field metres: offence KC on the +x side of x = -24."""
    s = {}
    teams = {}
    # KC offensive line, crouched, at x -22.6, y -5..+3; a wide receiver at y -10; the back 5.6 m behind
    for pid, y in ((18, -3.4), (17, 0.3), (12, 2.9), (3, 5.3)):
        s[pid] = (-22.6, y, 1.2, 100); teams[pid] = "KC"
    s[11] = (-22.6, -6.9, 1.24, 100); teams[11] = "KC"     # tight end in a stance outside the tackle
    s[9] = (-21.7, -10.0, 1.65, 100); teams[9] = "KC"
    s[5] = (-17.0, -1.6, 1.93, 100); teams[5] = "KC"        # the back, offset from the centre line
    s[8] = (-18.0, 0.9, 1.6, 100); teams[8] = "KC"          # the passer in the gun, on the centre's line
    s[16] = (-21.9, 0.6, 1.9, 100); teams[16] = "KC"        # an upright body right behind the centre: under centre
    # BAL: three crouched linemen at -25.2, a standing edge at -25.0 / -5.5, linebackers at -27.5, deep men at -34
    for pid, y, a in ((4, 2.3, 0.85), (13, -1.3, 0.94), (1, 5.6, 1.18)):
        s[pid] = (-25.2, y, a, 100); teams[pid] = "BAL"
    s[15] = (-25.0, -5.5, 1.87, 100); teams[15] = "BAL"
    for pid, y in ((7, 2.7), (14, -1.8)):
        s[pid] = (-27.5, y, 2.3, 100); teams[pid] = "BAL"
    for pid, y in ((6, 7.4), (21, -6.5)):                    # nickels off the ball, outside the box
        s[pid] = (-27.5, y, 2.3, 100); teams[pid] = "BAL"
    s[0] = (-27.6, -10.5, 2.2, 100); teams[0] = "BAL"
    for pid, y in ((2, -4.9), (27, 5.4), (10, 5.3)):
        s[pid] = (-34.0, y, 2.5, 100); teams[pid] = "BAL"
    return s, teams


def test_roles_from_the_formation():
    s, teams = formation()
    los, sign, yc = line_of_scrimmage(s, teams, "KC")
    assert -24.5 < los < -23.5 and sign > 0
    roles = assign_roles(s, teams, "KC", los, sign, yc)
    assert all(roles[p] == "OL" for p in (18, 17, 12, 3))
    assert roles[11] == "TE" and roles[9] == "WR"
    assert roles[5] == "RB" and roles[8] == "QB" and roles[16] == "QB"
    assert all(roles[p] == "DL" for p in (4, 13, 1))
    assert roles[15] == "LB" and all(roles[p] == "LB" for p in (7, 14))
    assert all(roles[p] == "DB" for p in (6, 21, 0, 2, 27, 10))


def test_only_unnamed_ids_take_the_build():
    class Ident:
        def __init__(self, player, h=1.85, w=0.0):
            self.player, self.height_m, self.weight_lb = player, h, w
    merged = {4: Ident("P4"), 5: Ident("Isiah Pacheco", 1.78, 215.0)}
    n = apply_role_builds(merged, {4: "DL", 5: "RB"})
    assert n == 1
    assert abs(merged[4].height_m - POSITION_BUILDS["DL"][0]) < 1e-9
    assert abs(merged[4].weight_lb * 0.4536 - POSITION_BUILDS["DL"][1]) < 0.1
    assert merged[5].height_m == 1.78 and merged[5].weight_lb == 215.0


def test_inheritance_needs_a_clear_predecessor():
    from nfl_gsplat.identity.roles import inherit_roles
    roles = {1: "DL", 2: "DB", 3: "OL"}
    teams = {1: "BAL", 2: "BAL", 3: "KC", 10: "BAL", 11: "BAL", 12: "KC"}
    spans = {1: (0, 300), 2: (0, 310), 3: (0, 300), 10: (330, 500), 11: (340, 500), 12: (600, 650)}
    ends = {1: ((0, 0), (10.0, 0.0)), 2: ((0, 0), (30.0, 5.0)), 3: ((0, 0), (12.0, 1.0)),
            10: ((10.5, 0.3), (0, 0)),      # continues 1 (2 is 20 m away)
            11: ((11.0, 0.5), (0, 0)),      # 1 already claimed, 2 too far: nothing
            12: ((12.0, 1.0), (0, 0))}      # KC, but 300 frames later: nothing
    got = inherit_roles(roles, teams, spans, ends)
    assert got == {10: ("DL", 1)}
    # two predecessors within a metre of each other: ambiguous, nothing inherited
    roles2 = {1: "DL", 2: "LB"}
    ends2 = {1: ((0, 0), (10.0, 0.0)), 2: ((0, 0), (10.5, 0.0)), 10: ((10.2, 0.1), (0, 0))}
    spans2 = {1: (0, 300), 2: (0, 300), 10: (320, 400)}
    assert inherit_roles(roles2, {1: "BAL", 2: "BAL", 10: "BAL"}, spans2, ends2) == {}


def test_snap_is_where_the_bodies_start_moving_together():
    ground = {}
    rng = np.random.default_rng(0)
    for f in range(0, 300):
        d = {}
        for pid in range(20):
            x0, y0 = -25.0 + (pid % 5), -8.0 + 4 * (pid // 5)
            if f < 200:
                d[pid] = np.array([x0, y0]) + rng.normal(0, 0.01, 2)
            else:
                d[pid] = np.array([x0 + 0.1 * (f - 200), y0]) + rng.normal(0, 0.01, 2)
        ground[f] = d
    snap = snap_frame(ground, fps=60.0)
    assert snap is not None and 190 <= snap <= 205, snap


def test_the_line_of_scrimmage_survives_a_formation_with_no_stances():
    s, teams = formation()
    upright = {pid: (x, y, 2.4, n) for pid, (x, y, a, n) in s.items()}   # nobody in a stance
    los, sign, yc = line_of_scrimmage(upright, teams, "KC")
    assert sign > 0 and -25.5 < los < -22.5, (los, sign)
    roles = assign_roles(upright, teams, "KC", los, sign, yc)
    assert roles[9] == "WR" and roles[5] in ("RB", "QB")
    assert all(roles[p] == "DB" for p in (2, 27, 10))
