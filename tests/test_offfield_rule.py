"""offfield_rule: sideline dwellers by position; the stripe signature on synthetic torsos."""
import numpy as np

from nfl_gsplat.render.offfield_rule import sideline_dwellers, stripe_stats


def test_sideline_dwellers_by_share_of_frames_beyond_the_line():
    ground = {}
    for f in range(20):
        ground[f] = {1: (0.0, 25.0),                     # staff at the boundary every frame
                     2: (0.0, 5.0),                      # a player mid-field
                     3: (0.0, 25.0 if f < 8 else 10.0),  # steps in-bounds at frame 8: 40 % beyond
                     4: (0.0, -24.0)}                    # the far sideline counts too
    assert sideline_dwellers(ground) == {1, 4}
    assert sideline_dwellers(ground, frac=0.3) == {1, 3, 4}


def test_stripe_stats_tell_vertical_stripes_from_a_flat_or_horizontal_torso():
    stripes = np.zeros((40, 30, 3), np.uint8)
    stripes[:, ::6] = 255
    stripes[:, 1::6] = 255
    stripes[:, 2::6] = 255                              # 3 white, 3 black columns
    flat = np.full((40, 30, 3), 200, np.uint8)
    bands = np.zeros((40, 30, 3), np.uint8)
    bands[::6] = 255
    bands[1::6] = 255
    bands[2::6] = 255                                   # horizontal bands
    r_s, d_s = stripe_stats(stripes)
    r_f, d_f = stripe_stats(flat)
    r_b, d_b = stripe_stats(bands)
    assert r_s > 2.5 and d_s >= 0.25
    assert not (r_f > 2.5 and d_f >= 0.25)
    assert r_b < 1.0


def test_bodies_behind_the_offence_are_officials():
    from nfl_gsplat.render.offfield_rule import behind_the_offence
    # the offence stands at larger x than the defence: sign +1, line at -24
    ground = {}
    for f in range(60):
        ground[f] = {
            1: (-22.0, 1.0),      # a lineman on the line
            2: (-17.0, -1.0),     # the passer, 7 m behind it
            3: (-9.5, 5.0),       # the referee, 14.5 m behind it
            4: (-30.0, 2.0),      # a defender, well the other side
        }
    got = behind_the_offence(ground, -24.0, 1.0)
    assert got == {3}
    assert behind_the_offence(ground, -24.0, 1.0, ids={1, 2}) == set()   # the rule can be limited
