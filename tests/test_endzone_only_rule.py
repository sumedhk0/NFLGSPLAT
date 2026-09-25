"""endzone_only_rule: endzone-only ids leave the timeline; two-view and sideline ids stay."""
import pandas as pd

from nfl_gsplat.render.endzone_only_rule import endzone_only_ids


def _df(rows):
    return pd.DataFrame(rows, columns=["frame", "cam", "global_player_id"])


def test_endzone_only_ids_are_dropped_and_two_view_ids_kept():
    df = _df([
        (0, "sideline", 1), (1, "sideline", 1),                       # sideline-only: kept
        (0, "endzone", 2), (1, "endzone", 2),                         # endzone-only: dropped
        (0, "sideline", 3), (0, "endzone", 3), (1, "endzone", 3),     # two-view once: kept
        (0, "endzone", -1),                                           # unlinked: ignored
    ])
    views = {0: {1: ["sideline"], 2: ["endzone"], 3: ["sideline", "endzone"]},
             1: {1: ["sideline"], 2: ["endzone"], 3: ["endzone"]}}
    assert endzone_only_ids(df, views) == {2}


def test_the_camera_name_is_a_parameter():
    df = _df([(0, "sideline", 5), (0, "endzone", 6)])
    views = {0: {5: ["sideline"], 6: ["endzone"]}}
    assert endzone_only_ids(df, views, cam="sideline") == {5}


def test_frustum_aware_keeps_an_endzone_only_id_the_sideline_cannot_see():
    import numpy as np

    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    n = 4
    track = CameraTrack(K=np.stack([K] * n), R=np.stack([R] * n), t=np.stack([t] * n), conf=np.ones(n),
                        width=1920, height=1080)
    df = _df([(f, "endzone", 7) for f in range(n)] + [(f, "endzone", 8) for f in range(n)])
    views = {f: {7: ["endzone"], 8: ["endzone"]} for f in range(n)}
    ground = {f: {7: (0.0, 0.0), 8: (40.0, 0.0)} for f in range(n)}     # 7 at the aim point, 8 forty metres off-axis
    assert endzone_only_ids(df, views, ground=ground, sideline=track) == {7}
    assert endzone_only_ids(df, views) == {7, 8}                         # no cameras: the old behaviour


def test_beyond_sideline_span_drops_the_endzone_tail_the_sideline_could_see():
    import numpy as np
    import pandas as pd

    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    # id 1: sideline frames 10..20, endzone frames 0..60; id 2: sideline only, 0..60.
    # Both ids, as the real table carries them: the span is keyed by the PLAYER, because `ground` is.
    rows = [{"cam": "sideline", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(10, 21)]
    rows += [{"cam": "endzone", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(0, 61)]
    rows += [{"cam": "sideline", "track_id": 2, "global_player_id": 2, "frame": f} for f in range(0, 61)]
    df = pd.DataFrame(rows)
    ground = {f: {1: np.array([5.0, 0.0]), 2: np.array([-5.0, 0.0])} for f in range(0, 61)}
    # far downfield from frame 50 on: outside the sideline image
    for f in range(50, 61):
        ground[f][1] = np.array([80.0, 0.0])
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    n = 61
    track = CameraTrack(K=np.stack([K] * n), R=np.stack([R] * n), t=np.stack([t] * n), conf=np.ones(n),
                        width=1920, height=1080)                     # sees the field around x = 0
    out, dropped = beyond_sideline_span(ground, df, track, gap=5)
    for f in range(0, 61):
        assert 2 in out[f]                                            # the sideline's own id: untouched
        if 5 <= f <= 25:
            assert 1 in out[f], f                                     # the span plus the gap
        elif f >= 50:
            assert 1 in out[f], f                                     # beyond the span, unseen: kept
        else:
            assert 1 not in out[f], f                                 # beyond the span, in view: dropped
    assert dropped == 5 + (50 - 26)


def test_beyond_the_span_is_kept_when_the_sideline_has_nobody_there():
    """Past its sideline span an id is a second copy only if the sideline draws that man."""
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    class _Side:
        # a camera at the origin looking down +y; everything in front of it is inside the image
        K = [np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])] * 400
        R = [np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])] * 400
        t = [np.zeros(3)] * 400
        conf = np.ones(400)
        width, height = 1920, 1080

    df = pd.DataFrame({"cam": ["sideline"] * 3, "frame": [10, 11, 12], "track_id": [1, 1, 1],
                       "global_player_id": [1, 1, 1]})
    ground = {300: {1: np.array([0.5, 40.0])}}                  # id 1, long past its sideline span
    side = {300: {2: np.array([0.6, 40.1])}}                    # ... and the sideline draws that man as id 2
    out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side)
    assert dropped == 1 and out[300] == {}
    side_far = {300: {2: np.array([9.0, 40.0])}}                # nobody near: the sideline lost him
    out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side_far)
    assert dropped == 0 and 1 in out[300]


def test_beyond_sideline_span_hold_drops_the_endzone_lead_in_that_stands_far_from_the_join():
    """Two ids the sideline first sees at frame 40 and the endzone from 0: within the 30-frame gap
    both are drawn from the endzone alone. Id 1's endzone point stands 2.4 m from where the sideline
    then has him (a ghost that would glide 2.4 m into the man); id 2's stands 0.3 m off (held)."""
    import numpy as np
    import pandas as pd

    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    rows = [{"cam": "sideline", "track_id": p, "global_player_id": p, "frame": f} for p in (1, 2) for f in range(40, 61)]
    rows += [{"cam": "endzone", "track_id": p, "global_player_id": p, "frame": f} for p in (1, 2) for f in range(0, 61)]
    df = pd.DataFrame(rows)
    ground = {f: {1: np.array([2.4, 0.0]) if f < 40 else np.array([0.0, 0.0]),
                  2: np.array([10.3, 0.0]) if f < 40 else np.array([10.0, 0.0])} for f in range(10, 61)}
    side_ground = {f: {1: np.array([0.0, 0.0]), 2: np.array([10.0, 0.0])} for f in range(40, 61)}
    rep = {}
    out, dropped = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8, report=rep)
    assert all(1 not in out[f] for f in range(10, 40)) and all(1 in out[f] for f in range(40, 61))
    assert all(2 in out[f] for f in range(10, 61))
    assert dropped == 30
    assert rep[1][:3] == (30, 2.4, 30) and rep[2][0] == 30 and abs(rep[2][1] - 0.3) < 1e-9 and rep[2][2] == 0
    assert abs(rep[1][3] - 2.4) < 1e-9 and np.isnan(rep[1][4])                       # a lead-in only: no tail jump
    # a man who runs 3 m during his lead-in but joins the sideline's point within 0.2 m is held whole
    ground3 = {f: {3: np.array([20.0 + 0.1 * (f - 10), 0.0])} for f in range(10, 61)}
    side3 = {f: {3: np.array([23.1, 0.0])} for f in range(40, 61)}
    df3 = pd.DataFrame([{"cam": "sideline", "track_id": 3, "global_player_id": 3, "frame": f} for f in range(40, 61)]
                       + [{"cam": "endzone", "track_id": 3, "global_player_id": 3, "frame": f} for f in range(0, 61)])
    rep3 = {}
    out3, d3 = beyond_sideline_span(ground3, df3, None, gap=30, side_ground=side3, hold_m=0.8, report=rep3)
    assert d3 == 0 and all(3 in out3[f] for f in range(10, 61)) and abs(rep3[3][1] - 0.2) < 1e-9
    # hold off: the old behaviour, every lead-in frame drawn
    out2, d2 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=None)
    assert d2 == 0 and all(1 in out2[f] for f in range(10, 61))


def test_beyond_sideline_span_before_the_snap_every_frame_is_held_to_the_join_point():
    """A ghost that slides onto its man: 2.4 m off at the start of the lead-in, 0.3 m at the join.
    The join test alone keeps it; with the snap given, the frames before the snap are measured one
    by one and the far ones go, the near ones stay."""
    import numpy as np
    import pandas as pd

    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    df = pd.DataFrame([{"cam": "sideline", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(40, 61)]
                      + [{"cam": "endzone", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(0, 61)])
    ground = {f: {1: np.array([2.4 - 2.1 * (f - 10) / 29.0, 0.0]) if f < 40 else np.array([0.0, 0.0])} for f in range(10, 61)}
    side_ground = {f: {1: np.array([0.0, 0.0])} for f in range(40, 61)}
    out, d = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8)
    assert d == 0                                                       # join jump 0.3: kept whole
    out2, d2 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8, snap=45)   # the join at 40 is at the snap
    far = [f for f in range(10, 40) if 2.4 - 2.1 * (f - 10) / 29.0 > 0.8]
    assert d2 == len(far) and all(1 not in out2[f] for f in far) and all(1 in out2[f] for f in range(10, 61) if f not in far)
    # a join deep in the play (snap 20 frames before it) says nothing about the set man: nothing dropped
    out5, d5 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8, snap=20, presnap_join_max=10)
    assert d5 == 0
    # with the snap inside the lead-in, only the frames before it are held per frame
    out3, d3 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8, snap=20, presnap_join_max=30)
    assert d3 == len([f for f in far if f < 20])
    # "hold": the far pre-snap frames stay, moved to where the sideline first has the man
    out4, d4 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side_ground, hold_m=0.8, snap=45, presnap="hold")
    assert d4 == 0 and all(1 in out4[f] for f in range(10, 61)) and all(np.allclose(out4[f][1], [0.0, 0.0]) for f in far)
    assert all(np.allclose(out4[f][1], ground[f][1]) for f in range(10, 40) if f not in far)


def test_hold_holes_moves_an_endzone_filled_hole_onto_the_sideline_line_only_when_far():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import hold_holes

    # sideline sees id 1 at 0..10 and 16..30 walking +x at 0.1 m/frame; the endzone fills 11..15,
    # two metres off the line at 12-13 and 0.3 m off at 14
    side = {f: {1: np.array([0.1 * f, 0.0])} for f in list(range(0, 11)) + list(range(16, 31))}
    ground = {f: dict(d) for f, d in side.items()}
    for f in range(11, 16):
        ground[f] = {1: np.array([0.1 * f, 2.0 if f in (12, 13) else 0.3])}
    out, moved = hold_holes(ground, side, hold_m=0.8)
    assert len(moved) == 2 and all(abs(m - 2.0) < 1e-9 for m in moved)
    assert np.allclose(out[12][1], [1.2, 0.0]) and np.allclose(out[13][1], [1.3, 0.0])
    assert np.allclose(out[14][1], [1.4, 0.3]) and np.allclose(out[11][1], ground[11][1])
    # a long hole: the middle is left alone, the frames within reach of an end follow that end at
    # the sideline's own velocity (0.1 m/frame here), and the off switch holds nothing
    side2 = {f: {1: np.array([0.1 * f, 0.0])} for f in list(range(0, 11)) + list(range(40, 51))}
    ground2 = {f: dict(d) for f, d in side2.items()}
    for f in (12, 20, 45 - 5 - 3):          # 12 near the start, 20 in the middle, 37 near the end
        ground2[f] = {1: np.array([0.1 * f, 3.0])}
    out2, moved2 = hold_holes(ground2, side2, hold_m=0.8)
    held2 = [m for m in moved2 if np.isfinite(m)]
    assert len(held2) == 2 and np.allclose(out2[12][1], [1.2, 0.0]) and np.allclose(out2[37][1], [3.7, 0.0])
    assert 1 not in out2[20] and sum(1 for m in moved2 if not np.isfinite(m)) == 1     # the middle frame is left out
    assert hold_holes(ground, side, hold_m=None)[1] == []


def test_formation_hold_fills_a_set_man_and_leaves_a_moving_one():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import formation_hold

    # id 1 set at (5, 0) seen on 10 scattered pre-snap frames; id 2 in motion across the field; snap 100, clip from 20
    side = {}
    for f in (25, 30, 41, 50, 55, 61, 70, 77, 80, 85):
        side.setdefault(f, {})[1] = np.array([5.0, 0.0]) + np.random.default_rng(f).normal(scale=0.05, size=2)
    for f in range(20, 91):
        side.setdefault(f, {})[2] = np.array([0.0, 0.1 * f])
    ground = {f: dict(d) for f, d in side.items()}
    out, added = formation_hold(ground, side, start=20, snap=100, still_m=0.3)
    assert set(added) == {1} and added[1] == (90 - 20 + 1) - 10
    assert all(1 in out[f] for f in range(20, 91)) and np.allclose(out[24][1], np.median([side[f][1] for f in side if 1 in side[f]], axis=0))
    assert all(2 in out[f] for f in range(20, 91)) and 2 not in out.get(95, {})
    assert formation_hold(ground, side, start=20, snap=100, still_m=None)[1] == {}
    # a spot another sideline body occupies is not filled: id 3 set at (0, 2) is on id 2's path there
    side3 = {f: dict(d) for f, d in side.items()}
    for f in (25, 30, 41, 50, 55, 61):
        side3[f][3] = np.array([0.0, 2.0])
    out3, added3 = formation_hold({f: dict(d) for f, d in side3.items()}, side3, start=20, snap=100, still_m=0.3, empty_m=0.8)
    assert added3.get(3, 0) < (90 - 20 + 1) - 6 and all(3 not in out3[f] for f in range(20, 27) if abs(0.1 * f - 2.0) < 0.8 and 3 not in side3.get(f, {}))


def test_formation_hold_only_ids_restricts_the_hold():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import formation_hold

    side = {}
    for f in (25, 30, 41, 50, 55):
        side.setdefault(f, {})[1] = np.array([5.0, 0.0])
        side.setdefault(f, {})[2] = np.array([9.0, 0.0])
    ground = {f: dict(d) for f, d in side.items()}
    out, added = formation_hold(ground, side, start=20, snap=100, still_m=0.5, min_frames=3, only_ids={1})
    assert set(added) == {1} and all(2 not in out[f] for f in range(20, 91) if f not in side)


def test_qb_hold_holds_the_man_who_steps_back_from_behind_the_centre():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import qb_hold

    # centre at (-23, 0); id 80 first seen at 377, 0.8 m behind him (offence side +x); id 38 first seen at 375 but 1.9 m across
    side = {377: {80: np.array([-22.2, 0.1])}, 375: {38: np.array([-22.8, 1.9])}}
    ground = {f: {} for f in range(213, 400)}
    ground[377][80] = np.array([-22.2, 0.1])
    out, pid, n = qb_hold(ground, side, start=213, snap=393, centre_xy=(-23.0, 0.0), sign=1.0, team_ids={80, 38}, first_frame={80: 377, 38: 375})
    assert pid == 80 and n == 377 - 213 and np.allclose(out[300][80], [-22.2, 0.1]) and 38 not in out[300]
    # nobody steps back: nothing held
    assert qb_hold(ground, {375: {38: np.array([-22.8, 1.9])}}, start=213, snap=393, centre_xy=(-23.0, 0.0), sign=1.0, team_ids={38}, first_frame={38: 375})[1] is None


def test_beyond_sideline_span_gap_same_body_drops_a_lead_in_on_another_mans_spot():
    import numpy as np
    import pandas as pd

    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    # id 1's sideline span 40..60, endzone from 10; in the gap (10..39) it stands 0.5 m from id 2, whom the sideline draws
    df = pd.DataFrame([{"cam": "sideline", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(40, 61)]
                      + [{"cam": "endzone", "track_id": 1, "global_player_id": 1, "frame": f} for f in range(0, 61)]
                      + [{"cam": "sideline", "track_id": 2, "global_player_id": 2, "frame": f} for f in range(0, 61)])
    ground = {f: {1: np.array([0.5, 0.0]), 2: np.array([0.0, 0.0])} for f in range(10, 61)}
    side = {f: {2: np.array([0.0, 0.0])} for f in range(0, 61)}
    for f in range(40, 61):
        side[f][1] = np.array([0.5, 0.0])
    out, d = beyond_sideline_span(ground, df, None, gap=30, side_ground=side, hold_m=None, same_body_gap_m=0.8)
    assert d == 30 and all(1 not in out[f] for f in range(10, 40)) and all(1 in out[f] for f in range(40, 61))
    out2, d2 = beyond_sideline_span(ground, df, None, gap=30, side_ground=side, hold_m=None, same_body_gap_m=None)
    assert d2 == 0


def test_line_vouch_keeps_the_hidden_lineman_and_not_the_endzone_copy_of_a_drawn_man():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import line_vouch

    los_x, sign = -24.0, 1.0
    teams = {17: "KC", 38: "KC", 66: "KC", 9: "BAL"}
    ground, views, side = {}, {}, {}
    for f in range(100, 130):
        # 17 the centre (both views) at across 0; 38 an endzone-only guard 1.3 m across; 66 an endzone copy
        # of the centre 0.2 m across from him; 9 a linebacker of the other team 4 m off the line
        ground[f] = {17: np.array([-23.0, 0.0]), 38: np.array([-23.1, 1.3]), 66: np.array([-22.6, 0.2]),
                     9: np.array([-28.0, 1.4])}
        views[f] = {17: ("endzone", "sideline"), 38: ("endzone",), 66: ("endzone",), 9: ("endzone",)}
        side[f] = {17: np.array([-23.0, 0.1])}
    keep, counts = line_vouch(ground, views, side, start=100, snap=140, teams=teams, los_x=los_x, sign=sign, across_m=0.7)
    assert counts == {38: 30}                       # 66 shares the centre's across position; 9 is off the line
    assert all(keep[f] == {38} for f in range(100, 130))
    # off the line (5 m back) nothing is vouched; None switches the rule off
    ground2 = {f: {38: np.array([-18.0, 1.3])} for f in range(100, 130)}
    assert line_vouch(ground2, views, side, start=100, snap=140, teams=teams, los_x=los_x, sign=sign, across_m=0.7)[1] == {}
    assert line_vouch(ground, views, side, start=100, snap=140, teams=teams, los_x=los_x, sign=sign, across_m=None) == ({}, {})


def test_qb_hold_takes_out_the_teammate_already_standing_on_the_held_spot():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import qb_hold

    # the centre 17 on the line, 204 (the quarterback under another id) 0.6 m behind him, a guard 19 beside him
    ground = {f: {17: np.array([-23.0, 0.0]), 204: np.array([-22.4, 0.3]), 19: np.array([-22.9, -0.7])} for f in range(213, 300)}
    ground.update({f: {17: np.array([-23.0, 0.0]), 19: np.array([-22.9, -0.7])} for f in range(300, 377)})
    side = {377: {80: np.array([-22.2, 0.1])}}
    removed = {}
    out, pid, n = qb_hold(ground, side, start=213, snap=393, centre_xy=(-23.0, 0.0), sign=1.0, team_ids={80, 204, 17, 19},
                          first_frame={80: 377}, removed=removed)
    assert pid == 80 and n == 377 - 213
    assert removed == {204: 87} and all(204 not in out[f] and 80 in out[f] for f in range(213, 300))
    assert all(17 in out[f] and 19 in out[f] for f in range(213, 377))      # the centre and the guard beside him stay
    # with the rule off the teammate stays too
    out2, _, _ = qb_hold(ground, side, start=213, snap=393, centre_xy=(-23.0, 0.0), sign=1.0, team_ids={80, 204, 17, 19},
                         first_frame={80: 377}, same_m=None)
    assert all(204 in out2[f] for f in range(213, 300))


def test_presnap_holes_are_filled_between_an_ids_points_only_where_the_spot_is_empty():
    import numpy as np

    from nfl_gsplat.render.endzone_only_rule import fill_presnap_holes

    teams = {82: "KC", 17: "KC", 19: "KC", 166: "KC", 9: "BAL"}
    ground = {f: {17: np.array([-23.0, 0.0]), 19: np.array([-22.6, -3.3])} for f in range(213, 400)}
    for f in [217, 218, 219, 260, 261, 300, 307]:                   # the guard, with holes
        ground[f][82] = np.array([-23.0, -1.0])
    for f in list(range(291, 320)) + list(range(340, 393)):           # a twin on 19's man, with a hole 320-339
        ground[f][166] = np.array([-22.6, -3.2])
    out, added = fill_presnap_holes(ground, start=213, snap=393, teams=teams)
    assert added == {82: 307 - 217 + 1 - 7}                         # every hole frame of the guard ...
    assert 166 not in added                                           # ... none of the twin's: 19 stands there
    assert 82 not in out[216] and 82 not in out[308]                 # nothing beyond the first or last sighting
    assert np.allclose(out[240][82], [-23.0, -1.0])
    # a vouched id is filled only between its vouched frames
    out2, added2 = fill_presnap_holes(ground, start=213, snap=393, teams=teams, only_ids={82}, frames_of={82: {260, 261, 300, 307}})
    assert added2 == {82: 307 - 260 + 1 - 4} and 82 not in out2[240]
    assert fill_presnap_holes({250: {5: np.array([0.0, 0.0])}}, start=213, snap=393, teams={5: "KC"})[1] == {}


def test_pocket_vouch_admits_the_rusher_the_sideline_cannot_see_and_not_a_ghost():
    import numpy as np

    from nfl_gsplat.render import endzone_only_rule as ezr

    los_x, sign = -24.0, 1.0                      # the offence on x > -24
    ground, views, side = {}, {}, {}
    for f in range(400, 430):
        ground[f] = {1: np.array([-21.0, 0.5]),   # a KC lineman, sideline-backed
                     2: np.array([-20.5, 2.5]),   # a BAL rusher the sideline draws
                     3: np.array([-19.8, -1.6]),  # a BAL rusher only the endzone sees, 2 m across from 2: hidden
                     4: np.array([-20.7, 2.4]),   # a BAL endzone-only copy of 2, on his across position: a ghost
                     5: np.array([-40.0, 5.0])}   # a BAL endzone-only body far downfield: not the pocket
        views[f] = {1: ("sideline", "endzone"), 2: ("sideline",), 3: ("endzone",), 4: ("endzone",), 5: ("endzone",)}
        side[f] = {1: ground[f][1], 2: ground[f][2]}
    teams = {1: "KC", 2: "BAL", 3: "BAL", 4: "BAL", 5: "BAL"}
    keep, counts = ezr.pocket_vouch(ground, views, side, snap=400, end=429, teams=teams, los_x=los_x, sign=sign, across_m=0.7)
    assert counts == {3: 30}
    assert all(keep[f] == {3} for f in range(400, 430))
    assert ezr.pocket_vouch(ground, views, side, snap=400, end=429, teams=teams, los_x=los_x, sign=sign, across_m=None) == ({}, {})


def test_beyond_the_span_two_endzone_boxes_apart_are_two_men():
    """A sideline man within SAME_BODY_M is the same man -- unless the endzone boxes both, apart, on the frame."""
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    class _Side:
        K = [np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])] * 400
        R = [np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])] * 400
        t = [np.zeros(3)] * 400
        conf = np.ones(400)
        width, height = 1920, 1080

    def frame(cam, f, tid, gid, x1, x2):
        return dict(cam=cam, frame=f, track_id=tid, global_player_id=gid, bbox_x1=x1, bbox_y1=100.0, bbox_x2=x2, bbox_y2=300.0)
    ground = {300: {1: np.array([0.5, 40.0])}}                  # id 1, long past its sideline span (10-12)
    side = {300: {2: np.array([0.6, 40.1])}}                    # the sideline draws a man 0.14 m away as id 2
    apart = pd.DataFrame([frame("sideline", 10, 1, 1, 0, 10), frame("sideline", 11, 1, 1, 0, 10), frame("sideline", 12, 1, 1, 0, 10),
                          frame("endzone", 300, 7, 1, 500.0, 600.0), frame("endzone", 300, 8, 2, 650.0, 750.0)])   # two boxes, no overlap
    out, dropped = beyond_sideline_span(ground, apart, _Side(), gap=30, side_ground=side, apart_iou=0.3)
    assert dropped == 0 and 1 in out[300]
    same = pd.DataFrame([frame("sideline", 10, 1, 1, 0, 10), frame("sideline", 11, 1, 1, 0, 10), frame("sideline", 12, 1, 1, 0, 10),
                         frame("endzone", 300, 7, 1, 500.0, 600.0), frame("endzone", 300, 8, 2, 510.0, 610.0)])    # one man, two boxes
    out, dropped = beyond_sideline_span(ground, same, _Side(), gap=30, side_ground=side, apart_iou=0.3)
    assert dropped == 1 and out[300] == {}
    out, dropped = beyond_sideline_span(ground, apart, _Side(), gap=30, side_ground=side, apart_iou=None)   # the test off: the old rule
    assert dropped == 1 and out[300] == {}


def test_beyond_the_span_same_body_is_same_team():
    """An endzone-only Raven 0.14 m from a KC sideline man is not that man's copy when teams are given."""
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    class _Side:
        K = [np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])] * 400
        R = [np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])] * 400
        t = [np.zeros(3)] * 400
        conf = np.ones(400)
        width, height = 1920, 1080

    df = pd.DataFrame({"cam": ["sideline"] * 3, "frame": [10, 11, 12], "track_id": [1, 1, 1], "global_player_id": [1, 1, 1]})
    ground = {300: {1: np.array([0.5, 40.0])}}
    side = {300: {2: np.array([0.6, 40.1])}}
    import nfl_gsplat.render.endzone_only_rule as ezr
    was = ezr.SAME_BODY_SAME_TEAM
    ezr.SAME_BODY_SAME_TEAM = True
    try:
        out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side, teams={1: "BAL", 2: "KC"})
        assert dropped == 0 and 1 in out[300]                                        # the other team: two men
    finally:
        ezr.SAME_BODY_SAME_TEAM = was
    ezr.SAME_BODY_SAME_TEAM = True
    try:
        out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side, teams={1: "BAL", 2: "BAL"})
        assert dropped == 1 and out[300] == {}                                       # the same team: a copy
        out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side, teams={1: "BAL"})
        assert dropped == 1 and out[300] == {}                                       # team unknown: the old rule
    finally:
        ezr.SAME_BODY_SAME_TEAM = was
    out, dropped = beyond_sideline_span(ground, df, _Side(), gap=30, side_ground=side, teams={1: "BAL", 2: "KC"})
    assert dropped == (0 if ezr.SAME_BODY_SAME_TEAM else 1)                          # the module default decides


def test_hold_holes_keeps_a_continuous_endzone_chain_inside_a_long_hole():
    """Inside a 40-frame hole the endzone's own points (a walk of 0.1 m a frame from the near end) are kept; a lone
    point 3 m off the chain is still removed as beyond reach."""
    import numpy as np
    from nfl_gsplat.render.endzone_only_rule import hold_holes

    side = {f: {7: np.array([0.0 + 0.0 * f, 0.0])} for f in list(range(100, 110)) + list(range(150, 160))}
    ground = {f: dict(d) for f, d in side.items()}
    for f in range(110, 135):                                                        # the chain from the near end
        ground[f] = {7: np.array([0.1 * (f - 109), 0.0])}
    ground[140] = {7: np.array([3.0, 5.0])}                                           # a stray point, off the chain
    out, moved = hold_holes(ground, side, hold_m=0.8, max_hole=17, reach=8, vel_frames=4, chain_step_m=0.6, chain_gap=5)
    assert all(7 in out[f] for f in range(110, 135))                                  # kept as they are
    assert abs(float(out[130][7][0]) - 2.1) < 1e-9
    assert 7 not in out[140]                                                          # beyond reach, not on the chain
    out2, _ = hold_holes(ground, side, hold_m=0.8, max_hole=17, reach=8, vel_frames=4, chain_step_m=None)
    assert 7 not in out2[125]                                                         # the chain off: the old rule


def test_beyond_the_span_same_body_test_is_for_short_spans():
    """A long-lived sideline id's endzone tail beside another sideline man is his own; a 3-frame fragment's is a copy."""
    import numpy as np
    import pandas as pd
    from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span

    class _Side:
        K = [np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])] * 400
        R = [np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])] * 400
        t = [np.zeros(3)] * 400
        conf = np.ones(400)
        width, height = 1920, 1080

    ground = {300: {1: np.array([0.5, 40.0])}}
    side = {300: {2: np.array([0.6, 40.1])}}
    short = pd.DataFrame({"cam": ["sideline"] * 3, "frame": [10, 11, 12], "track_id": [1] * 3, "global_player_id": [1] * 3})
    long = pd.DataFrame({"cam": ["sideline"] * 200, "frame": list(range(10, 210)), "track_id": [1] * 200, "global_player_id": [1] * 200})
    out, dropped = beyond_sideline_span(ground, short, _Side(), gap=30, side_ground=side, same_body_max_span=60)
    assert dropped == 1 and out[300] == {}                                           # a fragment: the copy
    out, dropped = beyond_sideline_span(ground, long, _Side(), gap=30, side_ground=side, same_body_max_span=60)
    assert dropped == 0 and 1 in out[300]                                            # 200 sideline frames: his own man
    out, dropped = beyond_sideline_span(ground, long, _Side(), gap=30, side_ground=side, same_body_max_span=None)
    assert dropped == 1 and out[300] == {}                                           # off: the old rule

def test_hole_chain_skips_a_point_off_the_chain_and_bounds_the_step_across_the_camera_ray():
    """A 40-frame hole walked at 0.1 m a frame along the endzone camera's ray (the camera on the +x side): one point
    1.3 m across at 120 ends the shipped chain (121-134 lost); skipping it with the across bound keeps 121-134 and
    not 120; 0.4 m of depth jitter along the ray stays on the chain; the flag off is the shipped result exactly."""
    import numpy as np
    from nfl_gsplat.render.endzone_only_rule import hole_chain, hold_holes

    side = {f: {7: np.array([0.0, 0.0])} for f in list(range(100, 110)) + list(range(150, 160))}
    ground = {f: dict(d) for f, d in side.items()}
    for f in range(110, 135):
        ground[f] = {7: np.array([0.1 * (f - 109), 0.0])}
    ground[120] = {7: np.array([1.1, 1.3])}                                           # the merged box beside him
    ground[126] = {7: np.array([0.1 * (126 - 109) + 0.4, 0.0])}                        # depth jitter along the ray
    kw = dict(hold_m=0.8, step_m=0.6, gap=5)
    shipped = hole_chain(ground, side, 7, 109, 150, **kw)
    assert shipped == set(range(110, 120))
    new = hole_chain(ground, side, 7, 109, 150, **kw, skip=True, across_m=0.2, speed_m=0.17, cam_xy=(100.0, 0.0))
    assert 120 not in new and set(range(121, 135)) <= new and set(range(110, 120)) <= new
    base = hold_holes(ground, side, hold_m=0.8, max_hole=17, reach=8, vel_frames=4, chain_step_m=0.6, chain_gap=5)
    off = hold_holes(ground, side, hold_m=0.8, max_hole=17, reach=8, vel_frames=4, chain_step_m=0.6, chain_gap=5,
                     chain_skip=False, chain_across_m=None, cam_xy=(100.0, 0.0))
    assert np.array_equal(np.asarray(base[1]), np.asarray(off[1]), equal_nan=True)
    assert all(set(base[0][f]) == set(off[0][f]) and all(np.allclose(base[0][f][q], off[0][f][q]) for q in base[0][f])
               for f in base[0])
    on = hold_holes(ground, side, hold_m=0.8, max_hole=17, reach=8, vel_frames=4, chain_step_m=0.6, chain_gap=5,
                    chain_skip=True, chain_across_m=0.2, chain_speed_m=0.17, cam_xy=(100.0, 0.0))
    assert all(7 in on[0][f] for f in range(121, 135)) and 7 not in base[0][130]
