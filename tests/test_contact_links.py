import numpy as np
import pandas as pd

from nfl_gsplat.tracking import contact_links as cl


def _rows(pid, track, frames, x0, vx, y, w=40, h=100, team="T1", jersey=-1, cam="sideline"):
    out = []
    for i, f in enumerate(frames):
        cx = x0 + vx * i
        out.append(dict(frame=f, cam=cam, track_id=track, global_player_id=pid, bbox_x1=cx - w / 2, bbox_y1=y - h,
                        bbox_x2=cx + w / 2, bbox_y2=y, team=team, jersey_number_ocr=float(jersey)))
    return out


def _ground(df):
    # a stand-in for box_ground: the box bottom's pixel position scaled to metres
    return {(r.cam, int(r.frame), int(r.global_player_id)): np.array([(r.bbox_x1 + r.bbox_x2) / 200.0, r.bbox_y2 / 100.0])
            for r in df.itertuples()}


def test_receiver_case_is_found_with_contact_and_the_overlap_is_dropped_first():
    # id 9 runs right, dies at 93 on the defender 6's box; id 77 is born at 89 on 9's path; same team
    rows = _rows(9, 9, range(0, 94), 100, 4.0, 300)
    rows += _rows(6, 6, range(0, 120), 460, 0.0, 300, team="T0")            # standing where 9 arrives
    rows += _rows(77, 77, range(89, 130), 100 + 4.0 * 89, 4.0, 300)
    df = pd.DataFrame(rows)
    teams = {9: "KC", 6: "BAL", 77: "KC"}
    links = cl.find_links(df, _ground(df), teams, cam="sideline", lo=0, hi=130)
    best = links[0]
    assert (best.keep, best.drop, best.track) == (9, 77, 77) and best.reject is None
    assert best.overlap == 5 and best.death_last == 93 and best.birth_first == 89
    assert best.score >= cl.MIN_SCORE and any("contact" in r for r in best.reasons) and any("same team" in r for r in best.reasons)
    cmds = cl.fold_commands(best, "P")
    assert cmds[0][0].endswith("08za_drop_rows.py") and cmds[0][cmds[0].index("--frames") + 1:cmds[0].index("--frames") + 3] == ["89", "93"]
    assert cmds[1][0].endswith("08z_fold_ids.py") and "--track-id" in cmds[1]
    assert not any(l.drop == 6 or l.keep == 6 and l.drop == 77 and l.reject is None for l in links)   # the defender is not linked


def test_vetoes_teams_jerseys_and_the_other_camera():
    rows = _rows(1, 1, range(0, 50), 100, 4.0, 300, jersey=15)
    rows += _rows(2, 2, range(52, 100), 100 + 4.0 * 52, 4.0, 300, jersey=15)
    df = pd.DataFrame(rows)
    # different identity teams -> rejected
    links = cl.find_links(df, _ground(df), {1: "KC", 2: "BAL"}, cam="sideline", lo=0, hi=100)
    assert links and links[0].reject and "teams differ" in links[0].reject
    # same team, same jersey -> a link with the jersey counted
    links = cl.find_links(df, _ground(df), {1: "KC", 2: "KC"}, cam="sideline", lo=0, hi=100)
    assert links[0].reject is None and any("jersey 15" in r for r in links[0].reasons)
    # disagreeing jerseys -> rejected
    df2 = df.copy(); df2.loc[df2.global_player_id == 2, "jersey_number_ocr"] = 87.0
    links = cl.find_links(df2, _ground(df2), {1: "KC", 2: "KC"}, cam="sideline", lo=0, hi=100)
    assert links[0].reject and "jerseys differ" in links[0].reject
    # the other camera sees both ids at once, 5 m apart -> two men
    rows3 = rows + _rows(1, 11, range(40, 70), 100, 0.0, 300, cam="endzone") + _rows(2, 12, range(40, 70), 1100, 0.0, 300, cam="endzone")
    df3 = pd.DataFrame(rows3)
    links = cl.find_links(df3, _ground(df3), {1: "KC", 2: "KC"}, cam="sideline", lo=0, hi=100)
    assert links[0].reject and "apart" in links[0].reject
    # the other camera carries the dying id on to the birth's spot -> one man
    rows4 = rows + _rows(1, 11, range(40, 70), 100 + 4.0 * 40, 4.0, 300, cam="endzone")
    df4 = pd.DataFrame(rows4)
    links = cl.find_links(df4, _ground(df4), {1: "KC", 2: "KC"}, cam="sideline", lo=0, hi=100)
    assert links[0].reject is None and any("carries" in r for r in links[0].reasons)


def test_far_or_late_births_are_not_candidates():
    rows = _rows(1, 1, range(0, 50), 100, 4.0, 300)
    rows += _rows(2, 2, range(90, 120), 100 + 4.0 * 90, 4.0, 300)               # 40 frames later: past the gap
    rows += _rows(3, 3, range(52, 80), 100, 0.0, 300)                            # born back at the start, 2 m+ away
    df = pd.DataFrame(rows)
    links = cl.find_links(df, _ground(df), {1: "KC", 2: "KC", 3: "KC"}, cam="sideline", lo=0, hi=120)
    assert not any(l.drop == 2 for l in links)                                   # past the gap: not a candidate
    assert all(l.score < cl.MIN_SCORE for l in links)                            # the far birth is listed weak, never applied
