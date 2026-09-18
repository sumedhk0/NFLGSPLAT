"""08x_play_end: the snap from the bodies' motion, the carrier by travel, the stop frame."""
import importlib.util
from pathlib import Path

import numpy as np


def _load():
    spec = importlib.util.spec_from_file_location("play_end", Path(__file__).resolve().parents[1] / "scripts" / "08x_play_end.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _play(snap=100, n_bodies=22, motion_man=True, t_end=300):
    """Bodies standing until ``snap``, then all running at 0.08 m/frame in random directions; one man
    in motion at 0.05 m/frame from frame 40; two fragments gliding at 0.04 m/frame pre-snap."""
    rng = np.random.default_rng(0)
    pos = {}
    for pid in range(n_bodies):
        d = rng.normal(size=2); d /= np.linalg.norm(d)
        byf = {}
        x = rng.uniform(-5, 5, size=2)
        for f in range(t_end):
            v = 0.0
            if f >= snap:
                v = 0.08
            elif pid == 0 and motion_man and f >= 40:
                v = 0.05
            elif pid in (1, 2) and f < snap:
                v = 0.04
            x = x + v * d + rng.normal(scale=0.002, size=2)      # the placement's own jitter
            byf[f] = x.copy()
        pos[pid] = byf
    return pos


def test_snap_is_found_where_the_whole_line_fires_not_where_the_motion_man_or_the_gliders_move():
    m = _load()
    pos = _play(snap=100)
    snap = m.snap_from_motion(pos)
    assert snap is not None and 92 <= snap <= 100
    assert m.snap_from_motion(_play(snap=100, motion_man=False)) == snap
    # nobody ever runs: no snap
    still = {pid: {f: xy for f, xy in byf.items() if f < 90} for pid, byf in _play(snap=100).items()}
    assert m.snap_from_motion(still) is None
    # too few bodies drawn: the frames do not vote
    few = {pid: byf for pid, byf in pos.items() if pid < 8}
    assert m.snap_from_motion(few) is None


def test_moving_share_counts_only_bodies_drawn_on_both_sides():
    m = _load()
    pos = {1: {0: (0.0, 0.0), 2: (0.1, 0.0)}, 2: {0: (5.0, 0.0), 2: (5.0, 0.0)}, 3: {0: (1.0, 1.0)}}
    share, n = m.moving_share(pos, 1)
    assert n == 2 and abs(share - 0.5) < 1e-9


def test_carrier_by_travel_and_stop_frame():
    m = _load()
    byf_run = {f: np.array([0.1 * min(f, 60), 0.0]) for f in range(0, 100)}          # runs 6 m then stands
    byf_still = {f: np.array([3.0, 3.0]) for f in range(0, 100)}
    got = m.carrier_by_travel({7: byf_run, 8: byf_still}, 0, 90)
    assert got[0] == 7 and abs(got[1] - 6.0) < 1e-9
    assert m.stop_frame(byf_run, 0) == 61
    assert m.stop_frame(byf_still, 0) == 1


def test_dead_ball_from_the_balls_ground_frame_after_the_release_only():
    m = _load()
    frames = {str(f): {"src": s} for f, s in [(300, "centre"), (393, "snap"), (400, "passer"), (527, "flight"),
                                              (583, "carried"), (639, "ground"), (640, "ground")]}
    assert m.dead_from_ball({"release": 527, "frames": frames}) == 639
    # a "ground" frame before the release never ends the play
    frames["100"] = {"src": "ground"}
    assert m.dead_from_ball({"release": 527, "frames": frames}) == 639
    assert m.dead_from_ball({"release": 527, "frames": {"600": {"src": "carried"}}}) is None
    assert m.dead_from_ball({}) is None


def test_dead_ball_where_the_crowd_stops_and_the_clip_end_when_it_never_does():
    m = _load()
    rng = np.random.default_rng(1)
    def crowd(stop):
        pos = {}
        for pid in range(22):
            d = rng.normal(size=2); d /= np.linalg.norm(d)
            x = rng.uniform(-5, 5, size=2); byf = {}
            for f in range(400):
                v = 0.08 if 100 <= f < stop else 0.0
                x = x + v * d + rng.normal(scale=0.002, size=2)
                byf[f] = x.copy()
            pos[pid] = byf
        return pos
    dead = m.dead_from_motion(crowd(stop=300), 100)
    assert dead is not None and 298 <= dead <= 304
    assert m.dead_from_motion(crowd(stop=10_000), 100) is None          # runs to the last frame
    assert m.dead_from_motion(crowd(stop=120), 100) is not None          # ... but never before snap + 60
    assert m.dead_from_motion(crowd(stop=120), 100) >= 160
