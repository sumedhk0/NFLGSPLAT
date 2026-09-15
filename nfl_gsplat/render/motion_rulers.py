"""Plausibility rulers for a drawn timeline: the numbers behind "players teleport", "movement is
jittery", "a man appears in the wrong line".

Each ruler is a pure function over plain dicts so that a probe, the pipeline and the tests all measure the
same thing. Inputs are built from ``Timeline.states`` by :func:`positions_by_id` and, for the limbs, by the
caller's forward pass (the renderer's own, so the ruler reads what is drawn).

The rulers, with the reference values that make a number readable:

* **contiguous step** -- distance between an id's positions on consecutive frames, 59.94 fps. No player
  moves faster than 12 m/s, i.e. 0.20 m/frame; the smoother spreads a switch over its window, so a
  switch hides below 0.6 m and is read at ``STEP_M`` = 0.25. v38 of play 1 had 26 steps over 0.6 m.
* **root jitter** -- second difference of position over contiguous triples. A real player reads about
  0.003 m/frame^2 (10 m/s^2); v38's worst ids read 0.15-0.25, fifty times that.
* **census** -- |KC - 11| + |BAL - 11| per frame, meaned over a window. Blind to where a body stands, so
  it is the guard that a fix did not delete men, never the fix's own score.
* **joint jitter / speed** -- the same first and second differences on pelvis-relative joint positions,
  max over joints. A sprinting hand does 0.17 m/frame; 0.05 m/frame^2 is a visible twitch.

Every function returns numbers, never verdicts: thresholds are the caller's, quoted with their window.
"""
from __future__ import annotations

import numpy as np

FPS = 59.94
STEP_M = 0.25            # a contiguous step past this is a switch or a handover, not running
STEP_HARD_M = 0.6        # past this even the smoother cannot hide it (v38's 26)
TEAMS = ("KC", "BAL")


def positions_by_id(states_by_frame) -> dict:
    """``{pid: {frame: xy}}`` from ``Timeline.states`` (frame -> [PlayerState])."""
    out: dict = {}
    for f, states in states_by_frame.items():
        for s in states:
            out.setdefault(int(s.pid), {})[int(f)] = np.asarray(s.xy, float)
    return out


def views_by_id(states_by_frame) -> dict:
    """``{pid: {frame: views tuple}}`` -- which cameras stood behind each drawn frame."""
    out: dict = {}
    for f, states in states_by_frame.items():
        for s in states:
            out.setdefault(int(s.pid), {})[int(f)] = tuple(getattr(s, "views", ()))
    return out


def handover_steps(steps, views_of) -> list:
    """The subset of ``steps`` whose two frames were drawn from different camera sets: a body handed from
    the endzone alone to the sideline (or back) jumps by the two cameras' disagreement, p50 0.42 m on
    play 1's control pairs. Those are a placement offset, not a track switch, and want a different fix."""
    out = []
    for d, pid, f in steps:
        v = views_of.get(pid, {})
        if f in v and f + 1 in v and v[f] != v[f + 1]:
            out.append((d, pid, f))
    return out


def contiguous_steps(pos_by_id) -> list:
    """``[(step_m, pid, frame)]`` for every pair of CONSECUTIVE frames an id is drawn on, largest first.
    A pair across a gap is not a step: the id was not drawn between, so nothing moved on screen."""
    out = []
    for pid, byf in pos_by_id.items():
        fs = sorted(byf)
        for a, b in zip(fs[:-1], fs[1:]):
            if b - a == 1:
                out.append((float(np.linalg.norm(byf[b] - byf[a])), int(pid), int(a)))
    out.sort(reverse=True)
    return out


def second_differences(pos_by_id, *, min_count: int = 10) -> dict:
    """``{pid: array}`` of ||x[f+2] - 2 x[f+1] + x[f]|| over contiguous triples, ids with at least
    ``min_count`` of them. Works for any vector-valued series: root xy, or a joint."""
    out = {}
    for pid, byf in pos_by_id.items():
        fs = sorted(byf)
        js = [float(np.linalg.norm(byf[fs[i + 2]] - 2 * byf[fs[i + 1]] + byf[fs[i]]))
              for i in range(len(fs) - 2) if fs[i + 2] - fs[i] == 2]
        if len(js) >= min_count:
            out[int(pid)] = np.asarray(js)
    return out


def census_error(states_by_frame, team_of, lo: int, hi: int, *, teams=TEAMS, per_side: int = 11):
    """``(error, per_team_mean)`` over frames lo..hi: error = sum_team |drawn - 11|, meaned over frames.
    ``team_of``: pid -> team label (anything not in ``teams`` -- referees, None -- is not counted)."""
    counts = {t: [] for t in teams}
    for f, states in states_by_frame.items():
        if lo <= int(f) <= hi:
            for t in teams:
                counts[t].append(sum(1 for s in states if team_of.get(int(s.pid)) == t))
    if not counts[teams[0]]:
        return float("nan"), {t: float("nan") for t in teams}
    arrays = {t: np.asarray(v, float) for t, v in counts.items()}
    err = float(sum(np.abs(a - per_side) for a in arrays.values()).mean())
    return err, {t: float(a.mean()) for t, a in arrays.items()}


def joint_motion(joints_by_id) -> dict:
    """``{pid: (jitter, speed)}`` from ``{pid: {frame: joints[J, 3]}}`` (pelvis-relative): per contiguous
    triple the max over joints of the second difference, and of the first difference of its leading pair.
    Ids with fewer than 10 triples are skipped."""
    out = {}
    for pid, byf in joints_by_id.items():
        fs = sorted(byf)
        jit, spd = [], []
        for i in range(len(fs) - 2):
            if fs[i + 2] - fs[i] != 2:
                continue
            a, b, c = byf[fs[i]], byf[fs[i + 1]], byf[fs[i + 2]]
            jit.append(float(np.linalg.norm(c - 2 * b + a, axis=1).max()))
            spd.append(float(np.linalg.norm(b - a, axis=1).max()))
        if len(jit) >= 10:
            out[int(pid)] = (np.asarray(jit), np.asarray(spd))
    return out


HINGES = {"L_knee": (3, 0, +1.0), "R_knee": (4, 0, +1.0), "L_elbow": (17, 1, -1.0), "R_elbow": (18, 1, +1.0)}


def hinge_violations(body_poses, *, hyper_deg: float = -15.0, off_deg: float = 35.0) -> dict:
    """Shares of hinge-frames (four hinges per body-frame) bent the wrong way (flexion below
    ``hyper_deg``) or sideways (more than ``off_deg`` about the two non-hinge axes), from drawn
    ``body_poses [N, 21, 3]``. A knee does neither; play 1's fits did both on 3-14 % of frames."""
    bp = np.asarray(body_poses, float)
    if bp.ndim != 3 or not len(bp):
        return {"hyperextended": float("nan"), "off_axis": float("nan"), "n": 0}
    hyper = off = 0
    for j, ax, sign in HINGES.values():
        hyper += int((np.degrees(sign * bp[:, j, ax]) < hyper_deg).sum())
        others = [a for a in range(3) if a != ax]
        off += int((np.degrees(np.linalg.norm(bp[:, j][:, others], axis=1)) > off_deg).sum())
    n = 4 * len(bp)
    return {"hyperextended": hyper / n, "off_axis": off / n, "n": int(len(bp))}


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(pos_by_id, states_by_frame, team_of, *, lo: int, hi: int, joints_by_id=None,
              step_m: float = STEP_M, hard_m: float = STEP_HARD_M) -> dict:
    """One JSON-able report: the rulers above on the whole timeline and on the live window lo..hi.
    Worst lists are short on purpose -- they name the ids to look at, the arrays stay with the caller."""
    steps = contiguous_steps(pos_by_id)
    live = [s for s in steps if lo <= s[2] <= hi]
    views_of = views_by_id(states_by_frame)
    hand = {(p, f) for _d, p, f in handover_steps([s for s in steps if s[0] > step_m], views_of)}
    jit = second_differences(pos_by_id)
    all_j = np.concatenate(list(jit.values())) if jit else np.zeros(0)
    live_pos = {p: {f: xy for f, xy in byf.items() if lo <= f <= hi} for p, byf in pos_by_id.items()}
    jit_live = second_differences(live_pos)
    all_jl = np.concatenate(list(jit_live.values())) if jit_live else np.zeros(0)
    frames = sorted(int(f) for f in states_by_frame)
    err_live, teams_live = census_error(states_by_frame, team_of, lo, hi)
    err_full, teams_full = census_error(states_by_frame, team_of, frames[0], frames[-1]) if frames else (float("nan"), {})
    rep = {
        "window": [lo, hi],
        "ids_drawn": len(pos_by_id),
        "body_frames": int(sum(len(b) for b in pos_by_id.values())),
        "steps": {
            "full_over_step": sum(1 for s in steps if s[0] > step_m),
            "full_over_hard": sum(1 for s in steps if s[0] > hard_m),
            "live_over_step": sum(1 for s in live if s[0] > step_m),
            "live_over_hard": sum(1 for s in live if s[0] > hard_m),
            "full_handover": sum(1 for s in steps if s[0] > step_m and (s[1], s[2]) in hand),
            "live_handover": sum(1 for s in live if s[0] > step_m and (s[1], s[2]) in hand),
            "worst": [{"m": round(d, 3), "pid": p, "frame": f, "handover": (p, f) in hand,
                       "views": [list(views_of.get(p, {}).get(f, ())), list(views_of.get(p, {}).get(f + 1, ()))]}
                      for d, p, f in steps[:12]],
            "worst_live": [{"m": round(d, 3), "pid": p, "frame": f, "handover": (p, f) in hand,
                            "views": [list(views_of.get(p, {}).get(f, ())), list(views_of.get(p, {}).get(f + 1, ()))]}
                           for d, p, f in live[:24]],
        },
        "root_jitter": {
            "full": {"p50": _pct(all_j, 50), "p90": _pct(all_j, 90), "p99": _pct(all_j, 99)},
            "live": {"p50": _pct(all_jl, 50), "p90": _pct(all_jl, 90), "p99": _pct(all_jl, 99)},
            "worst_live": [{"pid": p, "p90": round(_pct(a, 90), 4), "n": int(len(a))}
                           for p, a in sorted(jit_live.items(), key=lambda kv: -_pct(kv[1], 90))[:8]],
        },
        "census": {"live": err_live, "live_teams": teams_live, "full": err_full, "full_teams": teams_full},
    }
    if joints_by_id is not None:
        jm = joint_motion(joints_by_id)
        allj = np.concatenate([v[0] for v in jm.values()]) if jm else np.zeros(0)
        alls = np.concatenate([v[1] for v in jm.values()]) if jm else np.zeros(0)
        rep["joints"] = {
            "ids": len(jm),
            "jitter": {"p50": _pct(allj, 50), "p90": _pct(allj, 90), "p99": _pct(allj, 99)},
            "speed": {"p50": _pct(alls, 50), "p90": _pct(alls, 90), "p99": _pct(alls, 99)},
            "worst": [{"pid": p, "jitter_p90": round(_pct(v[0], 90), 4), "speed_p90": round(_pct(v[1], 90), 4)}
                      for p, v in sorted(jm.items(), key=lambda kv: -_pct(kv[1][0], 90))[:8]],
        }
    return rep
