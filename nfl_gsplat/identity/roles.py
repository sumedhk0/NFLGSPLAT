"""A player's role from where he stands before the snap, and the build that role carries.

WHY. The render's body build comes from the roster (height sets one shape
coefficient, weight another through mesh volume), but only for the ids the
jersey OCR named -- about half of play 1's on-field ids. An unnamed id gets a
1.85 m default, so a defensive lineman and a cornerback come out the same
body although 45 kg apart (the user, 2026-09-10).

WHAT. Before the snap the formation says the role: crouched bodies on the
line of scrimmage are linemen, a body 5 m behind the centre is the back, a
body 10 m off the ball is a safety, a body split wide is a receiver or the
corner over him. Each role takes the KC/BAL 2024 rosters' median height and
weight for that position group. Named ids keep their roster row.

The pre-snap window is the static stretch before the snap (the first frame
after which fewer than STATIC_FRAC of the bodies stand still); crouch is the
sideline box's height over width (a stance is wider than it is tall).
"""
from __future__ import annotations

import numpy as np

STATIC_MPS: float = 0.5
MOVING_FRAC: float = 0.4          # the snap: fewer than this fraction of bodies still static ...
SNAP_HOLD: int = 12               # ... for this many frames
WINDOW_BEFORE: int = 130          # pre-snap window: [snap - WINDOW_BEFORE, snap - WINDOW_MARGIN]
WINDOW_MARGIN: int = 30
MIN_FRAMES: int = 30
CROUCH_ASPECT: float = 1.4        # sideline box height / width below this = a stance
LINE_DX_M: float = 2.5            # within this of the line of scrimmage = on the line
BOX_DX_M: float = 7.0             # linebackers and the back within this of the line
OL_DY_M: float = 5.5              # on the line within this of the formation's centre = interior lineman
TE_DY_M: float = 9.0              # a stance on the line beyond OL_DY_M and within this = tight end
QB_DY_M: float = 0.8              # a standing body in the backfield this close to the centre's line = passer
EDGE_DY_M: float = 9.0            # a standing defender on the line within this of centre = edge / walked-up backer
BOX_DY_M: float = 4.0             # second-level defender within this of centre = backer, outside it = nickel

# KC and BAL 2024 rosters, medians per position group (data/rosters/2024/rosters.parquet)
POSITION_BUILDS: dict[str, tuple[float, float]] = {
    "OL": (1.96, 143.0), "DL": (1.90, 134.0), "LB": (1.88, 107.0), "DB": (1.83, 91.0),
    "WR": (1.83, 89.0), "TE": (1.96, 111.0), "RB": (1.78, 98.0), "QB": (1.87, 96.0),
}


def position_builds(roster_df, teams) -> dict[str, tuple[float, float]]:
    """``{position: (height m, mass kg)}`` medians from a roster table (inches, lb)."""
    r = roster_df[roster_df["team"].isin(list(teams))].drop_duplicates("gsis_id")
    out = dict(POSITION_BUILDS)
    for pos, g in r.groupby("position"):
        h, w = g["height"].median(), g["weight"].median()
        if np.isfinite(h) and np.isfinite(w) and len(g) >= 3:
            out[str(pos)] = (round(float(h) * 0.0254, 2), round(float(w) * 0.4536, 1))
    return out


def speeds(ground: dict, *, fps: float, half: int = 6) -> dict:
    """``{frame: {pid: m/s}}`` from the ground positions over +-half frames."""
    out: dict = {}
    for f in sorted(ground):
        a, b = ground.get(f - half), ground.get(f + half)
        if a is None or b is None:
            continue
        for pid, _xy in ground[f].items():
            if pid in a and pid in b:
                out.setdefault(f, {})[pid] = float(np.linalg.norm(np.asarray(b[pid]) - np.asarray(a[pid])) / (2 * half) * fps)
    return out


def snap_frame(ground: dict, *, fps: float) -> int | None:
    """The first frame after which fewer than MOVING_FRAC of the bodies stand still for SNAP_HOLD
    frames (a receiver in motion does not count: one body). None when the play never starts."""
    sp = speeds(ground, fps=fps)
    frames = sorted(sp)
    static_frac = {f: np.mean([v < STATIC_MPS for v in sp[f].values()]) if sp[f] else 1.0 for f in frames}
    for i, f in enumerate(frames):
        if f < frames[0] + WINDOW_MARGIN:
            continue
        run = [g for g in frames[i:i + SNAP_HOLD]]
        if len(run) == SNAP_HOLD and all(static_frac[g] < MOVING_FRAC for g in run):
            return int(f)
    return None


def presnap_summary(ground: dict, boxes_df, window, *, cam: str = "sideline") -> dict:
    """``{pid: (x, y, aspect, n)}``: median ground position and sideline box aspect over the window."""
    f0, f1 = window
    sub = boxes_df[(boxes_df["cam"] == cam) & boxes_df["frame"].between(f0, f1) & (boxes_df["track_id"] >= 0)]
    aspect = ((sub["bbox_y2"] - sub["bbox_y1"]) / (sub["bbox_x2"] - sub["bbox_x1"]).clip(lower=1.0)).groupby(sub["track_id"]).median()
    out = {}
    for pid, a in aspect.items():
        pts = [np.asarray(ground[f][pid], float) for f in range(f0, f1 + 1) if f in ground and pid in ground[f]]
        if len(pts) < MIN_FRAMES:
            continue
        xy = np.median(np.stack(pts), axis=0)
        out[int(pid)] = (float(xy[0]), float(xy[1]), float(a), len(pts))
    return out


def line_of_scrimmage(summary: dict, teams: dict, offence: str):
    """``(los_x, sign, y_centre)``: the midpoint between the two teams' crouched lines, the
    sign that makes the offence's side positive, and the offence line's lateral centre."""
    off = [(x, y) for pid, (x, y, a, n) in summary.items() if teams.get(pid) == offence and a < CROUCH_ASPECT]
    de = [(x, y) for pid, (x, y, a, n) in summary.items() if teams.get(pid) not in (offence, None) and a < CROUCH_ASPECT]
    if len(off) < 3 or len(de) < 2:
        raise ValueError(f"too few crouched bodies to place the line of scrimmage (offence {len(off)}, defence {len(de)})")
    xo, xd = float(np.median([x for x, _ in off])), float(np.median([x for x, _ in de]))
    return 0.5 * (xo + xd), (1.0 if xo > xd else -1.0), float(np.median([y for _, y in off]))


def assign_roles(summary: dict, teams: dict, offence: str, los_x: float, sign: float, y_centre: float) -> dict:
    """``{pid: role}`` from the pre-snap summary."""
    roles = {}
    for pid, (x, y, a, n) in summary.items():
        team = teams.get(pid)
        if team is None:
            continue
        dx = (x - los_x) * sign               # positive on the offence's side
        dy = abs(y - y_centre)
        crouched = a < CROUCH_ASPECT
        if team == offence:
            if abs(dx) <= LINE_DX_M:
                # on the line: the five interior linemen (two-point or three-point), then the ends
                if dy <= OL_DY_M:
                    roles[pid] = "OL"
                elif dy <= TE_DY_M and crouched:
                    roles[pid] = "TE"
                else:
                    roles[pid] = "WR"
            elif dx <= BOX_DX_M and dy <= 3.0:
                # the backfield: the passer stands on the centre's line, the back is offset
                roles[pid] = "QB" if (dy <= QB_DY_M and not crouched) else "RB"
            elif dx > BOX_DX_M and dy <= 3.0:
                roles[pid] = "RB"
            else:
                roles[pid] = "WR"
        else:
            if dx >= -LINE_DX_M - 0.5:
                # on the line: a stance is a lineman, a standing body is an edge or a walked-up backer
                roles[pid] = "DL" if crouched else ("LB" if dy <= EDGE_DY_M else "DB")
            elif dx >= -BOX_DX_M:
                # the second level: backers in the box, nickels and apex defenders outside it
                roles[pid] = "LB" if dy <= BOX_DY_M else "DB"
            else:
                roles[pid] = "DB"
    return roles


def apply_role_builds(merged: dict, roles: dict, builds: dict | None = None) -> int:
    """Set height and weight on the UNNAMED identities (player 'P<id>') from their role's
    build; named ids keep the roster's. Returns how many changed."""
    import dataclasses

    builds = builds or POSITION_BUILDS
    n = 0
    for pid, role in roles.items():
        key = pid if pid in merged else str(pid)
        p = merged.get(key)
        if p is None or not str(getattr(p, "player", "")).startswith("P"):
            continue
        h, kg = builds.get(role, POSITION_BUILDS["DB"])
        h, lb = float(h), float(kg) / 0.4536
        if dataclasses.is_dataclass(p):
            merged[key] = dataclasses.replace(p, height_m=h, weight_lb=lb)
        else:
            p.height_m, p.weight_lb = h, lb
        n += 1
    return n
