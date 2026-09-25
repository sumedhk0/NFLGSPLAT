#!/usr/bin/env python
"""Learn which way a man moves relative to his chest from REAL tracking (measured orientation), then apply it to a play.

    C:/venvs/nflgsplat/Scripts/python scripts/09l_facing_prior.py --bdb data/tracking/input_2023_w01.csv \
        --joints J.json --play-dir P --roles roles.json [--model-out data/models/facing_prior.pt]

WHY. The per-play classifier trained on the play's own two-view fits did not generalise (09k: 37-42 % held out by
player against a 37 % majority class) and its labels were not physical (19 % "backward" above 5 m/s). The Big Data
Bowl tracking carries each player's measured orientation ``o`` (shoulder-pad sensors) and motion direction ``dir``
at 10 Hz: the relative class (forward / backward / sideways, pose.action_class) is MEASURED there.

WHAT. BDB input rows (snap to the throw; 13 players a play: receivers, backs, the passer, the coverage) become
pose.action_class features with the camera cues blank (the tracking has no film) -- speed, motion against the attack
direction, side, position group, time since the snap, depth behind the line, the bearing to the passer (the ball
before the throw), the nearest opponent -- and labels from ``o - dir``. Scored held out by PLAY (grouped folds)
against the majority class and a hand rule; trained on everything; applied to the play's drawn bodies from the snap
to the release for the position groups the tracking covers. Reports the agreement with the play's two-view labels
(noisy, a sanity check) and lists the one-view stretches where a confident prediction disagrees with the drawn facing.
BDB angles are degrees clockwise from +y; only differences and the attack axis are used, so the convention cancels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.pose import action_class as ac  # noqa: E402

YD = 0.9144
BDB_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB", "CB": "DB", "FS": "DB", "SS": "DB", "S": "DB",
             "DB": "DB", "ILB": "LB", "MLB": "LB", "OLB": "LB", "LB": "LB", "DE": "DL", "DT": "DL", "NT": "DL"}
COVERED = {"WR", "TE", "RB", "QB", "DB", "LB"}


def hand_rule(row) -> int:
    """Offence: downfield forward, back toward its own end zone backward; defence: toward its own end zone in the first
    two seconds backward (the drop), later forward (turned and running), toward the line forward; sideways otherwise."""
    down = np.cos(ac.wrap(row["heading"] - (0.0 if row["attack_sign"] > 0 else np.pi)))
    if row["offence"]:
        return ac.FWD if down > 0.5 else (ac.BACK if down < -0.5 else ac.LAT)
    if down > 0.5:
        return ac.BACK if row["t_snap"] < 2.0 else ac.FWD
    return ac.FWD if down < -0.5 else ac.LAT


def bdb_samples(paths):
    rows, labels, groups = [], [], []
    for path in paths:
        t = pd.read_csv(path)
        for (g, pl), play in t.groupby(["game_id", "play_id"]):
            sign = 1.0 if play["play_direction"].iloc[0] == "right" else -1.0
            los = float(play["absolute_yardline_number"].iloc[0])
            for fid, fr in play.groupby("frame_id"):
                P = fr[["x", "y"]].to_numpy(float) * YD
                side = (fr["player_side"] == "Offense").to_numpy()
                passer = fr["player_role"].to_numpy() == "Passer"
                qb = P[passer][0] if passer.any() else None
                for i, r in enumerate(fr.itertuples()):
                    sp = float(r.s) * YD
                    if sp < ac.V_MIN:
                        continue
                    h = np.radians(90.0 - float(r.dir))
                    fac = np.radians(90.0 - float(r.o))
                    opp = np.linalg.norm(P[side != side[i]] - P[i], axis=1)
                    row = dict(speed=sp, heading=h, attack_sign=sign, offence=bool(side[i]),
                               role=BDB_GROUP.get(r.player_position), t_snap=(int(fid) - 1) / 10.0, phase="pre",
                               depth=sign * (float(r.x) - los) * YD * (-1.0 if side[i] else 1.0),
                               ball_bearing=None if (qb is None or passer[i]) else float(np.arctan2(qb[1] - P[i][1], qb[0] - P[i][0])),
                               nearest_opp=float(opp.min()) if len(opp) else 10.0, nose=0.0, cam_bearing=None, lr_sign=0.0)
                    rows.append(row)
                    labels.append(ac.relative_class(fac, sp * np.array([np.cos(h), np.sin(h)])))
                    groups.append(f"{g}_{pl}")
    return rows, np.asarray(labels), np.asarray(groups)


def play_samples(D, P, roles):
    ev = json.loads((P / "ball.json").read_text())
    snap, release = int(ev["snap"]), int(ev["release"])
    teams = {int(k): v for k, v in D["teams"].items()}
    offence = teams.get(int(ev["passer"]))
    passer = int(ev["passer"])
    los_x = float(D["los"]["x"])
    off = int(json.loads((P / "clip_offset.json").read_text()).get("offset", -15))
    tr = pd.read_parquet(P / "tracks.parquet", columns=["frame", "cam", "global_player_id"])
    ez = set(zip((tr[tr.cam == "endzone"].frame.astype(int) - off).tolist(), tr[tr.cam == "endzone"].global_player_id.astype(int).tolist()))
    bodies = {int(f): {int(q[0]): np.asarray(q[2], float) for q in rows} for f, rows in D["bodies"].items()}
    f0 = min((f for f in bodies if f >= snap), default=min(bodies))
    xs = [J[0][0] for p, J in bodies[f0].items() if teams.get(p) == offence]
    attack = -1.0 if np.median(xs) > los_x else 1.0
    xy = {}
    for f, d in bodies.items():
        for p, J in d.items():
            xy.setdefault(p, {})[f] = J[0][:2]
    out = []
    for p, track in xy.items():
        role = roles.get(str(p))
        if role not in COVERED or p not in teams:
            continue
        for f in sorted(track):
            if not snap <= f < release:
                continue
            m = ac.track_motion(track, f, half=4)
            if m is None or m[0] < ac.V_MIN:
                continue
            J = bodies[f][p]
            is_off = teams[p] == offence
            opp = [np.linalg.norm(q[0][:2] - J[0][:2]) for p2, q in bodies[f].items() if teams.get(p2) and teams.get(p2) != teams[p]]
            qb = bodies[f].get(passer)
            row = dict(speed=m[0], heading=m[1], attack_sign=attack, offence=is_off, role=role, t_snap=(f - snap) / ac.FPS,
                       phase="pre", depth=attack * (J[0][0] - los_x) * (-1.0 if is_off else 1.0),
                       ball_bearing=None if (qb is None or p == passer) else float(np.arctan2(qb[0][1] - J[0][1], qb[0][0] - J[0][0])),
                       nearest_opp=min(opp) if opp else 10.0, nose=0.0, cam_bearing=None, lr_sign=0.0)
            hip = np.cross([0.0, 0.0, 1.0], J[2] - J[1])[:2]
            drawn = ac.relative_class(float(np.arctan2(hip[1], hip[0])), m[0] * np.array([np.cos(m[1]), np.sin(m[1])]))
            out.append((p, f, row, drawn, (f, p) in ez))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdb", nargs="+", required=True, type=Path)
    ap.add_argument("--joints", required=True, type=Path)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--roles", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-p", type=float, default=0.8)
    ap.add_argument("--model-out", type=Path, default=None)
    a = ap.parse_args()
    rows, y, groups = bdb_samples(a.bdb)
    X = np.stack([ac.feature_vector(r) for r in rows])
    print(f"BDB: {len(y)} moving samples from {len(np.unique(groups))} plays; "
          + ", ".join(f"{ac.CLASS_NAMES[c]} {np.mean(y == c):.1%}" for c in ac.CLASSES))
    maj = np.bincount(y, minlength=3).argmax()
    print(f"baseline, majority ({ac.CLASS_NAMES[maj]}): {np.mean(y == maj):.1%};  hand rule: "
          f"{np.mean(np.array([hand_rule(r) for r in rows]) == y):.1%}")
    acc, per, _ = ac.grouped_cv_accuracy(X, y, groups, folds=a.folds, per_class=True)
    print(f"MLP held out by play: {acc:.1%}  " + ", ".join(f"{k} {v:.0%}" for k, v in per.items()))
    model = ac.train(X, y)
    if a.model_out:
        import torch

        a.model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model[0].state_dict(), "mu": model[1], "sd": model[2], "features": ac.FEATURE_NAMES}, a.model_out)
        print(f"saved {a.model_out}")
    D = json.loads(a.joints.read_text())
    roles = json.loads(a.roles.read_text())
    S = play_samples(D, a.play_dir, roles)
    Xp = np.stack([ac.feature_vector(s[2]) for s in S])
    pr = ac.predict_proba(model, Xp)
    pc, pm = pr.argmax(1), pr.max(1)
    drawn = np.array([s[3] for s in S]); two = np.array([s[4] for s in S])
    L = two & (drawn != ac.SLOW)
    print(f"play: {len(S)} moving pre-release samples of covered groups; two-view {int(L.sum())}: agreement with the drawn "
          f"(two-view) class {np.mean(pc[L] == drawn[L]):.1%} (majority {np.mean(drawn[L] == np.bincount(drawn[L], minlength=3).argmax()):.1%})")
    one = ~two & (drawn != ac.SLOW)
    conf = one & (pm >= a.min_p)
    dis = conf & (pc != drawn)
    print(f"one-view: {int(one.sum())}; confident (p >= {a.min_p}) {int(conf.sum())}; disagreeing with the drawn facing {int(dis.sum())}")
    by = {}
    for i in np.flatnonzero(dis):
        p, f = S[i][0], S[i][1]
        by.setdefault(p, []).append((f, ac.CLASS_NAMES[int(pc[i])], ac.CLASS_NAMES[int(drawn[i])], round(float(pm[i]), 2)))
    for p, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"   id {p} ({roles.get(str(p))}): {len(v)}: " + ", ".join(f"{f}:{a_}/{b_}" for f, a_, b_, _q in v[:10]))


if __name__ == "__main__":
    main()
