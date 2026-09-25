#!/usr/bin/env python
"""Train and score the per-play relative-motion classifier (pose.action_class) and predict the one-view frames.

    C:/venvs/smplx312/Scripts/python scripts/09k_action_classes.py --play-dir P --joints J.json [--out P/actions.json]

Samples: every drawn body-frame in the live window (snap to the down) moving faster than V_MIN, from the joints
export's pelvis track. Labels: where the endzone has a row for the id on that frame (two views), the class of the
drawn hip facing against the motion. Features: pose.action_class.FEATURE_NAMES (never the facing), with the face cues
from the PRETRAINED keypoint table (the fine-tuned one learnt the fit's own faces). Reports the grouped (by player)
cross-validated accuracy per class against two baselines -- the majority class, and the same model without the
camera cues -- then trains on all two-view samples and writes the class probabilities of every sample, with the
one-view frames where a confident prediction disagrees with the drawn facing listed for the film.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.pose import action_class as ac  # noqa: E402


def hip_facing(J) -> float:
    J = np.asarray(J, float)
    f = np.cross([0.0, 0.0, 1.0], J[2] - J[1])[:2]                 # SMPL-X 1 = left hip, 2 = right hip
    return float(np.arctan2(f[1], f[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--joints", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-p", type=float, default=0.85, help="confidence for the disagreement list")
    a = ap.parse_args()
    P = a.play_dir
    D = json.loads(a.joints.read_text())
    ev = json.loads((P / "ball.json").read_text())
    ev = ev.get("events", ev)
    snap, release, catch = int(ev["snap"]), int(ev["release"]), int(ev["catch"])
    down = int(ev.get("down", 10 ** 9))
    passer = int(ev["passer"])
    teams = {int(k): v for k, v in D["teams"].items()}
    offence = teams.get(passer)
    los_x = float(D["los"]["x"])
    roles = {int(k): v for k, v in (pickle.load(open(P / "identity_resolved.pkl", "rb")).get("roles", {}) or {}).items()}
    off = int(json.loads((P / "clip_offset.json").read_text()).get("offset", -15))
    tr = pd.read_parquet(P / "tracks.parquet", columns=["frame", "cam", "global_player_id"])
    ez = set(zip((tr[tr.cam == "endzone"].frame.astype(int) - off).tolist(),
                 tr[tr.cam == "endzone"].global_player_id.astype(int).tolist()))
    kp = pd.read_parquet(P / "keypoints_2d.parquet")
    kp = kp[(kp.cam == "sideline") & (kp.joint.isin([0, 5, 6]))]
    kpd = {}
    for (f, p), g in kp.groupby(["frame", "global_player_id"]):
        g = g.set_index("joint")
        kpd[(int(f), int(p))] = g
    bodies = {int(f): {int(q[0]): np.asarray(q[2], float) for q in rows} for f, rows in D["bodies"].items()}
    ball = {int(k): np.asarray(v, float) for k, v in D["ball"].items()}
    cams = D["cameras"]["sideline"]
    # the offence's attack direction from the median offensive man at the snap (as render.gaze)
    f0 = min((f for f in bodies if f >= snap), default=min(bodies))
    xs = [J[0][0] for p, J in bodies[f0].items() if teams.get(p) == offence]
    attack = -1.0 if np.median(xs) > los_x else 1.0
    xy = {}
    for f, d in bodies.items():
        for p, J in d.items():
            xy.setdefault(p, {})[f] = J[0][:2]
    rows, keys, labels, drawn_cls, twoview = [], [], [], [], []
    for p, track in xy.items():
        if p not in teams:
            continue
        for f in sorted(track):
            if not snap <= f <= down:
                continue
            m = ac.track_motion(track, f, half=4)
            if m is None or m[0] < ac.V_MIN:
                continue
            speed, heading = m
            J = bodies[f][p]
            v = speed * np.array([np.cos(heading), np.sin(heading)])
            c = cams.get(str(f))
            cam_bearing = None
            if c is not None:
                R, t = np.asarray(c["R"], float).reshape(3, 3), np.asarray(c["t"], float)
                C = -R.T @ t
                cam_bearing = float(np.arctan2(C[1] - J[0][1], C[0] - J[0][0]))
            k = kpd.get((f, p))
            nose, lr = 0.0, 0.0
            if k is not None:
                if 0 in k.index:
                    nose = float(k.loc[0, "conf"])
                if 5 in k.index and 6 in k.index and min(k.loc[5, "conf"], k.loc[6, "conf"]) > 0.5:
                    lr = 1.0 if k.loc[6, "x"] < k.loc[5, "x"] else -1.0   # right shoulder on the image left = faces the camera
            opp = [np.linalg.norm(q[0][:2] - J[0][:2]) for p2, q in bodies[f].items()
                   if teams.get(p2) and teams.get(p2) != teams[p]]
            bp = ball.get(f)
            is_off = teams[p] == offence
            row = dict(speed=speed, heading=heading, attack_sign=attack, offence=is_off, role=roles.get(p),
                       t_snap=(f - snap) / ac.FPS, phase="pre" if f < release else ("flight" if f < catch else "after"),
                       depth=(attack * (J[0][0] - los_x)) * (-1.0 if is_off else 1.0),
                       ball_bearing=None if bp is None else float(np.arctan2(bp[1] - J[0][1], bp[0] - J[0][0])),
                       nearest_opp=min(opp) if opp else 10.0, nose=nose, cam_bearing=cam_bearing, lr_sign=lr)
            rows.append(ac.feature_vector(row))
            keys.append((p, f, heading))
            cls = ac.relative_class(hip_facing(J), v)
            drawn_cls.append(cls)
            two = (f, p) in ez
            twoview.append(two)
            labels.append(cls if two else -1)
    X = np.asarray(rows); y = np.asarray(labels); two = np.asarray(twoview); dc = np.asarray(drawn_cls)
    ids = np.asarray([k[0] for k in keys])
    L = two & (y >= 0) & (y != ac.SLOW)
    print(f"samples: {len(X)} moving body-frames {snap}-{down}; two-view labelled {int(L.sum())} "
          f"({', '.join(f'{ac.CLASS_NAMES[c]} {int(np.sum(y[L] == c))}' for c in ac.CLASSES)}), one-view {int((~two).sum())}")
    maj = np.bincount(y[L], minlength=3).argmax()
    print(f"baseline, the majority class ({ac.CLASS_NAMES[maj]}): {np.mean(y[L] == maj):.1%}")
    cam_cols = [ac.FEATURE_NAMES.index(n) for n in ("nose", "cam_cos", "cam_sin", "nose_cam_cos", "lr_cam_cos")]
    keep = [i for i in range(X.shape[1]) if i not in cam_cols]
    acc0, per0, _ = ac.grouped_cv_accuracy(X[L][:, keep], y[L], ids[L], folds=a.folds, per_class=True)
    print(f"held out by player, no camera cues: {acc0:.1%}  " + ", ".join(f"{k} {v:.0%}" for k, v in per0.items()))
    acc, per, pred_cv = ac.grouped_cv_accuracy(X[L], y[L], ids[L], folds=a.folds, per_class=True)
    print(f"held out by player, all features:   {acc:.1%}  " + ", ".join(f"{k} {v:.0%}" for k, v in per.items()))
    model = ac.train(X[L], y[L])
    proba = ac.predict_proba(model, X)
    out = {}
    for (p, f, h), pr in zip(keys, proba):
        out.setdefault(str(p), {})[str(f)] = [round(float(v), 4) for v in pr] + [round(float(h), 4)]
    dest = a.out or (P / "actions.json")
    dest.write_text(json.dumps({"classes": [ac.CLASS_NAMES[c] for c in ac.CLASSES], "fields": "p per class, then the "
                                "motion heading (rad)", "cv_accuracy": acc, "by_id": out}))
    print(f"wrote {dest}")
    one = ~two
    pc = np.argmax(proba, axis=1); pm = proba.max(axis=1)
    conf = one & (pm >= a.min_p) & (dc != ac.SLOW)
    dis = conf & (pc != dc)
    print(f"one-view samples with a confident prediction (p >= {a.min_p}): {int(conf.sum())} of {int(one.sum())}; "
          f"disagreeing with the drawn facing: {int(dis.sum())}")
    by = {}
    for i in np.flatnonzero(dis):
        p, f, _h = keys[i]
        by.setdefault(p, []).append((f, ac.CLASS_NAMES[int(pc[i])], ac.CLASS_NAMES[int(dc[i])]))
    for p, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        fs = [x[0] for x in v]
        print(f"   id {p} ({teams.get(p)}, {roles.get(p)}): {len(v)} -- frames {fs[0]}..{fs[-1]}, predicted "
              f"{max(set(x[1] for x in v), key=[x[1] for x in v].count)} vs drawn {max(set(x[2] for x in v), key=[x[2] for x in v].count)}")


if __name__ == "__main__":
    main()
