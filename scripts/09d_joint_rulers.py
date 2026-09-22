"""Three rulers on the exported world joints (05k --export-joints): hinge flexion out of range, hinge-angle jerk,
and the bend PLANE of a bent limb against the body's own right axis -- the numbers behind "the movement looks odd".

WHY. On 2026-09-22 the user saw unnatural joint angles in play 1's render. Measured here: the flexion itself is in
range (2 odd body-frames in 17,584 -- the hard hinge bounds hold), the knees SNAP (55 body-frames of angle jerk over
25 deg/frame^2, at speed changes), and arms bend SIDEWAYS -- the bend plane in the frontal plane -- on 11 % of the
right arm's bent frames against 4 % of the left's: the camera-far arm, fitted into the wrong plane when occluded.
These three are the scoreboard for the motion work (confidence-gated joint angles, the endzone's keypoints in the fit,
a temporal lifter): each change must move them without moving the reprojection rulers in 07l.

USAGE (nflgsplat env):
  python scripts/09d_joint_rulers.py JOINTS.json [--lo 395 --hi 607] [--jerk-deg 25] [--side 0.4]
"""
from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np

# SMPL-X body joints: (parent, joint, child, "bent" limit in degrees for the plane test)
LIMBS = {"L arm": (16, 18, 20, 150.0), "R arm": (17, 19, 21, 150.0), "L leg": (1, 4, 7, 160.0), "R leg": (2, 5, 8, 160.0)}


def angle(a, b, c) -> float:
    u = np.asarray(a, float) - np.asarray(b, float)
    v = np.asarray(c, float) - np.asarray(b, float)
    d = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def rulers(doc: dict, *, lo: int | None, hi: int | None, jerk_deg: float, side: float) -> dict:
    frames = [f for f in doc["frames"] if (lo is None or f >= lo) and (hi is None or f <= hi)]
    series: dict = collections.defaultdict(dict)
    plane = {k: [] for k in LIMBS}
    n = collections.Counter()
    for f in frames:
        for pid, team, J in doc["bodies"].get(str(f), []):
            J = np.asarray(J, float)
            r = J[17] - J[16]
            r[2] = 0.0
            rn = float(np.linalg.norm(r))
            for name, (p, j, c, lim) in LIMBS.items():
                a = angle(J[p], J[j], J[c])
                series[(pid, name)][f] = a
                n[name] += 1
                if a <= lim and rn > 1e-6:
                    u = J[p] - J[j]; v = J[c] - J[j]
                    nrm = np.cross(u, v); nn = float(np.linalg.norm(nrm))
                    if nn > 1e-6:
                        plane[name].append((abs(float(np.dot(nrm / nn, r / rn))), pid, f, a))
    out = {"body_frames_per_hinge": dict(n), "flexion": {}, "jerk": {}, "sideways": {}}
    for name in LIMBS:
        vals = [a for (pid, nm), s in series.items() if nm == name for a in s.values()]
        out["flexion"][name] = {"over_bent_lt_35": int(sum(a < 35 for a in vals)), "locked_gt_178": int(sum(a > 178 for a in vals))}
    jerks = collections.Counter(); worst = []
    for (pid, name), s in series.items():
        fs = sorted(s)
        for i in range(2, len(fs)):
            if fs[i] - fs[i - 1] == fs[i - 1] - fs[i - 2]:
                j2 = abs((s[fs[i]] - s[fs[i - 1]]) - (s[fs[i - 1]] - s[fs[i - 2]]))
                if j2 > jerk_deg:
                    jerks[name] += 1; worst.append((round(j2), pid, name, fs[i]))
    worst.sort(reverse=True)
    out["jerk"] = {"per_hinge": dict(jerks), "total": int(sum(jerks.values())), "worst": worst[:10]}
    for name, rows in plane.items():
        bent = len(rows); sideways = sum(1 for s, *_ in rows if s < side)
        by_id = collections.Counter(pid for s, pid, f, a in rows if s < side)
        out["sideways"][name] = {"bent": bent, "sideways": sideways, "pct": round(100.0 * sideways / max(1, bent), 1),
                                 "worst_ids": by_id.most_common(5)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joints")
    ap.add_argument("--lo", type=int, default=None)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--jerk-deg", type=float, default=25.0)
    ap.add_argument("--side", type=float, default=0.4, help="bend-plane sagittal-ness below this = sideways")
    a = ap.parse_args()
    doc = json.load(open(a.joints, encoding="utf-8"))
    out = rulers(doc, lo=a.lo, hi=a.hi, jerk_deg=a.jerk_deg, side=a.side)
    print("body-frames per hinge:", out["body_frames_per_hinge"])
    print("flexion out of range:", out["flexion"])
    print(f"hinge-angle jerk > {a.jerk_deg:g} deg/frame^2:", out["jerk"]["per_hinge"], "total", out["jerk"]["total"])
    print("  worst:", out["jerk"]["worst"])
    print(f"sideways bends (plane sagittal-ness < {a.side:g}):")
    for name, r in out["sideways"].items():
        print(f"  {name}: {r['sideways']} of {r['bent']} bent ({r['pct']} %), worst ids {r['worst_ids']}")


if __name__ == "__main__":
    main()
