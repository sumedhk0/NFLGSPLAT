"""The frame offset between the two clips, from the players who MOVE.

WHY. The sideline and endzone clips are cut independently (NFL Pro); play 1's
were 15 frames apart (0.25 s). The triangulator's search (05n) scored the
median reprojection over every player, and most players stand still around
the snap: a standing player's rays meet at any offset, so that ruler was
flat-to-monotone across its range and settled on the edge (+3 with the old
endzone camera, -10 with the refined one). A runner at 8 m/s moves 0.13 m a
frame: fifteen frames put him 2 m from himself in the other camera, which is
exactly where every two-view fit of play 1's runner went.

WHAT. For every (frame, player) whose sideline box-bottom ground speed is at
least ``speed_min``, the two cameras' rays through the same hip and shoulder
keypoints (endzone at frame + offset) and their miss distance in metres; the
offset with the smallest median miss over those pairs wins, and the curve
must have a minimum inside the range (a monotone curve is not an offset: it
is a camera bias, and the caller refuses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.errors import CalibrationError

JOINTS = (5, 6, 11, 12)          # COCO shoulders and hips: on the torso, seen from both sides
SPEED_MIN_MPS: float = 3.0
MIN_CONF: float = 0.5
SMOOTH_HALF: int = 4
BOX_MARGIN_FRAC: float = 0.078


def _rays(K, R, t, uv):
    C = -R.T @ t
    d = (R.T @ np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T)).T
    return C, d / np.linalg.norm(d, axis=1, keepdims=True)


def _miss(C1, d1, C2, d2):
    w = C1 - C2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = d1 @ w
    e = d2 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    tt = (a * e - b * d) / den
    return np.linalg.norm((C1 + s[:, None] * d1) - (C2 + tt[:, None] * d2), axis=1)


def ground_speeds(tracks_df: pd.DataFrame, track, *, fps: float, cam: str = "sideline",
                  speed_min: float = SPEED_MIN_MPS, margin_frac: float = BOX_MARGIN_FRAC):
    """``{(frame, id)}`` of the camera's tracks moving at least ``speed_min`` (box-bottom ground
    speed over +-SMOOTH_HALF frames)."""
    from nfl_gsplat.pose.place_on_field import ground_point

    side = tracks_df[(tracks_df["cam"] == cam) & (tracks_df["track_id"] >= 0)].sort_values(["track_id", "frame"])
    moving = set()
    for pid, g in side.groupby("track_id"):
        fr = g["frame"].to_numpy(int)
        if len(fr) < 2 * SMOOTH_HALF + 4:
            continue
        xy = np.full((len(fr), 2), np.nan)
        for i, r in enumerate(g.itertuples()):
            f = int(r.frame)
            if f >= len(track.conf) or track.conf[f] <= 0:
                continue
            try:
                xy[i] = ground_point((0.5 * (r.bbox_x1 + r.bbox_x2), r.bbox_y2 - margin_frac * (r.bbox_y2 - r.bbox_y1)),
                                     track.K[f], track.R[f], track.t[f])[:2]
            except Exception:
                continue
        k = SMOOTH_HALF
        for i in range(k, len(fr) - k):
            if np.isfinite(xy[i + k]).all() and np.isfinite(xy[i - k]).all():
                v = np.linalg.norm(xy[i + k] - xy[i - k]) / max(fr[i + k] - fr[i - k], 1) * fps
                if v >= speed_min:
                    moving.add((int(fr[i]), int(pid)))
    return moving


def miss_by_offset(keypoints: pd.DataFrame, track_a, track_b, moving, offsets, *, cam_a="sideline",
                   cam_b="endzone", min_conf: float = MIN_CONF):
    """``{offset: (median miss m, n pairs)}`` over the moving (frame, id) pairs."""
    k = keypoints[(keypoints["conf"] >= min_conf) & keypoints["joint"].isin(JOINTS)]
    A = k[k["cam"] == cam_a].set_index(["frame", "global_player_id", "joint"]).sort_index()
    B = k[k["cam"] == cam_b].set_index(["frame", "global_player_id", "joint"]).sort_index()
    ia = set(zip(A.index.get_level_values(0), A.index.get_level_values(1)))
    ib = set(zip(B.index.get_level_values(0), B.index.get_level_values(1)))
    out = {}
    for off in offsets:
        ms = []
        for f, pid in moving:
            fe = f + off
            if f >= len(track_a.conf) or fe < 0 or fe >= len(track_b.conf) or track_a.conf[f] <= 0 or track_b.conf[fe] <= 0:
                continue
            if (f, pid) not in ia or (fe, pid) not in ib:
                continue
            a = A.loc[(f, pid)]
            b = B.loc[(fe, pid)]
            common = sorted(set(a.index) & set(b.index))
            if len(common) < 2:
                continue
            C1, d1 = _rays(track_a.K[f], track_a.R[f], track_a.t[f], a.loc[common][["x", "y"]].to_numpy(float))
            C2, d2 = _rays(track_b.K[fe], track_b.R[fe], track_b.t[fe], b.loc[common][["x", "y"]].to_numpy(float))
            ms.extend(_miss(C1, d1, C2, d2).tolist())
        out[int(off)] = (float(np.median(ms)) if ms else float("nan"), len(ms))
    return out


def solve_offset(curve: dict, *, min_pairs: int = 50):
    """The offset at the curve's minimum. Raises when the minimum sits on the range's edge
    (no minimum inside: a camera bias, not an offset) or the pairs are too few."""
    offs = sorted(o for o, (m, n) in curve.items() if np.isfinite(m) and n >= min_pairs)
    if len(offs) < 3:
        raise CalibrationError(f"clip offset: only {len(offs)} offsets have {min_pairs}+ moving keypoint pairs")
    best = min(offs, key=lambda o: curve[o][0])
    if best in (offs[0], offs[-1]):
        raise CalibrationError(f"clip offset: the miss keeps falling to the edge of the range ({best:+d}: "
                               f"{curve[best][0]:.3f} m); widen the range or the cameras are biased")
    return best
