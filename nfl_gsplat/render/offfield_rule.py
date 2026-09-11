"""Bodies that are not players: sideline dwellers by position, officials by stripes.

WHY. Play 1's render drew coaches, chain crew and photographers standing at
the boundary as white-kit players, and the officials in the white kit too.
Measured 2026-09-07 on play 1's sideline ids (`diag/p1_offfield_stats.csv`,
crops in `diag/p1_officials_sheet.jpg`): every roster-named player has a
median |y| under 6 m and no frame beyond 23.5 m; fourteen unnamed ids stand
at |y| 24.4-25.9 m for their whole track -- staff in dark jackets and the
sideline official. Colour cannot tell staff from players (a dark jacket, a
player's purple pants and a striped shirt all read "dark"), position can.
Officials inside the field need the stripes: the horizontal-to-vertical
gradient ratio of the torso window is 3.96 on the official against under
1.8 on every player and staff member, so a ratio of 2.5 with a dark share
of 0.25 is the signature.

WHAT. ``sideline_dwellers``: ids whose ground point sits at or beyond
``SIDELINE_M`` in at least ``DWELL_FRAC`` of their frames. ``striped_ids``:
ids whose sampled sideline torso crops show the stripe signature. Both are
sets of ids to leave out of the timeline, the shape of the edge rule.
"""
from __future__ import annotations

import numpy as np

SIDELINE_M: float = 23.5       # the sideline is 24.38 m from midfield; the box bottom is 1 m noisy
DWELL_FRAC: float = 0.8
STRIPE_RATIO: float = 2.5      # horizontal / vertical mean gradient in the torso window
STRIPE_DARK: float = 0.25      # share of torso pixels under gray 70
STRIPE_FRAMES: int = 6


def sideline_dwellers(ground_by_frame, *, sideline_m: float = SIDELINE_M, frac: float = DWELL_FRAC,
                      min_frames: int = 5) -> set:
    """``ground_by_frame``: frame -> {pid: (x, y)}."""
    beyond: dict = {}
    total: dict = {}
    for d in ground_by_frame.values():
        for pid, xy in d.items():
            y = float(np.asarray(xy, float)[1])
            if not np.isfinite(y):
                continue
            total[int(pid)] = total.get(int(pid), 0) + 1
            if abs(y) >= sideline_m:
                beyond[int(pid)] = beyond.get(int(pid), 0) + 1
    return {pid for pid, n in total.items() if n >= min_frames and beyond.get(pid, 0) / n >= frac}


def stripe_stats(crop_bgr) -> tuple[float, float]:
    """``(gradient ratio, dark share)`` of a torso crop (BGR)."""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = float(np.abs(np.diff(gray, axis=1)).mean())
    gy = float(np.abs(np.diff(gray, axis=0)).mean())
    return gx / (gy + 1e-6), float((gray < 70).mean())


def striped_ids(df, video, *, cam: str = "sideline", ratio: float = STRIPE_RATIO, dark: float = STRIPE_DARK,
                n_frames: int = STRIPE_FRAMES) -> set:
    """Ids whose median torso crop from ``cam`` shows the stripe signature.
    ``df`` is tracks.parquet; ``video`` that camera's clip."""
    import cv2

    g_all = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    cap = cv2.VideoCapture(str(video))
    out = set()
    for pid, g in g_all.groupby("track_id"):
        g = g.sort_values("frame")
        pick = g.iloc[np.linspace(0, len(g) - 1, min(n_frames, len(g))).astype(int)]
        vals = []
        for r in pick.itertuples():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, im = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = int(r.bbox_x1), int(r.bbox_y1), int(r.bbox_x2), int(r.bbox_y2)
            h = y2 - y1
            crop = im[max(0, y1 + int(0.25 * h)):max(0, y1 + int(0.60 * h)), max(0, x1):max(0, x2)]
            if crop.size < 100:
                continue
            vals.append(stripe_stats(crop))
        if len(vals) >= 3:
            rr, dd = np.median(np.array(vals), axis=0)
            if rr >= ratio and dd >= dark:
                out.add(int(pid))
    cap.release()
    return out
