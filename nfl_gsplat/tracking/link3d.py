"""Link per-frame ground placements into players, on the field plane.

WHY. Measured on the Helmet Assignment set (scripts/07j), the 2D tracker
(YOLO + BoT-SORT, per view) breaks each player into 5-7 pieces per play, with
a tenth-percentile track purity under 0.6, and the gap stitcher merges wrong
pieces (switches went UP). Tracking in the image has to cope with players
crossing in front of one another, boxes merging, and a camera that pans and
zooms; none of that exists on the turf. Once the calibration puts every
detection on the ground in metres -- which it now does from either view, and
from both views reconciled -- linking becomes a well-posed problem in a
space where players have physical speed limits and never overlap.

WHAT THIS DOES. Frame to frame, in metres: predict every live track's next
position from its velocity, assign detections one-to-one (Hungarian) under a
gate that grows with the time gap and the player speed limit, start tracks
for what nothing claimed, retire tracks unseen for longer than a short gap.
Constant velocity is enough at 60 Hz -- a player's speed cannot change much
in a sixtieth of a second -- and a robust velocity estimate keeps one noisy
placement from throwing the prediction.

WHAT POSITION ALONE CANNOT DO. Two players passing within noise of each
other at a closing speed of 14 m/s are a coin flip for any position-only
linker at 60 Hz -- measured on synthetic data, and after a swap the velocity
fit blends two players and the track fragments. Real players do not pass
through each other, but they do brush past, and the cue that settles it is
cheap: the team. Each detection may carry a label (team from jersey colour,
identity/team_color); a track votes its label from its detections, and a
detection is never assigned across a known label mismatch.

WHAT IT REFUSES TO DO. It does not bridge long gaps by guessing: a player
missing for a second comes back as a new track, and the caller can decide
what to do with two pieces. Guessing was what the stitcher did, and it was
measured to be worse than not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# An NFL player sprints at about 10 m/s; the gate allows a little more.
MAX_SPEED_M_S: float = 11.0

# Placement noise: the two views disagree by ~0.9 m at the median on the real
# play, so a gate below that would break tracks on noise alone.
GATE_FLOOR_M: float = 1.0

# Unseen for longer than this and the track is retired rather than extended.
MAX_GAP_S: float = 0.5

# A track shorter than this is clutter (a referee glimpsed, a merged box).
MIN_TRACK_FRAMES: int = 3

# A new track is TENTATIVE until it has this many hits, and while tentative it
# only gets the detections confirmed tracks did not want. Traced on synthetic
# data: one 4-sigma placement outside the gate spawned a second track on the
# same player, and the impostor then won half of that player's detections for
# the rest of the play. Confirmed tracks assign first; impostors starve.
CONFIRM_HITS: int = 3
TENTATIVE_MAX_MISSES: int = 2

# Velocity is a straight-line fit over this much recent history. At 60 Hz a
# single-step velocity is noise: 0.3 m of placement error over 1/60 s reads
# as 18 m/s, and the first version predicted players off the field. Over a
# quarter of a second the same noise is ~2 m/s, and 2 m/s over one frame is
# 3 cm of prediction error -- well inside the gate.
VELOCITY_WINDOW_S: float = 0.25
VELOCITY_MIN_POINTS: int = 3


@dataclass
class Track3D:
    id: int
    frames: list[int] = field(default_factory=list)
    xy: list[np.ndarray] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # The fitted position at the last time -- what the prediction starts
    # from. Starting from the last RAW point stacked two placements' noise
    # into every residual (95th percentile 1.04 m against a 1.08 m gate) and
    # fragmented perfect synthetic tracks; the fit averages the window.
    state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    labels: list[int] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)      # detection index per point
    misses: int = 0

    @property
    def confirmed(self) -> bool:
        return len(self.frames) >= CONFIRM_HITS

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    @property
    def label(self) -> int:
        """Majority label over the track's detections, or -1 if none known."""
        known = [v for v in self.labels if v >= 0]
        if not known:
            return -1
        return int(np.bincount(known).argmax())

    def predict(self, frame: int, fps: float) -> np.ndarray:
        dt = (frame - self.last_frame) / fps
        return self.state + self.velocity * dt

    def extend(self, frame: int, xy: np.ndarray, fps: float, label: int = -1,
               row: int = -1) -> None:
        self.frames.append(int(frame))
        self.xy.append(np.asarray(xy, float))
        self.labels.append(int(label))
        self.rows.append(int(row))
        t = np.asarray(self.frames, float) / fps
        recent = t >= t[-1] - VELOCITY_WINDOW_S
        if recent.sum() >= VELOCITY_MIN_POINTS and np.ptp(t[recent]) > 0:
            tt = t[recent] - t[recent].mean()
            pts = np.asarray(self.xy, float)[recent]
            self.velocity = (tt[:, None] * (pts - pts.mean(0))).sum(0) / (tt ** 2).sum()
            self.state = pts.mean(0) + self.velocity * (t[-1] - t[recent].mean())
        else:
            self.velocity = np.zeros(2)
            self.state = np.asarray(self.xy, float)[recent].mean(0)


def link(placements, *, labels=None, fps: float = 59.94,
         max_speed: float = MAX_SPEED_M_S, gate_floor: float = GATE_FLOOR_M,
         max_gap_s: float = MAX_GAP_S,
         min_frames: int = MIN_TRACK_FRAMES) -> list[Track3D]:
    """``placements`` maps frame -> ``[N, 2]`` ground positions in metres.

    ``labels`` optionally maps frame -> ``[N]`` ints (e.g. team; -1 unknown);
    a detection is never linked to a track whose known label differs.
    Returns tracks with at least ``min_frames`` points, in order of creation.
    """
    from scipy.optimize import linear_sum_assignment

    live: list[Track3D] = []
    done: list[Track3D] = []
    next_id = 0
    for f in sorted(placements):
        dets = np.asarray(placements[f], float).reshape(-1, 2)
        labs = (np.asarray(labels[f], int).reshape(-1)
                if labels is not None and f in labels else np.full(len(dets), -1))
        keep = np.isfinite(dets).all(1)
        rows = np.flatnonzero(keep)                 # index into the caller's array
        dets, labs = dets[keep], labs[keep]
        # Retire what has been gone too long.
        still = []
        for tr in live:
            if (f - tr.last_frame) / fps > max_gap_s:
                done.append(tr)
            else:
                still.append(tr)
        live = still

        matched_det = np.zeros(len(dets), bool)

        def assign(cands, avail):
            """One-to-one within a gate and a label; marks what it takes."""
            if not cands or not avail.any():
                return
            idx = np.flatnonzero(avail)
            preds = np.stack([tr.predict(f, fps) for tr in cands])
            gates = np.array([gate_floor + max_speed * (f - tr.last_frame) / fps
                              for tr in cands])
            cost = np.linalg.norm(preds[:, None] - dets[idx][None], axis=2)
            blocked = cost > gates[:, None]
            tl = np.array([tr.label for tr in cands])
            mismatch = ((tl[:, None] >= 0) & (labs[idx][None, :] >= 0)
                        & (tl[:, None] != labs[idx][None, :]))
            blocked = blocked | mismatch
            cost = np.where(blocked, 1e6, cost)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if not blocked[i, j]:
                    cands[i].extend(f, dets[idx[j]], fps, labs[idx[j]], rows[idx[j]])
                    cands[i].misses = 0
                    matched_det[idx[j]] = True

        if live and len(dets):
            # Confirmed tracks choose first; tentative ones take what is left.
            assign([tr for tr in live if tr.confirmed], ~matched_det)
            assign([tr for tr in live if not tr.confirmed], ~matched_det)
        # A tentative track that keeps missing was born on a stray point.
        kept = []
        for tr in live:
            if tr.last_frame != f:
                tr.misses += 1
            if not tr.confirmed and tr.misses > TENTATIVE_MAX_MISSES:
                continue
            kept.append(tr)
        live = kept
        for j in np.flatnonzero(~matched_det):
            tr = Track3D(next_id)
            next_id += 1
            tr.extend(f, dets[j], fps, labs[j], rows[j])
            live.append(tr)
    done.extend(live)
    done.sort(key=lambda t: t.id)
    return [t for t in done if len(t.frames) >= min_frames]


def assignments(tracks, placements):
    """``{frame: [N] track id per input detection, -1 if none}``.

    Index-aligned with the caller's arrays, so two detections that project to
    the same ground point can never be confused: a value lookup keyed on
    (frame, x, y) would hand both the id written last (review finding).
    """
    out = {int(f): np.full(len(np.asarray(placements[f]).reshape(-1, 2)), -1, int)
           for f in placements}
    for tr in tracks:
        for f, row in zip(tr.frames, tr.rows):
            if row >= 0:
                out[int(f)][row] = tr.id
    return out


def smooth(track: Track3D, *, fps: float = 59.94, window_s: float = 0.25) -> np.ndarray:
    """Gaussian-weighted moving average in time; ``[len(track), 2]``.

    Weighted by time, not by sample index, so a gap in the frames does not
    drag neighbours across it.
    """
    t = np.asarray(track.frames, float) / fps
    xy = np.asarray(track.xy, float)
    sigma = window_s / 2.0
    out = np.empty_like(xy)
    for i in range(len(t)):
        w = np.exp(-0.5 * ((t - t[i]) / sigma) ** 2)
        out[i] = (w[:, None] * xy).sum(0) / w.sum()
    return out


def to_rows(tracks: list[Track3D], *, fps: float = 59.94, smoothed: bool = True):
    """``[(frame, track_id, x, y), ...]`` for every point of every track."""
    rows = []
    for tr in tracks:
        xy = smooth(tr, fps=fps) if smoothed else np.asarray(tr.xy)
        for f, p in zip(tr.frames, xy):
            rows.append((int(f), int(tr.id), float(p[0]), float(p[1])))
    return rows
