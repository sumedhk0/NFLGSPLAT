"""Where a man looks: the first-person gaze for the viewer.

WHY. The viewer's first-person camera followed a play rule -- moving faster than 1.5 m/s: along the run; a lineman or
the ball carrier: the attack direction; anyone else: at the ball -- that ignored the drawn body. On play 1's live
window (2,346 body-frames) that rule pointed more than 90 degrees away from the drawn torso on 39 % of them, and 30 %
were men moving BACKWARD relative to their torso (safeties backpedalling, tackles in their pass sets, the
quarterback's drop), where it looked behind them. The film ruler (both broadcast cameras, arrows from the helmet;
2026-09-24) found the drawn torso right on every one of the backward cases it checked, and found the one real
exception the torso misses: the quarterback in the pocket stands sideways to the target (a right-handed passer's
front shoulder points at it) with his head turned downfield, a quarter turn from his chest.

WHAT. The look is the drawn torso's facing plus a bounded head turn toward what the man is watching:

* base: the torso heading (up x the shoulder line), circular-smoothed along each man's frames;
* target, by role and phase (events from ball.json): the passer looks downfield, then at his receiver for the last
  READ_FRAMES before the release; while the ball is in flight everyone off the line watches it; the defence watches
  the ball (the quarterback before the throw, the carrier after the catch); offensive linemen and the offence's
  other players look where their chest points (no turn), and so does the carrier after the catch;
* the turn toward the target is clamped to +-MAX_TURN_DEG from the torso (the neck's range), and a target more
  than TURN_WITHIN_DEG off the chest is not watched at all (it is behind his shoulder);
* pitch: toward the ball's height when the target is the ball, else PITCH_DEG (a little below the horizon);
* the eye sits EYE_UP above and EYE_FWD ahead of the SMPL-X head joint along the look.

Numpy only; the export (05k --export-joints) writes the table as ``look`` and the viewer reads it.
"""
from __future__ import annotations

import numpy as np

MAX_TURN_DEG: float = 75.0      # the head's yaw range about the torso (neck rotation, without the eyes)
TURN_WITHIN_DEG: float = 125.0  # a target further than this off the chest is behind his shoulder: not watched. The
                                # quarterback's play-action fake (back to the defence, 420-440 on play 1) otherwise
                                # cranked his head 75 deg over his shoulder toward "downfield"; the endzone film has his
                                # face to his own backfield. In the pocket (chest 68-118 deg off downfield) he turns.
PITCH_DEG: float = -8.0         # the resting gaze: a little below the horizon
PITCH_MIN_DEG: float = -35.0
PITCH_MAX_DEG: float = 45.0
READ_FRAMES: int = 30           # play frames (60 fps) before the release the passer looks at his receiver
DOWNFIELD_M: float = 15.0       # the passer's scanning point: this far ahead along the attack direction
LINE_M: float = 3.0             # a man within this of the line of scrimmage at the snap is a lineman
SMOOTH_SIGMA: float = 1.5       # samples: circular Gaussian on the torso heading and on the look's yaw
# The eye: SMPL-X's own eye joints (23, 24; their midpoint) sit 4.1-4.5 cm above the head joint (15) and 6.5-6.8 cm
# ahead of it in the rest pose, over statures 1.62-1.87 m (neutral model, betas[0] -1..+1.5). The first guess (8 cm up,
# 9 cm ahead) put the first-person camera 4 cm above the man's eyes.
EYE_UP: float = 0.045           # metres above the SMPL-X head joint
EYE_FWD: float = 0.065          # metres ahead of it along the look
SHOULDERS = (16, 17)            # SMPL-X left / right shoulder
HEAD = 15


def torso_heading(J) -> float | None:
    """The drawn body's facing on the ground, radians (atan2 in the field frame, z up): up x (right - left
    shoulder). None when the shoulder line is vertical."""
    J = np.asarray(J, float)
    right = J[SHOULDERS[1]] - J[SHOULDERS[0]]
    f = np.cross([0.0, 0.0, 1.0], right)[:2]
    if np.linalg.norm(f) < 1e-6:
        return None
    return float(np.arctan2(f[1], f[0]))


def wrap(a):
    """Angle(s) to (-pi, pi]."""
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def smooth_angles(a, sigma: float):
    """Circular Gaussian smoothing of a heading sequence (radians): unwrap, filter, wrap."""
    a = np.asarray(a, float)
    if sigma <= 0 or len(a) < 3:
        return wrap(a)
    from scipy.ndimage import gaussian_filter1d

    return wrap(gaussian_filter1d(np.unwrap(a), sigma, mode="nearest"))


def turn_toward(base: float, target: float, max_turn_deg: float = None, within_deg: float = None) -> float:
    """``base`` turned toward ``target`` (radians) by at most ``max_turn_deg`` (default MAX_TURN_DEG); no turn at all
    when the target is more than ``within_deg`` (default TURN_WITHIN_DEG) off ``base``."""
    lim = np.radians(MAX_TURN_DEG if max_turn_deg is None else max_turn_deg)
    within = np.radians(TURN_WITHIN_DEG if within_deg is None else within_deg)
    d = float(wrap(target - base))
    if abs(d) > within:
        return float(wrap(base))
    return float(wrap(base + np.clip(d, -lim, lim)))


def attack_sign(pelvis_x_offence, los_x: float) -> float:
    """+1 when the offence attacks toward +x, -1 toward -x: it lines up on the other side of the line (the median
    man, so a receiver already downfield at the snap frame cannot flip it)."""
    return -1.0 if float(np.median(pelvis_x_offence)) > float(los_x) else 1.0


def look_table(bodies: dict, ball: dict, events: dict, *, los_x: float, teams: dict) -> dict:
    """``{pid: {frame: (x, y, z)}}`` unit look vectors and ``{pid: {frame: (x, y, z)}}`` eye points.

    ``bodies``: {frame: {pid: J [22, 3]}} (the drawn joints, field frame, z up); ``ball``: {frame: (x, y, z)};
    ``events``: snap, release, catch (frames) and passer, receiver (ids); ``teams``: {pid: team}.
    Returns ``(look, eye)``."""
    frames = sorted(int(f) for f in bodies)
    snap, release, catch = int(events["snap"]), int(events["release"]), int(events["catch"])
    passer, receiver = int(events["passer"]), int(events.get("receiver", -1))
    offence = teams.get(passer)
    f0 = next((f for f in frames if f >= snap), frames[0])
    at_snap = bodies.get(f0, {})
    off_x = [np.asarray(J, float)[0][0] for p, J in at_snap.items() if teams.get(p) == offence]
    sign = attack_sign(off_x, los_x) if off_x else -1.0
    lineman = {p for p, J in at_snap.items() if abs(np.asarray(J, float)[0][0] - los_x) < LINE_M and p != passer}

    by_id: dict = {}
    for f in frames:
        for p, J in bodies[f].items():
            by_id.setdefault(int(p), []).append((f, np.asarray(J, float)))
    look, eye = {}, {}
    for pid, rows in by_id.items():
        team = teams.get(pid)
        defence = team is not None and team != offence
        fs = [f for f, _ in rows]
        base_raw = [torso_heading(J) for _, J in rows]
        known = [b for b in base_raw if b is not None]
        if not known:
            continue
        fill, base_seq = known[0], []
        for b in base_raw:                                   # a vertical shoulder line keeps the last heading
            fill = b if b is not None else fill
            base_seq.append(fill)
        base = smooth_angles(base_seq, SMOOTH_SIGMA)
        yaws, pitches, heads = [], [], []
        for (f, J), b in zip(rows, base):
            head = J[HEAD]
            target, to_ball = None, False
            bp = ball.get(f)
            in_flight = release <= f < catch
            if pid == passer and f < release:
                rx = bodies.get(f, {}).get(receiver)
                if receiver >= 0 and rx is not None and f >= release - READ_FRAMES:
                    target = np.asarray(rx, float)[HEAD]
                else:
                    target = head + np.array([sign * DOWNFIELD_M, 0.0, 0.0])
            elif in_flight and bp is not None and not (pid in lineman and not defence):
                target, to_ball = np.asarray(bp, float), True
            elif defence and bp is not None:
                target, to_ball = np.asarray(bp, float), True
            yaw = float(b)
            pitch = np.radians(PITCH_DEG)
            if target is not None:
                d = target - head
                if np.hypot(d[0], d[1]) > 0.5:
                    yaw = turn_toward(float(b), float(np.arctan2(d[1], d[0])))
                    if to_ball:
                        pitch = float(np.clip(np.arctan2(d[2], np.hypot(d[0], d[1])),
                                              np.radians(PITCH_MIN_DEG), np.radians(PITCH_MAX_DEG)))
            yaws.append(yaw)
            pitches.append(pitch)
            heads.append(head)
        yaws = smooth_angles(yaws, SMOOTH_SIGMA)
        if len(pitches) >= 3 and SMOOTH_SIGMA > 0:
            from scipy.ndimage import gaussian_filter1d

            pitches = gaussian_filter1d(np.asarray(pitches, float), SMOOTH_SIGMA, mode="nearest")
        look[pid], eye[pid] = {}, {}
        for f, y, pt, h in zip(fs, yaws, pitches, heads):
            v = np.array([np.cos(pt) * np.cos(y), np.cos(pt) * np.sin(y), np.sin(pt)])
            look[pid][f] = tuple(float(x) for x in v)
            e = h + np.array([0.0, 0.0, EYE_UP]) + EYE_FWD * np.array([np.cos(y), np.sin(y), 0.0])
            eye[pid][f] = tuple(float(x) for x in e)
    return look, eye
