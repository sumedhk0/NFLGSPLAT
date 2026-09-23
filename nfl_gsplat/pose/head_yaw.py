"""Head yaw from the COCO face keypoints: where a man LOOKS, as distinct from where his shoulders face.

WHY. The first-person view followed the fitted shoulder line and the user saw the quarterback "facing the sideline"
in the pocket. The film (2026-09-23) shows his torso does face the far sideline at 470-500 (the sideline camera sees
the number on his back, the endzone camera his profile) while his head turns downfield: the fit is right, and a
first-person camera needs the head. SMPL-X's head joint carries no gaze of its own from the fit; the detector's
nose, eyes and ears do.

WHAT. In one camera's image the nose's position between the two ears tells the head's yaw about the vertical:
centred = facing the camera, on an ear = a quarter turn, one ear hidden = past a quarter turn to that side, both
ears hidden and the nose hidden = facing away. Converted to a field-frame heading with the camera's own azimuth,
and a confidence from the keypoints' confidences. Numpy only.

VERDICT (2026-09-23, play 1, fast men look where they run, 78 readings): NOT wired into the viewer. Heads are 8 px
(sideline) / 21 px (endzone) between nose and ear in All-22 footage and the keypoints sit on helmets: the sideline's
one-ear readings are 118 deg off (median), the endzone's "both ears, no nose = facing away" readings 94 deg off (the
ear holes of a helmet show from behind AND from the side); only the sideline's facing-away (8 deg, n 14) and the
endzone's one-ear (26 deg, n 17) cases read. On the quarterback the endzone one-ear case was RIGHT where the
play-derived gaze is wrong (430-450: the play-action fake, he faces his own end zone) and the sideline facing-away
case WRONG where the play-derived gaze is right (480-520: reads the far sideline, the film says downfield). 05k
exports the readings ("gaze") as raw material; the viewer keeps the play-derived gaze.
"""
from __future__ import annotations

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
MIN_CONF: float = 0.5


def head_yaw_camera(xy: np.ndarray, conf: np.ndarray, *, min_conf: float = MIN_CONF) -> tuple[float, float] | None:
    """``(yaw_rad, confidence)`` of the head about the vertical, in the CAMERA's frame: 0 = facing the camera,
    positive = turned toward the image's right (the man's left), pi = facing away. None when the face keypoints
    do not say. ``xy [17, 2]``, ``conf [17]``."""
    c = np.asarray(conf, float)
    p = np.asarray(xy, float)
    nose, le, re = c[NOSE] >= min_conf, c[L_EAR] >= min_conf, c[R_EAR] >= min_conf
    if nose and le and re:
        mid = 0.5 * (p[L_EAR] + p[R_EAR])
        half = 0.5 * abs(p[L_EAR, 0] - p[R_EAR, 0])
        if half < 1e-6:
            return None
        s = float(np.clip((p[NOSE, 0] - mid[0]) / half, -1.0, 1.0))
        # the man's left ear sits on the image's right when he faces the camera
        sign = 1.0 if p[L_EAR, 0] > p[R_EAR, 0] else -1.0
        return float(np.arcsin(s) * sign), float(min(c[NOSE], c[L_EAR], c[R_EAR]))
    if nose and (le != re):
        # one ear hidden: he has turned past a quarter turn toward the visible ear's side
        ear = L_EAR if le else R_EAR
        # the nose leads the way: an ear to the image-right of the nose means he faces image-LEFT (a negative turn)
        side = -1.0 if p[ear, 0] > p[NOSE, 0] else 1.0
        return float(side * np.pi * 0.35), float(min(c[NOSE], c[ear]))
    if not nose and le and re:
        return float(np.pi), float(min(c[L_EAR], c[R_EAR]))              # both ears, no nose: facing away
    return None


def camera_azimuth(R: np.ndarray) -> float:
    """The field-frame azimuth (radians, atan2(y, x)) of the camera's optical axis, from a world-to-camera ``R``."""
    fwd = np.asarray(R, float).T @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(fwd[1], fwd[0]))


def head_heading_world(yaw_cam: float, R: np.ndarray) -> float:
    """The head's field-frame heading (radians) from its camera-frame yaw: facing the camera means heading back
    along the optical axis; a positive camera yaw turns him toward the image's right = HIS left = counter-clockwise
    seen from above (z up), so the heading increases by the yaw."""
    az = camera_azimuth(R)
    facing_camera = az + np.pi
    return float((facing_camera + yaw_cam + np.pi) % (2 * np.pi) - np.pi)


def head_headings(kdf, tracks, *, cam: str = "sideline", frame_shift: int = 0, min_conf: float = MIN_CONF) -> dict:
    """``{pid: {frame: (heading_rad, confidence)}}`` from one camera's keypoints table (frame, cam, global_player_id,
    joint, x, y, conf) and its camera track; ``frame_shift`` moves that camera's clip frames to the play's frames
    (endzone rows: frame - offset)."""
    sub = kdf[kdf["cam"] == cam]
    out: dict = {}
    tr = tracks[cam]
    for (f, pid), g in sub.groupby(["frame", "global_player_id"]):
        if len(g) != 17:
            continue
        g = g.sort_values("joint")
        r = head_yaw_camera(g[["x", "y"]].to_numpy(float), g["conf"].to_numpy(float), min_conf=min_conf)
        if r is None:
            continue
        cf = int(f)
        if cf < 0 or cf >= tr.num_frames or tr.conf[cf] <= 0:
            continue
        _intr, pose = tr.at(cf)
        out.setdefault(int(pid), {})[cf - frame_shift] = (head_heading_world(r[0], np.asarray(pose.R, float)), r[1])
    return out
