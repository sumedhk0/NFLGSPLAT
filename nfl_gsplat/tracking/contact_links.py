"""Link a camera track that dies through contact to the track born on the same man moments later.

WHY. On play 1 (2026-09-22) the motion receiver (id 9) met a defender (id 6) at frames 480-493: the detector merged
the two men into one box, id 9's box slid onto the defender, and a fresh track (77) was born on the receiver at 494
-- the user saw "a new player spawn out of him". The fix was typed by hand from the 09c track table (fold 77 into 9,
give 9's stray rows to 6). This module finds such pairs from the tables: a track DEATH inside the play window with no
later rows of that id in the camera, a track BIRTH of another id nearby within a few frames (or overlapping by a few,
when the dying track lingers on the other man), and a score from what the hand-check used -- the identity's team, the
rows' kit votes, the jersey OCR, the kinematic gap between the death's extrapolated path and the birth, whether the
death happened in contact (another id's box overlapping the last box), and the other camera (two ids seen apart at
the same time there are two men). The script (scripts/09e_contact_links.py) prints the table and can apply the links
through 08z / 08za, so the fix stays a per-camera-track fold with its backups.

Every number here is a heuristic weight; the table is meant to be read before --apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_GAP: int = 20            # frames from a death to a birth
MAX_OVERLAP: int = 8         # frames the dying track may linger past the birth (on the other man)
MAX_M: float = 3.0           # metres between the death's extrapolated spot and the birth
VEL_FRAMES: int = 6          # frames of the death track used for its velocity
CONTACT_IOU: float = 0.10    # another box overlapping the last box this much = contact
OTHER_CAM_APART_M: float = 2.0
MIN_SCORE: float = 2.0


@dataclass
class Span:
    cam: str
    pid: int
    track: int
    first: int
    last: int
    n: int


@dataclass
class Link:
    cam: str
    keep: int              # the id that dies
    drop: int              # the id born on the same man
    track: int             # the born camera track (folded into keep)
    death_last: int
    birth_first: int
    dist_m: float
    score: float
    reasons: list = field(default_factory=list)
    reject: str | None = None

    @property
    def overlap(self) -> int:
        return max(0, self.death_last - self.birth_first + 1)


def spans(df: pd.DataFrame) -> list[Span]:
    out = []
    for (cam, pid, tid), g in df.groupby(["cam", "global_player_id", "track_id"]):
        out.append(Span(str(cam), int(pid), int(tid), int(g["frame"].min()), int(g["frame"].max()), len(g)))
    return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _mode(values) -> int | None:
    v = [int(x) for x in values if pd.notna(x) and int(x) > 0]
    if not v:
        return None
    return int(pd.Series(v).mode().iat[0])


def _kit(values) -> str | None:
    v = [str(x) for x in values if isinstance(x, str) and x]
    return pd.Series(v).mode().iat[0] if v else None


def find_links(df: pd.DataFrame, ground: dict, teams: dict, *, cam: str = "sideline", lo: int, hi: int,
               max_gap: int = MAX_GAP, max_overlap: int = MAX_OVERLAP, max_m: float = MAX_M,
               contact_iou: float = CONTACT_IOU, other_cam_apart_m: float = OTHER_CAM_APART_M) -> list[Link]:
    """``df``: tracks rows (frame, cam, track_id, global_player_id, bbox_*, team, jersey_number_ocr; endzone frames
    already on their sideline frames). ``ground``: {(cam, frame, pid): xy}. ``teams``: {pid: 'KC'|'BAL'} from the
    identity. Returns every candidate pair with its score and reasons, best first; ``reject`` set where a hard
    veto fired (different teams, disagreeing jerseys, two men in the other camera)."""
    sub = df[df["cam"] == cam]
    other = df[df["cam"] != cam]
    other_cam = str(other["cam"].iloc[0]) if len(other) else None
    all_spans = spans(sub)
    by_pid: dict = {}
    for s in all_spans:
        by_pid.setdefault(s.pid, []).append(s)
    frames_of = {pid: set(sub.loc[sub["global_player_id"] == pid, "frame"].astype(int)) for pid in by_pid}
    deaths = [s for s in all_spans if lo <= s.last < hi
              and not any(s.last < f <= s.last + max_gap for f in frames_of[s.pid])]
    births = [s for s in all_spans if lo < s.first <= hi
              and not any(s.first - max_gap <= f < s.first for f in frames_of[s.pid])]
    rows_at = {(int(r.frame), int(r.global_player_id)): (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
               for r in sub.itertuples()}
    frame_rows: dict = {}
    for (f, pid), box in rows_at.items():
        frame_rows.setdefault(f, []).append((pid, box))
    track_rows = {(s.pid, s.track): sub[(sub["global_player_id"] == s.pid) & (sub["track_id"] == s.track)]
                  for s in all_spans}
    other_ground = {(f, pid): xy for (c, f, pid), xy in ground.items() if c == other_cam}
    links: list[Link] = []
    for d in deaths:
        # the death's spot and velocity from its last frames
        pts = [(f, ground.get((cam, f, d.pid))) for f in range(d.last - VEL_FRAMES, d.last + 1)]
        pts = [(f, np.asarray(xy, float)) for f, xy in pts if xy is not None]
        if not pts:
            continue
        f_last, xy_last = pts[-1]
        vel = ((pts[-1][1] - pts[0][1]) / max(1, pts[-1][0] - pts[0][0])) if len(pts) > 1 else np.zeros(2)
        for b in births:
            if b.pid == d.pid:
                continue
            gap = b.first - d.last
            if gap > max_gap or gap < -max_overlap:
                continue
            xy_b = ground.get((cam, b.first, b.pid))
            if xy_b is None:
                continue
            if gap >= 0:
                pred = xy_last + vel * gap
            else:
                at = ground.get((cam, b.first, d.pid))
                pred = np.asarray(at, float) if at is not None else xy_last
            dist = float(np.linalg.norm(np.asarray(xy_b, float) - pred))
            if dist > max_m:
                continue
            link = Link(cam=cam, keep=d.pid, drop=b.pid, track=b.track, death_last=d.last, birth_first=b.first,
                        dist_m=round(dist, 2), score=0.0)
            # kinematics
            kin = max(0.0, 1.0 - dist / max_m)
            link.score += kin
            link.reasons.append(f"kinematic {kin:.2f} ({dist:.1f} m)")
            # the identity's team
            td, tb = teams.get(d.pid), teams.get(b.pid)
            if td and tb and td != tb:
                link.reject = f"teams differ ({td} vs {tb})"
            elif td and tb:
                link.score += 1.0
                link.reasons.append(f"same team {td}")
            # the rows' kit votes
            kd, kb = _kit(track_rows[(d.pid, d.track)]["team"]), _kit(track_rows[(b.pid, b.track)]["team"])
            if kd and kb:
                if kd == kb:
                    link.score += 0.5
                    link.reasons.append(f"kit {kd} agrees")
                else:
                    link.score -= 0.5
                    link.reasons.append(f"kit differs ({kd} vs {kb})")
            # the jersey OCR
            if "jersey_number_ocr" in sub.columns:
                jd = _mode(track_rows[(d.pid, d.track)]["jersey_number_ocr"])
                jb = _mode(track_rows[(b.pid, b.track)]["jersey_number_ocr"])
                if jd is not None and jb is not None:
                    if jd == jb:
                        link.score += 1.0
                        link.reasons.append(f"jersey {jd} agrees")
                    else:
                        link.reject = link.reject or f"jerseys differ ({jd} vs {jb})"
            # contact at the death
            last_box = rows_at.get((d.last, d.pid))
            if last_box is not None:
                touching = [pid for pid, box in frame_rows.get(d.last, []) if pid != d.pid and _iou(last_box, box) > contact_iou]
                if touching:
                    link.score += 1.0
                    link.reasons.append(f"died in contact with {touching}")
            # the other camera: both ids seen apart at the same time there = two men; the dying id carrying on
            # there near the birth = one man
            if other_cam is not None:
                span_lo, span_hi = min(d.last, b.first), max(d.last, b.first) + 10
                both = [f for f in range(span_lo, span_hi + 1) if (f, d.pid) in other_ground and (f, b.pid) in other_ground]
                if both:
                    apart = float(np.median([np.linalg.norm(np.asarray(other_ground[(f, d.pid)]) - np.asarray(other_ground[(f, b.pid)]))
                                             for f in both]))
                    if apart > other_cam_apart_m:
                        link.reject = link.reject or f"{other_cam} sees both {apart:.1f} m apart"
                    else:
                        link.score += 0.5
                        link.reasons.append(f"{other_cam} sees both on one spot ({apart:.1f} m)")
                else:
                    near = [f for f in range(b.first, span_hi + 1) if (f, d.pid) in other_ground and (cam, f, b.pid) in ground
                            and np.linalg.norm(np.asarray(other_ground[(f, d.pid)]) - np.asarray(ground[(cam, f, b.pid)])) < other_cam_apart_m]
                    if near:
                        link.score += 1.0
                        link.reasons.append(f"{other_cam} carries {d.pid} on to the birth's spot")
            links.append(link)
    links.sort(key=lambda l: (l.reject is not None, -l.score))
    return links


def fold_commands(link: Link, play_dir: str) -> list[list[str]]:
    """The 08za / 08z argument lists that apply a link: the dying id's rows over the overlap are dropped first
    (they sit on the other man), then the born track is folded into the dying id."""
    cmds = []
    if link.overlap > 0:
        cmds.append(["scripts/08za_drop_rows.py", "--play-dir", play_dir, "--id", str(link.keep), "--cam", link.cam,
                     "--frames", str(link.birth_first), str(link.death_last), "--apply"])
    cmds.append(["scripts/08z_fold_ids.py", "--play-dir", play_dir, "--keep", str(link.keep), "--drop", str(link.drop),
                 "--cam", link.cam, "--track-id", str(link.track), "--apply"])
    return cmds
