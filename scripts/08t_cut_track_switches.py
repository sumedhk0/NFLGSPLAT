"""Cut a global id where its track switches to a different man, and give the tail its own id and team.

    python scripts/08t_cut_track_switches.py --play-dir P            # dry run
    python scripts/08t_cut_track_switches.py --play-dir P --apply    # split the tables, backups first

Two switch shapes, both found on play 1 (2026-09-15) by a plausibility ruler (per-id contiguous step;
nobody exceeds 0.20 m/frame = 12 m/s) and then traced to their line in the tables:

  SAME-CAMERA SWITCH. One camera's track jumps metres across a short detection gap: id 82's sideline sat at
  x = -23.1 through frame 307 and resumed at 315 at x = -28.9, 5.9 m away, 44 m/s. fill_gaps bridged the
  eight frames and the body was drawn crossing the field at 0.74 m/frame. Both cameras then agree on the NEW
  man, so no cross-camera test (08u) can see it; only the camera's own discontinuity can.

  WELDED TAIL. The sideline span ends and, after a gap, an ENDZONE span of a different man continues under
  the same id: id 19 = a Kansas City sideline track (14-430) plus a Baltimore endzone track (483-639). The
  tail's own kit majority reads BAL against the head's KC. The cut goes at the END OF THE SIDELINE SPAN --
  not at the frame where the drawn body's step peaks, which was interpolation racing toward the tail (my
  first cut was there and left the jump in place).

In both, EVERY row of the id after the cut frame moves to a fresh id in BOTH cameras (endzone rows with the
clip offset applied), keypoints_2d.parquet is split identically, and the tail gets a team from its OWN
kit_margin majority via dataclasses.replace on a copy of the source identity. Never a deletion: both halves
are real men.

Detection is :func:`plan_cuts`, pure over per-camera position dicts, so the tests exercise it directly.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.tracking.relabel import backup_path  # noqa: E402
from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.render.play_timeline import ankle_ground, clip_offset, ground_positions  # noqa: E402

FPS: float = 59.94
V_MAX: float = 12.0 / FPS       # 0.200 m/frame: no player exceeds this
MIN_JUMP_M: float = 1.0         # ... and a switch is a real distance, not one frame of box jitter
MAX_GAP: int = 30               # fill_gaps bridges gaps up to this; a switch across a longer gap is not drawn
TAIL_GAP: int = 1               # a tail is "disjoint" when it begins after the head's last frame


def per_id(ground) -> dict:
    out: dict = {}
    for f, d in ground.items():
        for pid, xy in d.items():
            out.setdefault(int(pid), {})[int(f)] = np.asarray(xy, float)
    return out


PERSIST: int = 5                # frames either side whose medians must stay apart: a blip is not a switch
MIN_JUMP_BY_CAM = {"sideline": 1.5, "endzone": 2.5}   # sideline: a wide-box re-lock moves a foot ~1 m;
                                                        # endzone: its depth noise blips ~1 m constantly


def same_camera_switches(track: dict, *, v_max: float = V_MAX, min_jump: float = MIN_JUMP_M,
                         max_gap: int = MAX_GAP, persist: int = PERSIST) -> list:
    """``[(cut_frame, jump_m, gap)]``: consecutive detections of one camera whose displacement over a
    short gap implies an impossible speed AND persists -- the median of the ``persist`` detections after
    the gap stays ``min_jump`` from the median of those before it. A one-frame blip that returns is
    camera noise (the endzone does this at ~1 m constantly) and is not a switch. The cut is at the frame
    BEFORE the gap."""
    fs = sorted(track)
    out = []
    for i, (a, b) in enumerate(zip(fs[:-1], fs[1:])):
        gap = b - a
        if gap < 1 or gap > max_gap:
            continue
        d = float(np.linalg.norm(track[b] - track[a]))
        if d < min_jump or d / gap <= v_max:
            continue
        pre = np.median(np.stack([track[f] for f in fs[max(0, i - persist + 1):i + 1]]), axis=0)
        post = np.median(np.stack([track[f] for f in fs[i + 1:i + 1 + persist]]), axis=0)
        if float(np.linalg.norm(post - pre)) >= min_jump:
            out.append((a, d, gap))
    return out


def plan_cuts(S: dict, E: dict, *, v_max: float = V_MAX, min_jump: float = MIN_JUMP_M,
              max_gap: int = MAX_GAP) -> list:
    """``[(pid, cut_frame, kind, detail)]``. Same-camera switches from either camera; welded tails as
    candidates (``kind == 'tail'``) for the caller to confirm by kit colour, since a tail of the SAME man is
    just a continuation and the ghost rule already handles it."""
    plan = []
    for pid in sorted(set(S) | set(E)):
        s, e = S.get(pid, {}), E.get(pid, {})
        for cam, t in (("sideline", s), ("endzone", e)):
            floor = max(min_jump, MIN_JUMP_BY_CAM.get(cam, min_jump))
            for f, d, gap in same_camera_switches(t, v_max=v_max, min_jump=floor, max_gap=max_gap):
                plan.append((pid, f, "switch", f"{cam} jumps {d:.1f} m across {gap} frames "
                                                f"({d / gap * FPS:.0f} m/s)"))
        if s and e:
            s_hi, e_lo = max(s), min(e)
            if e_lo > s_hi + TAIL_GAP and not (set(s) & set(e)):
                plan.append((pid, s_hi, "tail", f"endzone span {e_lo}-{max(e)} begins after the "
                                               f"sideline span ends at {s_hi}"))
    # one cut per id: the earliest, since everything after it moves anyway
    first: dict = {}
    for pid, f, kind, detail in plan:
        if pid not in first or f < first[pid][1]:
            first[pid] = (pid, f, kind, detail)
    return [first[p] for p in sorted(first)]


def team_from_kit(rows: pd.DataFrame, min_rows: int = 4):
    if "kit_margin" not in rows.columns:
        return None
    mg = rows["kit_margin"].to_numpy(float)
    mg = mg[np.isfinite(mg)]
    if len(mg) < min_rows:
        return None
    return "KC" if float(np.mean(mg > 0)) > 0.5 else "BAL"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--max-frame", type=int, default=None,
                    help="last LIVE frame of the play (timeline numbering). Switches after it are the "
                         "post-whistle crowd hopping between milling bodies -- real, but not worth "
                         "fragmenting. Required with --apply. Play 1: 470 (footage still live there).")
    args = ap.parse_args()
    if args.apply and args.max_frame is None:
        raise SetupError("08t: --apply needs --max-frame (the last live frame of the play)")
    P = args.play_dir
    tp, kp, ip = P / "tracks.parquet", P / "keypoints_2d.parquet", P / "identity_resolved.pkl"
    for need in (tp, kp, P / "cameras.npz"):
        if not need.exists():
            raise SetupError(f"08t: missing {need}")

    tracks = load_camera_track(P / "cameras.npz")
    raw = pd.read_parquet(tp)
    keys = pd.read_parquet(kp)
    offset = clip_offset(P)
    shift = {"endzone": offset} if offset else None
    df = raw[raw["track_id"] >= 0].copy()
    df.loc[df["cam"] == "endzone", "frame"] = df.loc[df["cam"] == "endzone", "frame"].astype(int) - offset
    kdf = keys.copy()
    kdf.loc[kdf["cam"] == "endzone", "frame"] = kdf.loc[kdf["cam"] == "endzone", "frame"].astype(int) - offset
    ank = ankle_ground(kdf, tracks, frame_shift=shift)
    S = per_id(ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ank, frame_shift=shift))
    E = per_id(ground_positions(df[df["cam"] == "endzone"], tracks, ankles=ank, frame_shift=shift))
    if args.only:
        S = {p: v for p, v in S.items() if p in args.only}
        E = {p: v for p, v in E.items() if p in args.only}

    blob = pickle.load(open(ip, "rb")) if ip.exists() else {}
    merged = blob.get("merged", {})

    def rows_after(pid, cut, cam):
        cam_cut = cut + offset if cam == "endzone" else cut
        return (raw["cam"] == cam) & (raw["global_player_id"] == pid) & (raw["frame"] > cam_cut)

    plan = []
    for pid, cut, kind, detail in plan_cuts(S, E):
        if args.max_frame is not None and cut > args.max_frame:
            continue                         # the post-whistle crowd
        head_team = getattr(merged.get(pid), "team", None)
        tail_team = team_from_kit(raw[rows_after(pid, cut, "sideline") | rows_after(pid, cut, "endzone")])
        if kind == "tail" and (tail_team is None or tail_team == head_team):
            continue                         # a continuation of the same man: not a switch
        plan.append((pid, cut, kind, detail, head_team, tail_team))

    nxt = max(int(raw["global_player_id"].max()), int(keys["global_player_id"].max())) + 1
    print(f"08t: bound {V_MAX:.3f} m/frame, min jump {MIN_JUMP_M} m; clip offset {offset:+d}; "
          f"{len(plan)} cuts")
    moved_total = 0
    for i, (pid, cut, kind, detail, ht, tt) in enumerate(plan):
        new = nxt + i
        m = rows_after(pid, cut, "sideline") | rows_after(pid, cut, "endzone")
        mk = pd.Series(False, index=keys.index)
        for cam in ("sideline", "endzone"):
            cam_cut = cut + offset if cam == "endzone" else cut
            mk |= (keys["cam"] == cam) & (keys["global_player_id"] == pid) & (keys["frame"] > cam_cut)
        moved_total += int(m.sum())
        flag = "   <-- CROSS-TEAM" if (ht and tt and ht != tt) else ""
        print(f"  id {pid:>3} {kind:>6} cut at {cut}: {int(m.sum())} track rows, {int(mk.sum())} keypoint "
              f"rows -> id {new}; head {ht}, tail {tt}   [{detail}]{flag}")
        if args.apply:
            raw.loc[m, "global_player_id"] = new
            keys.loc[mk, "global_player_id"] = new
            src = merged.get(pid)
            if src is not None and tt is not None:
                merged[new] = dataclasses.replace(src, team=tt)
    if not args.apply:
        print(f"dry run: {moved_total} track rows would move. Re-run with --apply.")
        return
    # A second apply (the fragments of the first pass carry switches of their own: play 1 needed
    # four passes to a fixpoint) must not overwrite the first pass's backup -- the only copy of the
    # tables before ANY cut. The backup name takes a counter when it is taken.
    names = [backup_path(f, ".pre08t") for f in (tp, kp)]
    for f, b in zip((tp, kp), names):
        shutil.copy2(f, b)
    raw.to_parquet(tp, index=False)
    keys.to_parquet(kp, index=False)
    if ip.exists():
        names.append(backup_path(ip, ".pre08t"))
        shutil.copy2(ip, names[-1])
        blob["merged"] = merged
        pickle.dump(blob, open(ip, "wb"))
    print(f"wrote {tp}, {kp}, {ip} (backups {', '.join(b.name for b in names)}); {moved_total} track rows moved")


if __name__ == "__main__":
    main()
