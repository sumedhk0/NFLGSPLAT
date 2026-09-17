"""When is the play dead?  Writes ``<play-dir>/play_end.json`` (``{"snap": f, "end": f, "tail": n, ...}``)
for 05k to stop the clip at ``end + tail`` and for 07l's live window.

The snap is where the LINE fires: the first frame from which at least SNAP_SHARE of the drawn bodies
move faster than MOVING_M for SNAP_HOLD frames, less SNAP_LEAD (play 1: 0.25 of the bodies move
before it -- a man in motion, a shifting defence, gliding fragments -- and 0.75-0.9 after it). The
play is dead when that share falls under DEAD_SHARE for DEAD_HOLD frames, and when it never does
before the clip ends, the clip's end is the dead ball: NFL Pro cuts the All-22 at the whistle (play
1: the tackle is at ~640 of 647 and the crowd runs to the last frame). ``--carrier ID`` uses that
id's stop instead (speed under --stop-m for --stop-frames frames, or its last confident sideline
keypoints); ``--end FRAME`` and ``--snap FRAME`` skip the guesses.

Retracted 2026-09-17: the ball carrier as "the id travelling furthest from its snap position" with
the snap at 300 chose play 1's motion man and ended the clip at 513, two seconds after the real
snap (~395) and in the middle of a pass play. The crowd's motion is the ruler; the footage checks it.

  08x_play_end.py --play-dir P [--snap F] [--tail 30] [--carrier ID] [--end FRAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOVING_M: float = 0.03        # m/frame (1.8 m/s at 59.94 fps): a body faster than this is moving
SNAP_SHARE: float = 0.4       # the snap: this share of the drawn bodies moving ...
SNAP_HOLD: int = 6            # ... for this many consecutive frames ...
SNAP_LEAD: int = 4            # ... and the ball moved this many frames before the line got going
SNAP_MIN_BODIES: int = 12     # frames with fewer drawn bodies do not vote


def moving_share(pos_by_id: dict, f: int, *, moving_m: float = MOVING_M, half: int = 1):
    """``(share, n_bodies)`` of the bodies drawn at ``f - half`` and ``f + half`` whose speed over
    that span exceeds ``moving_m`` per frame."""
    n = k = 0
    for byf in pos_by_id.values():
        a, b = byf.get(f - half), byf.get(f + half)
        if a is None or b is None:
            continue
        n += 1
        k += float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float))) / (2 * half) > moving_m
    return (k / n if n else 0.0), n


def snap_from_motion(pos_by_id: dict, *, lo: int | None = None, hi: int | None = None, moving_m: float = MOVING_M,
                     share: float = SNAP_SHARE, hold: int = SNAP_HOLD, lead: int = SNAP_LEAD,
                     min_bodies: int = SNAP_MIN_BODIES):
    """The snap: ``lead`` frames before the first frame from which at least ``share`` of the drawn
    bodies move faster than ``moving_m`` for ``hold`` consecutive frames. Before the snap a man in
    motion, a shifting defence and a couple of gliding fragments move (play 1: at most 6 of 22
    bodies); at the snap the whole line fires (11 of 23 within 5 frames, 18 of 24 within 20).
    ``None`` when the play never starts. identity.roles.snap_frame uses a 0.5 m/s "static" cut that
    the placement's own jitter crosses, and put play 1's snap at 300 where the footage has 395."""
    frames = sorted({f for byf in pos_by_id.values() for f in byf})
    if not frames:
        return None
    lo = frames[0] if lo is None else lo
    hi = frames[-1] if hi is None else hi
    run = 0
    for f in range(lo + 1, hi):
        sh, n = moving_share(pos_by_id, f, moving_m=moving_m)
        run = run + 1 if (n >= min_bodies and sh >= share) else 0
        if run >= hold:
            return max(lo, f - hold + 1 - lead)
    return None


START_BEFORE: int = 180       # the clip starts this many frames (3 s) before the snap: the formation and the motion
DEAD_SHARE: float = 0.4       # the play is dead once fewer than this share of the bodies move ...
DEAD_HOLD: int = 30           # ... for this many consecutive frames ...
DEAD_EARLIEST: int = 60       # ... looked for from this many frames after the snap


def dead_from_motion(pos_by_id: dict, snap: int, *, moving_m: float = MOVING_M, share: float = DEAD_SHARE,
                     hold: int = DEAD_HOLD, earliest: int = DEAD_EARLIEST, min_bodies: int = SNAP_MIN_BODIES):
    """The first frame from which fewer than ``share`` of the drawn bodies move for ``hold`` frames,
    searched from ``snap + earliest``; ``None`` when the crowd runs to the end of the clip."""
    frames = sorted({f for byf in pos_by_id.values() for f in byf})
    if not frames:
        return None
    run = 0
    for f in range(snap + earliest, frames[-1]):
        sh, n = moving_share(pos_by_id, f, moving_m=moving_m)
        run = run + 1 if (n >= min_bodies and sh < share) else 0
        if run >= hold:
            return f - hold + 1
    return None


def carrier_by_travel(pos_by_id: dict, snap: int, hi: int, *, min_frames: int = 30):
    """``(pid, metres)`` of the id drawn at the snap that gets furthest from its snap position by ``hi``."""
    best = None
    for pid, byf in pos_by_id.items():
        fs = [f for f in sorted(byf) if snap <= f <= hi]
        if len(fs) < min_frames or fs[0] > snap + 5:
            continue
        x0 = np.asarray(byf[fs[0]], float)
        far = max(float(np.linalg.norm(np.asarray(byf[f], float) - x0)) for f in fs)
        if best is None or far > best[1]:
            best = (int(pid), far)
    return best


def stop_frame(byf: dict, start: int, *, stop_m: float = 0.03, stop_frames: int = 10):
    """The first frame from ``start`` after which the id's speed stays under ``stop_m`` for
    ``stop_frames`` consecutive frames; the id's last frame when it never stops before ending."""
    fs = sorted(f for f in byf if f >= start)
    if len(fs) < 3:
        return fs[-1] if fs else start
    speeds = {}
    for a, c in zip(fs[:-2], fs[2:]):
        if c - a == 2:
            speeds[a + 1] = float(np.linalg.norm(np.asarray(byf[c], float) - np.asarray(byf[a], float)) / 2)
    run = 0
    for f in fs[1:-1]:
        if f not in speeds:
            run = 0
            continue
        run = run + 1 if speeds[f] < stop_m else 0
        if run >= stop_frames:
            return f - stop_frames + 1
    return fs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--snap", type=int, default=None, help="snap frame (default: found from the bodies' motion)")
    ap.add_argument("--live-hi", type=int, default=None, help="unused since the crowd rule; kept for the pipeline's call")
    ap.add_argument("--tail", type=int, default=30, help="frames drawn after the play is dead (0.5 s at 60 fps)")
    ap.add_argument("--stop-m", type=float, default=0.03)
    ap.add_argument("--stop-frames", type=int, default=10)
    ap.add_argument("--carrier", type=int, default=None, help="end the play where this id stops instead of where the crowd does")
    ap.add_argument("--end", type=int, default=None, help="the dead-ball frame itself (skips the guess)")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = {"snap": args.snap, "tail": args.tail}
    if args.end is not None:
        out.update(end=int(args.end), how="given")
    else:
        import smplx

        from nfl_gsplat.render import motion_rulers as mr
        from nfl_gsplat.render.play_timeline import load_play_timeline

        model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10,
                             use_pca=False, batch_size=1)
        tl, _tracks, df, _frames, _poses = load_play_timeline(args.play_dir, model)
        pos = mr.positions_by_id(tl.states)
        if args.snap is None:
            args.snap = snap_from_motion(pos)
            if args.snap is None:
                raise SystemExit("no snap found: the bodies never get going together; pass --snap")
            print(f"snap from the bodies' motion: frame {args.snap}")
            out["snap"] = int(args.snap)
        last_frame = max(max(byf) for byf in pos.values())
        if args.carrier is not None:
            pid = int(args.carrier)
            end = stop_frame(pos[pid], args.snap, stop_m=args.stop_m, stop_frames=args.stop_frames)
            last = max(pos[pid])
            # the box tracker outlives the detector's pose (play 1: boxes on 9 to 493, keypoints to 482), so the
            # last sighting is the last frame with confident hips or ankles in the keypoint table
            import pandas as pd

            kp = pd.read_parquet(args.play_dir / "keypoints_2d.parquet")
            kp = kp[(kp["cam"] == "sideline") & (kp["global_player_id"] == pid) & kp["joint"].isin([11, 12, 15, 16]) & (kp["conf"] >= 0.5)]
            last_seen = int(kp["frame"].max()) if len(kp) else last
            how = "carrier stops" if end < min(last, last_seen) else ("carrier last seen" if last_seen < last else "carrier's track ends")
            end = min(end, last_seen)
            out.update(end=int(end), carrier=pid, carrier_last_frame=int(last), carrier_last_seen=int(last_seen), how=how)
        else:
            end = dead_from_motion(pos, args.snap)
            if end is None:
                out.update(end=int(last_frame), tail=0, how="clip end (the crowd runs to the last frame)")
            else:
                out.update(end=int(end), how="crowd stops")
    if out.get("snap") is not None:
        out["start"] = max(0, int(out["snap"]) - START_BEFORE)
    print(f"clip from {out.get('start', 0)}; snap {out['snap']}; play dead at frame {out['end']} ({out['how']}" + (f"; carrier id {out['carrier']}, drawn to {out['carrier_last_frame']}" if "carrier" in out else "") + f"); clip to {out['end'] + out['tail']}")
    if not args.dry_run:
        (args.play_dir / "play_end.json").write_text(json.dumps(out, indent=1))
        print(f"wrote {args.play_dir / 'play_end.json'}")


if __name__ == "__main__":
    main()
