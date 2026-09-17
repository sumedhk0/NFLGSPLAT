"""When is the play dead?  Writes ``<play-dir>/play_end.json`` (``{"end": frame, "tail": n, ...}``) for
05k to stop the clip at ``end + tail``.

The share of moving bodies does not mark it (play 1: 5-25 % of bodies move faster than 3 m/s DURING
the play, linemen engaged and backs covering, 50-80 % AFTER it, everyone jogging to the pile), and
the pile does not either (the line scrum forms at 400 and the runner runs to 490). What does: the
ball carrier stops. He is not labelled, so he is taken as the id that travels furthest from where it
stood at the snap over the live window; the play is dead at the first frame after which his speed
stays under ``--stop-m`` per frame for ``--stop-frames`` frames, or where his track ends (a tackle
takes him into a pile the tracker loses). ``--carrier`` and ``--end`` override the guesses.

  08x_play_end.py --play-dir P [--snap 300] [--live-hi 460] [--tail 30] [--carrier ID] [--end FRAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    ap.add_argument("--snap", type=int, default=300)
    ap.add_argument("--live-hi", type=int, default=460)
    ap.add_argument("--tail", type=int, default=30, help="frames drawn after the play is dead (0.5 s at 60 fps)")
    ap.add_argument("--stop-m", type=float, default=0.03)
    ap.add_argument("--stop-frames", type=int, default=10)
    ap.add_argument("--carrier", type=int, default=None, help="the ball carrier's id (default: furthest traveller)")
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
        tl, *_ = load_play_timeline(args.play_dir, model)
        pos = mr.positions_by_id(tl.states)
        if args.carrier is not None:
            pid, far = args.carrier, float("nan")
        else:
            got = carrier_by_travel(pos, args.snap, args.live_hi)
            if got is None:
                raise SystemExit("no id drawn from the snap for 30+ frames: cannot guess the carrier; pass --end")
            pid, far = got
        end = stop_frame(pos[pid], args.snap, stop_m=args.stop_m, stop_frames=args.stop_frames)
        last = max(pos[pid])
        out.update(end=int(end), carrier=int(pid), carrier_travel_m=round(far, 2), carrier_last_frame=int(last),
                   how="carrier stops" if end < last else "carrier's track ends")
    print(f"play dead at frame {out['end']} ({out['how']}" + (f"; carrier id {out['carrier']}, {out['carrier_travel_m']} m from the snap, drawn to {out['carrier_last_frame']}" if "carrier" in out else "") + f"); clip to {out['end'] + out['tail']}")
    if not args.dry_run:
        (args.play_dir / "play_end.json").write_text(json.dumps(out, indent=1))
        print(f"wrote {args.play_dir / 'play_end.json'}")


if __name__ == "__main__":
    main()
