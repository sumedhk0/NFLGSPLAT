"""The ball: where it is on every frame, written to ``<play-dir>/ball.json`` for 05k to draw.

No detector sees a football at this scale, so the ball is placed from the bodies and two events the
footage gives -- the release and the catch (frames read off the film, passed in). Before the snap
it sits at the centre's hand on the turf; at the snap it rises into the quarterback's hands and
travels with him; from the release it flies a ballistic arc (gravity, no drag) from his hand to the
receiver's hands at the catch; then it rides with whoever carries it, chained frame to frame to the
nearest body of the offence, and drops to the turf when the carrier is on the ground or the play is
dead. Every segment names its source so a viewer can be told what is measured and what is inferred.

  08y_ball_path.py --play-dir P --release 527 --catch 583 [--qb 49] [--receiver 77] [--down 640]
  [--snap 393] [--dry-run]

The flight is checked for plausibility (speed under 27 m/s, apex under 9 m); the script refuses
otherwise -- a wrong event frame shows up as an impossible throw. Runs under smplx312.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FPS = 59.94
G = 9.81
HAND_RELEASE_Z = 2.0        # the ball leaves the hand about here (m above the turf)
HAND_CATCH_Z = 1.4
CARRY_Z = 1.1               # tucked at the chest while running
CARRY_FWD_M = 0.30          # ... and this far in front of the body along its facing (inside the torso otherwise)
GROUND_Z = 0.12
CENTRE_HAND_Z = 0.15
SNAP_FRAMES = 5             # frames the ball takes from the centre's hand to the quarterback's
WINDUP_FRAMES = 8           # frames before the release the ball rises to the throwing hand
CHAIN_M = 1.5               # the carrier chain follows the nearest offence body within this per frame
MAX_SPEED = 30.0            # m/s (67 mph): the hardest arms reach 60-plus; play 1 measured 27.7 on a 25.6 m dart
MIN_SPEED = 8.0             # m/s: slower than a lob is not a pass (v76: a film-read "flight" of 2.1 m at 4 m/s from a
                            # lineman's box into the quarterback's own went straight into the render)
MAX_APEX = 9.0              # m above the chord: a punt, not a pass


def nearest(states, xy, team_of, team, *, exclude=()):
    best = None
    for s in states:
        pid = int(s.pid)
        if team_of.get(pid) != team or pid in exclude:
            continue
        d = float(np.linalg.norm(np.asarray(s.xy[:2], float) - xy))
        if best is None or d < best[0]:
            best = (d, pid, np.asarray(s.xy[:2], float))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--release", type=int, default=None, help="frame the ball leaves the passer's hand (from the footage, or --from-film)")
    ap.add_argument("--catch", type=int, default=None, help="frame it reaches the receiver's hands (or --from-film)")
    ap.add_argument("--qb", type=int, default=None, help="the passer's id at the release (default: the offence body nearest the pocket)")
    ap.add_argument("--receiver", type=int, default=None, help="the receiver's id at the catch (or --from-film)")
    ap.add_argument("--down", type=int, default=None, help="frame the carrier is down (default: his box on the ground, else the play's end)")
    ap.add_argument("--snap", type=int, default=None)
    ap.add_argument("--from-film", action="store_true",
                    help="take release, catch and receiver not given above from <play-dir>/ball_film.json (09a_ball_in_film.py: "
                         "the flight read off the sideline video), so nothing is typed from a render")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.from_film:
        fp = args.play_dir / "ball_film.json"
        if not fp.exists():
            raise SystemExit(f"--from-film: {fp} is missing; run scripts/09a_ball_in_film.py first")
        film = json.loads(fp.read_text())
        for key in ("release", "catch", "receiver"):
            if getattr(args, key) is None and film.get(key) is not None:
                setattr(args, key, int(film[key]))
                print(f"from the film: {key} {film[key]}")
    missing = [k for k in ("release", "catch") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"--{' and --'.join(missing)} needed: read them off the footage, or --from-film after 09a")
    import smplx

    from nfl_gsplat.render import timeline as tlm
    from nfl_gsplat.render.play_timeline import load_play_timeline

    P = args.play_dir
    pe = json.loads((P / "play_end.json").read_text()) if (P / "play_end.json").exists() else {}
    snap = args.snap if args.snap is not None else int(pe["snap"])
    start = int(pe.get("start", 0))
    end = int(pe["end"]) + int(pe.get("tail", 0))
    blob = pickle.load(open(P / "identity_resolved.pkl", "rb"))
    team_of = {int(p): v.team for p, v in blob["merged"].items()}
    roles = blob.get("roles", {})
    los = blob["line_of_scrimmage"]
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10, use_pca=False, batch_size=1)
    tl, tracks, df, frames_all, poses = load_play_timeline(P, model)
    lying = tlm.lying_frames(df)
    # the offence: the team whose linemen stand on the LOS's positive side
    st_snap = tl.states.get(snap, [])
    cnt: dict = {}
    for s in st_snap:
        if roles.get(int(s.pid)) == "OL":
            cnt[team_of.get(int(s.pid))] = cnt.get(team_of.get(int(s.pid)), 0) + 1
    offence = max(cnt, key=cnt.get)
    ol = {int(s.pid): float(s.xy[1]) for s in st_snap if team_of.get(int(s.pid)) == offence and roles.get(int(s.pid)) == "OL"}
    centre = min(ol, key=lambda p: abs(ol[p] - float(np.mean(list(ol.values())))))
    sign = float(los["sign"])
    print(f"offence {offence}, centre {centre}, snap {snap}, release {args.release}, catch {args.catch}")

    def body_xy(f, pid):
        for s in tl.states.get(f, []):
            if int(s.pid) == pid:
                return np.asarray(s.xy[:2], float)
        return None

    def hands_xy(f, pid, fwd_m=CARRY_FWD_M):
        """The body's xy pushed ``fwd_m`` along its facing: where a carried ball sits, not inside the torso."""
        for s in tl.states.get(f, []):
            if int(s.pid) == pid:
                yaw = tlm.yaw_of(s.global_orient)
                return np.asarray(s.xy[:2], float) + fwd_m * np.array([np.cos(yaw), np.sin(yaw)])
        return None

    path: dict = {}
    holder: dict = {}           # frame -> the id whose hands hold the ball (None in flight); 05k poses his arms
    # 1. before the snap: the centre's hand, on the turf, a third of a metre toward the line
    for f in range(start, snap):
        c = body_xy(f, centre)
        if c is None:
            continue
        path[f] = (float(c[0] - 0.35 * sign), float(c[1]), CENTRE_HAND_Z, "centre")
        # the centre is not a holder: his hand rests on a ball that stays on the turf (a carry pose
        # here would stand him up with the ball at his chest for the whole pre-snap)
    # 2. the passer: from the snap to the release
    qb = args.qb
    if qb is None:
        # the offence body 1-9 m behind the line at the release, nearest the centre's line
        st = tl.states.get(args.release, [])
        cands = [(abs(float(s.xy[1]) - ol[centre]), int(s.pid)) for s in st
                 if team_of.get(int(s.pid)) == offence and 1.0 <= (float(s.xy[0]) - los["x"]) * sign <= 9.0]
        qb = min(cands)[1]
    c_last = path.get(snap - 1, path.get(start))
    for f in range(snap, args.release):
        q = hands_xy(f, qb)
        if q is None:
            # the passer's id may change on the way back (play 1: 80 then 49): chain to the nearest offence body
            got = nearest(tl.states.get(f, []), np.asarray(path[f - 1][:2]), team_of, offence)
            if got is None or got[0] > CHAIN_M:
                continue
            qb = got[1]
            q = hands_xy(f, qb)
        z = CARRY_Z
        holder[f] = qb
        if f < snap + SNAP_FRAMES and c_last is not None:
            u = (f - snap + 1) / float(SNAP_FRAMES)
            x = c_last[0] + u * (q[0] - c_last[0]); y = c_last[1] + u * (q[1] - c_last[1]); z = CENTRE_HAND_Z + u * (CARRY_Z - CENTRE_HAND_Z)
            path[f] = (float(x), float(y), float(z), "snap")
            continue
        if f >= args.release - WINDUP_FRAMES:
            z = CARRY_Z + (HAND_RELEASE_Z - CARRY_Z) * (f - (args.release - WINDUP_FRAMES) + 1) / float(WINDUP_FRAMES)
        path[f] = (float(q[0]), float(q[1]), float(z), "passer")
    # 3. the flight: ballistic from the hand at the release to the hands at the catch
    p0 = hands_xy(args.release, qb, 0.4)
    receiver = args.receiver
    if receiver is None:
        raise SystemExit("--receiver is needed until the flight's end can name him; pass the id at the catch")
    if team_of.get(int(receiver)) != offence:
        # a completed pass ends in the offence's hands; the film reader once named the defender draped on the
        # receiver (BAL 55 for KC 74) and a lineman's neighbour the quarterback -- refuse rather than render it
        raise SystemExit(f"receiver {receiver} is {team_of.get(int(receiver))}, the offence is {offence}: check the receiver on the footage")
    p1 = hands_xy(args.catch, receiver)
    if p0 is None or p1 is None:
        raise SystemExit(f"passer {qb} at {args.release} or receiver {receiver} at {args.catch} is not drawn")
    T = (args.catch - args.release) / FPS
    dist = float(np.linalg.norm(p1 - p0))
    speed = np.hypot(dist / T, (HAND_CATCH_Z - HAND_RELEASE_Z) / T + 0.5 * G * T)
    apex = 0.5 * G * (T / 2) ** 2
    print(f"flight: {dist:.1f} m in {T:.2f} s from {np.round(p0, 1).tolist()} to {np.round(p1, 1).tolist()}: speed {speed:.1f} m/s, apex {apex:.1f} m above the chord")
    if speed > MAX_SPEED or apex > MAX_APEX or speed < MIN_SPEED:
        raise SystemExit(f"implausible flight ({speed:.1f} m/s, apex {apex:.1f} m): check --release / --catch / --receiver on the footage")
    for f in range(args.release, args.catch + 1):
        u = (f - args.release) / float(args.catch - args.release)
        t = u * T
        x = p0[0] + u * (p1[0] - p0[0]); y = p0[1] + u * (p1[1] - p0[1])
        z = HAND_RELEASE_Z + (HAND_CATCH_Z - HAND_RELEASE_Z) * u + 0.5 * G * t * (T - t)
        path[f] = (float(x), float(y), float(z), "flight")
    # 4. carried, chained to the nearest offence body; down when his box is on the ground or the play ends
    carrier = receiver
    down = args.down
    for f in range(args.catch + 1, end + 1):
        q = hands_xy(f, carrier)
        if q is None:
            got = nearest(tl.states.get(f, []), np.asarray(path[f - 1][:2]), team_of, offence)
            if got is None or got[0] > CHAIN_M:
                # the chain lost the carrier: his track ended (play 1: the detector loses the receiver under the
                # tackle at 602; the renderer holds his body through the tail). The ball stays in his hands --
                # "held", or "down" from the down frame (08x ends the play there). It stays in his hands on the
                # down frames too: the man drawn there is the last one the detector saw, standing or lying, and
                # 05k puts the ball between the holder's wrists; a ball dropping to the turf beside a standing
                # avatar (v72) read as a fumble.
                src = "down" if (down is not None and f >= down) else "held"
                path[f] = (path[f - 1][0], path[f - 1][1], path[f - 1][2], src)
                holder[f] = carrier
                continue
            carrier = got[1]
            q = hands_xy(f, carrier)
        on_ground = (down is not None and f >= down) or ((f, carrier) in lying)
        if on_ground and down is None:
            down = f
        path[f] = (float(q[0]), float(q[1]), float(CARRY_Z), "down" if on_ground else "carried")
        holder[f] = carrier
    # velocities for the ball's orientation
    fs = sorted(path)
    out = {}
    for i, f in enumerate(fs):
        a = path[fs[max(0, i - 1)]]; b = path[fs[min(len(fs) - 1, i + 1)]]
        n = max(1, fs[min(len(fs) - 1, i + 1)] - fs[max(0, i - 1)])
        v = [(b[0] - a[0]) / n, (b[1] - a[1]) / n, (b[2] - a[2]) / n]
        out[str(f)] = {"xyz": [round(path[f][0], 3), round(path[f][1], 3), round(path[f][2], 3)], "v": [round(c, 4) for c in v], "src": path[f][3],
                       "pid": (int(holder[f]) if holder.get(f) is not None else None)}
    segs = {}
    for f in fs:
        segs.setdefault(path[f][3], []).append(f)
    print("segments: " + ", ".join(f"{k} {min(v)}-{max(v)}" for k, v in segs.items()))
    print(f"passer {qb}, receiver {receiver}, carrier at the end {carrier}, down {down}")
    blob_out = {"fps": FPS, "snap": snap, "release": args.release, "catch": args.catch, "down": down, "passer": qb, "receiver": receiver,
                "centre": centre, "frames": out}
    if not args.dry_run:
        (P / "ball.json").write_text(json.dumps(blob_out))
        print(f"wrote {P / 'ball.json'} ({len(out)} frames)")


if __name__ == "__main__":
    main()
