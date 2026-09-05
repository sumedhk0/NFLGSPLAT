#!/usr/bin/env python
"""Reconstruct one production All-22 play end to end, from the video alone.

Every stage of this has been measured on its own; this is where they meet on the
footage the project exists for. The chain is:

    paint    -> sideline cameras, per frame   calibrate_clip, no tracking
    YOLO     -> players per view              boxes; feet = bottom-centre
    players  -> endzone cameras, per frame    from_players: the sideline puts
                                              each player on the turf, the
                                              endzone sees the same people;
                                              mount from a coarse grid, lens
                                              from the boxes, rotation solved
    agreement -> which sideline candidate     the pair that reconciles the
                                              most players, in metres
    placement -> ground positions             midpoint of the two views'
                                              ground points per player

WHY THE ENDZONE IS NOT SOLVED FROM PAINT. It cannot be, and this was measured
rather than assumed: twenty-four paint solves of the endzone clip produced two
cameras, both impossible, and neither reconciled a single player with any
sideline candidate. Look at a frame: it is a high, steep, tightly zoomed shot
that pans with the play and shows about twelve metres of field width.

THERE IS NO GROUND TRUTH HERE, which is the point. So the result is checked
against things that must be true regardless:

    reconciled    two independent cameras put the same players in the same
                  places -- and the ceiling is what the tight endzone frame
                  holds, about 14-18 of 22, never all of them
    gap           how far apart the two views put the same player, in metres
    on the field  placements land inside the boundary
    head height   with feet on the turf, each view's head ray implies a height;
                  it must come out near 1.85 m in BOTH views, and nothing in the
                  pipeline forces it to
    spacing       real players stand about 1-3 m apart

The two clips are assumed to start together; they are the same play at the same
length. Nothing here verifies the offset, and an offset sweep on this play found
none.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.calibrate_clip import (
    MAX_PLAYER_COST,
    candidates_for_video,
)
from nfl_gsplat.calibration.from_players import feet_of, solve_second_view
from nfl_gsplat.calibration.joint_views import match_count
from nfl_gsplat.calibration.player_scale import implied_heights
from nfl_gsplat.errors import CalibrationError

EXPECTED_PLAYER_M = 1.85
FIELD_HALF_X_M = 56.0
FIELD_HALF_Y_M = 26.0


def boxes_by_frame(model, path, frames, *, conf=0.15):
    """Person boxes on the requested frames, plus the frame size."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        res = model.predict(img, classes=[0], conf=conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        out[int(f)] = res.boxes.xyxy.cpu().numpy()
    cap.release()
    return out, w, h


def nearest(cams, f):
    return cams[min(cams, key=lambda k: abs(k - f))]


def fov_deg(K, width):
    return float(2.0 * np.degrees(np.arctan(width / (2.0 * K[0, 0]))))


def sideline_candidate_from_track(play_dir: Path):
    """One candidate in the shape candidates_for_video returns, from a
    play-dir's sideline CameraTrack (every frame with conf > 0)."""
    from nfl_gsplat.calibration.cameras_io import load_camera_track

    track = load_camera_track(play_dir / "cameras.npz")["sideline"]
    cams = {int(f): (track.K[f], track.R[f], track.t[f])
            for f in np.flatnonzero(track.conf > 0)}
    if not cams:
        raise SystemExit(f"{play_dir / 'cameras.npz'} has no sideline frame with a camera")
    K, R, t = cams[min(cams)]
    return {"cams": cams, "centre": -R.T @ t,
            "quality": {"player_cost": 0.0, "fov_deg": fov_deg(K, track.width),
                        "coverage": float("nan")}}


def numeral_readability(video_path, cams, frames, *, reader=None):
    """Summed confidence of every numeral read through ``cams`` on ``frames``.

    The field is mirror-symmetric except for the glyphs: a camera behind the
    WRONG end zone sees every numeral mirrored, and mirrored digits do not
    read. So the mount side that reads more is the right one -- the ground
    cloud alone cannot say (both ends reconcile the same players at the
    same gap).
    """
    import cv2

    from nfl_gsplat.field import yard_numbers as yn

    if reader is None:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    cap = cv2.VideoCapture(str(video_path))
    total, n = 0.0, 0
    for f in frames:
        if f not in cams:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        K, Rm, t = cams[f]
        for r in yn.read_line_strips(img, K, Rm, t, reader):
            total += r.conf
            n += 1
    cap.release()
    return total, n


def mirrored_mount(mount):
    """The other end of the field: the paint's unresolvable symmetry is the
    half turn about the centre, ``(x, y) -> (-x, -y)``, not a mirror in x."""
    x, y, z = mount
    return (-float(x), -float(y), float(z))


# The mirror wins only when it is a comparable solve AND clearly more
# readable: within these fractions of the original's reconciliation and gap,
# and reading more numerals by a margin (summed confidence x1.5 or two more
# readings). Summed confidence alone let one arrow read as a '1' decide.
MIRROR_RECON_FRAC = 0.85
MIRROR_GAP_FRAC = 1.25
MIRROR_READ_FACTOR = 1.5
MIRROR_READ_EXTRA = 2


# The two paint rulers must agree on a sideline candidate's cross-field scale
# this well, and the scale must be this close to 1, for the candidate to go
# on to an endzone solve. Play 2 (fresh, corrected hash constant): candidate
# [1] read hashes 1.000 / numerals 1.052 and was the right camera; candidate
# [2] read 3.02 / 1.26, a 23-degree lens 55 m from the field, and WON the
# endzone-reconciliation ranking (14 per frame at 0.97 m). Feet from two
# views do not veto a wrong sideline; the rows on the turf do.
RULER_AGREE = 0.10
RULER_SCALE = (0.80, 1.25)
LENS_BAND = 1.6          # a row labelling may imply a lens this far from the seed play's
LOOSE_PLAYER_COST = 0.75


def ruler_scales(video_path, cams, frames, *, reader=None):
    """``(scale, by_ruler, n_readings)`` of the hash and numeral rows read
    through ``cams`` on ``frames``; ``by_ruler`` maps ruler -> its scale."""
    import cv2

    from nfl_gsplat.calibration import row_ruler as rr
    from nfl_gsplat.field import yard_numbers as yn

    if reader is None:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    cap = cv2.VideoCapture(str(video_path))
    ys, yt, rl = [], [], []
    for f in frames:
        f = min(cams, key=lambda k: abs(k - f))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        K, Rm, t = cams[f]
        for r in yn.read_line_strips(img, K, Rm, t, reader):
            if getattr(r, "weak", False):
                continue
            ys.append(r.y_m)
            yt.append(r.side * rr.ROW_Y_M)
            rl.append("numerals")
        for y, side in rr.measure_hash_rows(img, K, Rm, t):
            ys.append(y)
            yt.append(side * rr.HASH_Y_M)
            rl.append("hashes")
    cap.release()
    if len(ys) < 4:
        return float("nan"), {}, len(ys)
    fit = rr.fit_rows(ys, yt, rulers=rl)
    return float(fit.scale), dict(fit.by_ruler or {}), len(ys)


def select_by_rulers(scales, *, agree=RULER_AGREE, scale_range=RULER_SCALE):
    """Indices of candidates whose two rulers agree and whose scale is near 1,
    best first (closest to 1). ``scales`` is ``[(scale, by_ruler, n)]``.
    A candidate with only one ruler read is kept only if that ruler is in
    range; one with none read is dropped."""
    keep = []
    for i, (scale, by, n) in enumerate(scales):
        if not np.isfinite(scale) or n == 0:
            continue
        vals = [v for v in by.values() if np.isfinite(v)]
        if len(vals) >= 2 and abs(vals[0] - vals[1]) > agree * max(vals):
            continue
        if not (scale_range[0] <= scale <= scale_range[1]):
            continue
        keep.append((abs(scale - 1.0), i))
    return [i for _d, i in sorted(keep)]


# Player height a sideline candidate must imply (median over boxes): the
# rulers pass a camera on the wrong lens/distance branch (play 3: rulers
# 0.99/0.99, players 2.77 m); the players catch that.
HEIGHT_RANGE_M = (1.55, 2.15)


def select_candidates(scales, heights, grids, *, agree=RULER_AGREE, scale_range=RULER_SCALE,
                      height_range=HEIGHT_RANGE_M, max_grid_px=None):
    """Indices passing all three judges, best first by ruler scale.

    ``scales``: ``[(scale, by_ruler, n)]`` from the paint rows; ``heights``:
    median implied player height per candidate (m); ``grids``: median pixel
    distance of the projected 5-yard lines from the painted ones (1080p).
    Rulers test row positions along the lines the camera believes in, the
    height tests the lens, the grid tests whether the lines are on the paint
    at all; play 2 passed the first two with a grid 149 px off.
    """
    from nfl_gsplat.calibration.grid_fit import MAX_GRID_PX_1080

    max_grid_px = MAX_GRID_PX_1080 if max_grid_px is None else max_grid_px
    order = select_by_rulers(scales, agree=agree, scale_range=scale_range)
    out = []
    for i in order:
        h = heights[i]
        g = grids[i]
        if np.isfinite(h) and not (height_range[0] <= h <= height_range[1]):
            continue
        if np.isfinite(g) and g > max_grid_px:
            continue
        out.append(i)
    return out


def candidate_heights(cams, boxes_by_frame_):
    """Median implied player height (m) of a candidate over the sampled frames."""
    hs = []
    for f, boxes in boxes_by_frame_.items():
        K, Rm, t = nearest(cams, f)
        h = np.asarray(implied_heights(K, Rm, t, boxes))
        hs.extend(h[(h > 0.5) & (h < 4.0)].tolist())
    return float(np.median(hs)) if hs else float("nan")


def frame_count(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("data/all22/bal_at_kc_2024_wk1"))
    ap.add_argument("--sideline", default="001_Sideline_KC_2-20_BLT_24.mp4")
    ap.add_argument("--endzone", default="002_Endzone_KC_2-20_BLT_24.mp4")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--vertical-deg", type=float, default=45.0,
                    help="how far from vertical a segment may lean and still "
                         "be a yard line. 35 (the detector default) left a "
                         "dead zone 35-55 deg; play 2's red-zone view put its "
                         "yard lines there and solved nothing. 45 solved it.")
    ap.add_argument("--calib-frames", type=int, default=26)
    ap.add_argument("--play-frames", type=int, default=24)
    ap.add_argument("--top", type=int, default=3,
                    help="sideline candidates to try, best player cost first; "
                         "each costs an endzone solve of several minutes")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--sideline-from", type=Path, default=None,
                    help="play-dir whose cameras.npz sideline track is used as the one "
                         "sideline candidate (e.g. after scripts/08d refined it) instead "
                         "of solving from paint; the endzone is then solved against it")
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="a play-dir of the SAME game whose solved sideline camera gives the mount: "
                         "the joint solve HOLDS the centre there and fits rotation and focal per "
                         "frame to the paint (seeding the start alone was measured not to help: the "
                         "paint's minimum is not at the mount)")
    ap.add_argument("--no-ruler-gate", action="store_true",
                    help="skip reading the hash and numeral rows through each sideline "
                         "candidate before the endzone solves")
    ap.add_argument("--no-mirror-check", action="store_true",
                    help="skip the second endzone solve at the mirrored mount and the "
                         "numeral read that decides which end the camera is behind")
    ap.add_argument("--out", type=Path,
                    default=Path("C:/Users/sumedh/diag/all22_reconstruction.npz"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    side_path = args.root / args.sideline
    end_path = args.root / args.endzone

    print(f"Sideline: {args.sideline}")
    if args.sideline_from is not None:
        cands = [sideline_candidate_from_track(args.sideline_from)]
        print(f"   sideline camera taken from {args.sideline_from / 'cameras.npz'}")
    else:
        seed = None
        if args.seed_from is not None:
            from nfl_gsplat.calibration.cameras_io import load_camera_track

            tr = load_camera_track(args.seed_from / "cameras.npz")["sideline"]
            ok = np.flatnonzero(tr.conf > 0)
            seed = np.median(np.stack([-tr.R[f].T @ tr.t[f] for f in ok]), axis=0)
            fov_seed = float(np.median([fov_deg(tr.K[f], tr.width) for f in ok]))
            fov_band = (fov_seed / LENS_BAND, fov_seed * LENS_BAND)
            print(f"   joint solve holds the centre at the sideline mount of {args.seed_from.name}: "
                  f"{np.round(seed, 1)} m; row labellings must imply a lens within "
                  f"{fov_band[0]:.1f}..{fov_band[1]:.1f} deg (seed {fov_seed:.1f})")
        else:
            fov_band = None
        cands = candidates_for_video(side_path, attempts=args.attempts,
                                     n_frames=args.calib_frames, model=model,
                                     vertical_deg=args.vertical_deg, fixed_centre=seed,
                                     fov_band=fov_band)
    if not cands:
        raise SystemExit("no sideline camera could be solved from paint")
    # Players gate the pool, as they gate the single-clip path. On play 2 the
    # best-ranked candidate (8.3 deg, 70 m up, player cost 0.70) put every
    # player at an impossible height and the render laid every body flat on
    # the turf -- body orientation comes through the camera. Prefer the
    # candidates the players believe; fall back to all only if none do.
    # With the ruler gate on, the player cost is a loose pre-filter: play 2's
    # right and wrong sideline cameras scored 0.45 and 0.58, both under the
    # gate, and play 3's only candidates scored 0.62 -- the rows on the turf
    # decide, the players only weed out the impossible.
    player_gate = MAX_PLAYER_COST if args.no_ruler_gate else LOOSE_PLAYER_COST
    believed = [c for c in cands
                if c["quality"]["player_cost"] <= player_gate]
    if not believed:
        raise SystemExit(
            f"no sideline candidate passes the player gate ({player_gate}); "
            f"best {min(c['quality']['player_cost'] for c in cands):.2f}. "
            "Refusing: a camera the players do not believe lays every body "
            "flat in the render. The sideline paint solve is the weak stage "
            "on this clip.")
    cands = believed
    print(f"   {len(cands)} candidate cameras")

    # The same moments in both clips, away from the ends of the play.
    total = min(frame_count(side_path), frame_count(end_path))
    want = np.linspace(int(0.15 * total), int(0.85 * total),
                       args.play_frames).astype(int)

    boxes_s, _w_s, _h_s = boxes_by_frame(model, side_path, want)
    boxes_e, w_e, h_e = boxes_by_frame(model, end_path, want)

    # Three judges on every candidate before any endzone solve: the paint
    # rows (cross-field scale), the players (lens/distance), and the grid on
    # the paint (is the camera on the field at all). Each has caught a camera
    # the other two passed.
    # The judges rule on ONE candidate as much as on many: a single candidate
    # from a held mount (--seed-from) or --sideline-from skipped them and a
    # 2.81 m camera went through on play 3 (2026-09-05).
    if not args.no_ruler_gate and len(cands) >= 1:
        from nfl_gsplat.calibration.grid_fit import MAX_GRID_PX_1080, grid_scores

        probe = [int(f) for f in np.linspace(0.2, 0.8, 5) * frame_count(side_path)]
        scales = [ruler_scales(side_path, c["cams"], probe) for c in cands]
        heights = [candidate_heights(c["cams"], boxes_s) for c in cands]
        grids = [grid_scores(side_path, c["cams"], probe)[0] for c in cands]
        for i, c in enumerate(cands):
            sc, by, n = scales[i]
            print(f"   [{i}] {np.round(c['centre'], 1)} {c['quality']['fov_deg']:.1f} deg: rulers "
                  + ", ".join(f"{k} {v:.3f}" for k, v in by.items())
                  + f" -> scale {sc:.3f} ({n} readings); players {heights[i]:.2f} m; "
                  f"grid {grids[i]:.1f} px; paint rms {c['quality'].get('rms_px', float('nan')):.1f} px")
        order = select_candidates(scales, heights, grids)
        if not order:
            raise SystemExit("no sideline candidate passes the gates (rulers agree within "
                             f"{RULER_AGREE:.0%} with scale in {RULER_SCALE}; players "
                             f"{HEIGHT_RANGE_M} m; grid within {MAX_GRID_PX_1080} px of the "
                             "paint); the sideline paint solve is wrong on this clip")
        cands = [cands[i] for i in order]
        print(f"   {len(cands)} pass; order {order}")
    feet_s = {f: feet_of(b) for f, b in boxes_s.items()}
    feet_e = {f: feet_of(b) for f, b in boxes_e.items()}
    print(f"   detections per frame: sideline "
          f"{np.median([len(b) for b in boxes_s.values()]):.0f}, endzone "
          f"{np.median([len(b) for b in boxes_e.values()]):.0f}")

    # Choose the sideline candidate by what the OTHER view can reconcile.
    best = None
    for i, cand in enumerate(cands[:args.top]):
        cams_s = cand["cams"]
        q = cand["quality"]
        head = (f"   [{i}] sideline {np.round(cand['centre'], 1)} "
                f"{q['fov_deg']:.1f} deg, players {q['player_cost']:.2f}, "
                f"sees {100 * q['coverage']:.0f}%")
        try:
            cams_e, info = solve_second_view(cams_s, feet_s, boxes_e, w_e, h_e)
        except CalibrationError as exc:
            print(f"{head}: {str(exc)[:80]}")
            continue
        print(f"{head} -> endzone mount {info['mount']}, {info['frames']} "
              f"frames, reconciles {info['reconciled']:.0f}/frame at "
              f"{info['gap_m']:.2f} m, players {info['player_cost']:.2f} "
              f"({info['height_m']:.2f} m)")
        key = (info["reconciled"], -info["gap_m"])
        if best is None or key > best[0]:
            best = (key, cand, cams_e, info)
    if best is None:
        raise SystemExit("no sideline candidate let the players calibrate "
                         "the endzone")
    _key, cand, cams_e, info = best
    cams_s = cand["cams"]

    # The other end of the field: the same players reconcile at the same
    # gap from behind either end zone (measured: +60 and -95 m, 0.90 and
    # 0.91 m), so the side is decided by the numerals, which only read
    # unmirrored. One extra endzone solve.
    if not args.no_mirror_check:
        try:
            cams_m, info_m = solve_second_view(cams_s, feet_s, boxes_e, w_e, h_e,
                                               mounts=[mirrored_mount(info["mount"])])
        except CalibrationError as exc:
            print(f"   mirror mount {mirrored_mount(info['mount'])}: {str(exc)[:80]}")
            cams_m = None
        if cams_m is not None:
            common = sorted(set(cams_e) & set(cams_m))
            probe = common[::max(1, len(common) // 6)][:6]
            read_a, n_a = numeral_readability(end_path, cams_e, probe)
            read_b, n_b = numeral_readability(end_path, cams_m, probe)
            comparable = (info_m["reconciled"] >= MIRROR_RECON_FRAC * info["reconciled"]
                          and info_m["gap_m"] <= MIRROR_GAP_FRAC * info["gap_m"])
            clearer = (read_b >= MIRROR_READ_FACTOR * read_a) or (n_b >= n_a + MIRROR_READ_EXTRA)
            print(f"   mount side by numerals on {len(probe)} shared frames: {info['mount']} "
                  f"reads {n_a} ({read_a:.1f}), mirror {info_m['mount']} reads {n_b} ({read_b:.1f}); "
                  f"mirror reconciles {info_m['reconciled']:.0f}/frame at {info_m['gap_m']:.2f} m "
                  f"({'comparable' if comparable else 'worse'})")
            if comparable and clearer and read_b > read_a:
                print("   -> the mirror is a comparable solve and reads clearly better; taking it")
                cams_e, info = cams_m, info_m

    print("\nchosen cameras")
    print(f"   sideline {np.round(cand['centre'], 1)} m, "
          f"{cand['quality']['fov_deg']:.1f} deg, per-frame, "
          f"{len(cams_s)} frames")
    fovs = [fov_deg(K, w_e) for (K, _R, _t) in cams_e.values()]
    print(f"   endzone  mount {info['mount']} (a prior; the distance is not "
          f"observable from feet), lens {min(fovs):.1f}..{max(fovs):.1f} deg "
          f"per frame, {len(cams_e)} frames")

    # Place every reconciled player, and check what needs no ground truth.
    placed_all, heights_s, heights_e, spacing, per = [], [], [], [], []
    for f in want:
        f = int(f)
        if f not in feet_s or f not in feet_e or f not in cams_e:
            continue
        n, placed = match_count(nearest(cams_s, f), feet_s[f], cams_e[f],
                                feet_e[f])
        per.append(n)
        if n:
            placed_all.append(placed)
            if n > 2:
                d = np.linalg.norm(placed[:, None] - placed[None], axis=2)
                np.fill_diagonal(d, np.inf)
                spacing.append(np.median(d.min(1)))
        heights_s.extend(implied_heights(*nearest(cams_s, f), boxes_s[f]))
        heights_e.extend(implied_heights(*cams_e[f], boxes_e[f]))

    if not placed_all:
        raise SystemExit("the chosen pair placed nobody")
    allxy = np.concatenate(placed_all)
    inside = ((np.abs(allxy[:, 0]) <= FIELD_HALF_X_M)
              & (np.abs(allxy[:, 1]) <= FIELD_HALF_Y_M))
    hs, he = np.asarray(heights_s), np.asarray(heights_e)
    hs = hs[(hs > 0.5) & (hs < 4.0)]
    he = he[(he > 0.5) & (he < 4.0)]

    print("\nchecks that need no ground truth")
    print(f"   reconciled        median {np.median(per):.0f} players per frame "
          f"of {np.median([len(b) for b in boxes_e.values()]):.0f} detected  "
          f"{per}")
    print(f"   gap               {info['gap_m']:.2f} m between the views, "
          "same player, median")
    print(f"   on the field      {inside.mean():5.0%} of placements")
    print(f"   player height     sideline median {np.median(hs):4.2f} m, "
          f"endzone median {np.median(he):4.2f} m  (expected ~{EXPECTED_PLAYER_M})")
    print(f"   footprint         x {allxy[:, 0].min():6.1f}..{allxy[:, 0].max():6.1f}"
          f"   y {allxy[:, 1].min():6.1f}..{allxy[:, 1].max():6.1f} m")
    if spacing:
        print(f"   nearest neighbour median {np.median(spacing):4.2f} m "
              "(real players stand about 1-3 m apart)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fs = sorted(cams_s)
    fe = sorted(cams_e)
    np.savez(args.out,
             side_frames=np.array(fs),
             side_K=np.array([cams_s[f][0] for f in fs]),
             side_R=np.array([cams_s[f][1] for f in fs]),
             side_t=np.array([cams_s[f][2] for f in fs]),
             end_frames=np.array(fe),
             end_K=np.array([cams_e[f][0] for f in fe]),
             end_R=np.array([cams_e[f][1] for f in fe]),
             end_t=np.array([cams_e[f][2] for f in fe]),
             reconciled=np.array(per),
             placed=allxy)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
