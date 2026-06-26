"""Bring-up diagnostic for automatic field calibration.

Extracts one frame from a play's clip, runs the current line detector + white
mask, and prints detected yard-line x-positions so you can pick a reference
x for the calib_hints block in meta.yaml.

    python scripts/diag_calib.py --play-dir data/2025/week_04/SEA_at_AZ/play_001 --frame 0

Saves <out-dir>/diag_<cam>_f<NNNNN>.png and diag_<cam>_f<NNNNN>_mask.png;
prints detected yard-line x-positions. No display / GPU required.
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _player_boxes(img, weights: str):
    """YOLOv8 person boxes [(x1,y1,x2,y2), ...] for masking field detection."""
    from ultralytics import YOLO  # type: ignore
    model = YOLO(weights)
    res = model.predict(img, classes=[0], conf=0.25, verbose=False)[0]
    return [tuple(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()]


@app.command()
def main(
    play_dir: Path = typer.Option(..., "--play-dir"),
    frame: int = typer.Option(0, "--frame"),
    cam: str = typer.Option("sideline", "--cam"),
    out_dir: Path = typer.Option(Path("/tmp"), "--out-dir"),
    mask: bool = typer.Option(False, "--mask/--no-mask", help="YOLO-mask players before detection"),
    yolo_weights: str = typer.Option("data/body_models/yolov8x.pt", "--yolo-weights"),
    ref_x: float = typer.Option(None, "--ref-x", help="hint: image-x of an identifiable yard line"),
    yard: int = typer.Option(None, "--yard", help="hint: that line's yard (5..45, or 50)"),
    side: str = typer.Option("away", "--side", help="hint: home|away|mid"),
    increasing: str = typer.Option("left", "--increasing", help="hint: left|right (image dir yards grow)"),
) -> None:
    import cv2

    from nfl_gsplat.calibration.field_detect import (
        FieldDetectConfig, _white_mask, detect_lines,
    )

    video = Path(play_dir) / f"{cam}.mp4"
    if not video.exists():
        raise SystemExit(f"missing video {video}")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    okk, img = cap.read()
    cap.release()
    if not okk:
        raise SystemExit(f"could not read frame {frame} from {video}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FieldDetectConfig()
    tag = f"{cam}_f{int(frame):05d}"
    frame_png = out_dir / f"diag_{tag}.png"
    mask_png = out_dir / f"diag_{tag}_mask.png"
    annot_png = out_dir / f"diag_{tag}_lines.png"
    cv2.imwrite(str(frame_png), img)
    cv2.imwrite(str(mask_png), _white_mask(img, cfg))

    boxes = _player_boxes(img, yolo_weights) if mask else None
    if mask:
        print(f"YOLO player boxes: {len(boxes)} (masked from field detection)")

    print(f"frame {frame} of {video.name}: shape {img.shape}")
    lines = detect_lines(img, cfg, player_boxes=boxes)
    xs = sorted(round(0.5 * (s.p0[0] + s.p1[0])) for s in lines)
    print(f"yard lines detected: {len(lines)}")
    print(f"line x-positions: {xs}")
    print("  (use one of these x-positions as ref_x in meta.yaml calib_hints)")

    # Draw each detected line + its mean-x label onto the frame so you can read
    # off which line sits under which painted yard number (→ ref_x).
    annot = img.copy()
    for seg in lines:
        x0, y0 = int(seg.p0[0]), int(seg.p0[1])
        x1, y1 = int(seg.p1[0]), int(seg.p1[1])
        mx = round(0.5 * (seg.p0[0] + seg.p1[0]))
        cv2.line(annot, (x0, y0), (x1, y1), (0, 0, 255), 2)
        ty = max(20, min(y0, y1) + 24)
        cv2.putText(annot, str(mx), (int(mx) - 18, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(annot_png), annot)

    # Optional: single-frame hint -> PnP validation.
    if ref_x is not None and yard is not None:
        from nfl_gsplat.calibration.field_detect import detect_field_features
        from nfl_gsplat.calibration.field_identify import (
            identify_correspondences, seed_state_from_hint,
        )
        from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
        from nfl_gsplat.errors import CalibrationError
        from nfl_gsplat.utils.meta import CalibHint

        from nfl_gsplat.calibration.field_identify import fit_hash_rows
        feats = detect_field_features(img, cfg=cfg, player_boxes=boxes)
        _rows = fit_hash_rows(feats.hashes, image_width=img.shape[1])
        print(f"hashes detected: {len(feats.hashes)}  sidelines: {len(feats.sidelines)}  "
              f"hash rows fitted: {len(_rows)}")
        hint = CalibHint(ref_frame=frame, ref_x=float(ref_x), yard=int(yard),
                         side=side, increasing=increasing)
        state = seed_state_from_hint(feats, hint)
        corrs, out_state = identify_correspondences(feats, state)
        print(f"=== hint PnP (ref_x={ref_x} yard={yard} side={side} increasing={increasing}) ===")
        print(f"correspondences: {len(corrs)}  ", [c[0] for c in corrs])

        # Planar-homography sanity check: separates "correspondences inconsistent
        # (mislabeled)" from "PnP focal degenerate on near-affine view". A planar
        # homography is well-posed regardless of perspective amount, so a LOW
        # homography residual + exploding PnP focal ⇒ good points, focal problem;
        # a HIGH homography residual ⇒ the correspondences themselves are wrong.
        if len(corrs) >= 4:
            import numpy as np

            from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
            world = np.array([NFL_LANDMARKS[n][:2] for n, _ in corrs], dtype=np.float64)
            imgpts = np.array([uv for _, uv in corrs], dtype=np.float64)
            Hm, inl = cv2.findHomography(world, imgpts, cv2.RANSAC, 5.0)
            if Hm is not None:
                proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), Hm).reshape(-1, 2)
                resid = np.linalg.norm(proj - imgpts, axis=1)
                print(f"  HOMOGRAPHY: inliers={int(inl.sum())}/{len(corrs)}  "
                      f"median_resid={np.median(resid):.2f}px  max={resid.max():.1f}px")
                print("  (low resid + huge PnP focal => points OK, focal degeneracy; "
                      "high resid => mislabeled correspondences)")

        # Visualize: all detected hashes (green) + matched correspondences (yellow + label).
        corr_png = out_dir / f"diag_{tag}_corr.png"
        vis = annot.copy()
        for hx, hy in feats.hashes:
            cv2.circle(vis, (int(hx), int(hy)), 3, (0, 255, 0), -1)
        for name, (u, v) in corrs:
            cv2.circle(vis, (int(u), int(v)), 7, (0, 255, 255), 2)
            cv2.putText(vis, name.replace("away_", "a").replace("home_", "h").replace("_hash", "H"),
                        (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(corr_png), vis)
        print(f"  saved correspondence viz: {corr_png}")

        # Definitive check: project the WHOLE canonical field (every yard line +
        # both hash rows) through the recovered homography back onto the frame.
        # If the cyan grid lands on the real painted lines/hashes, calibration is
        # correct; drift = wrong. Far stronger than the few labeled points.
        if out_state.homography is not None:
            import numpy as np

            from nfl_gsplat.calibration.field_landmarks import (
                HALF_WIDTH_M, HASH_OFFSET_M, YARD_LINE_SPACING_M,
            )
            Hm = out_state.homography
            grid = annot.copy()

            def to_img(X, Y):
                p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), Hm).reshape(2)
                return (int(round(p[0])), int(round(p[1])))
            # yard lines every 5 yd from −50 to +50, drawn sideline-to-sideline
            for k in range(-10, 11):
                X = k * YARD_LINE_SPACING_M
                cv2.line(grid, to_img(X, +HALF_WIDTH_M), to_img(X, -HALF_WIDTH_M),
                         (255, 200, 0), 1, cv2.LINE_AA)
            # two hash rows spanning the drawn X range
            xspan = 10 * YARD_LINE_SPACING_M
            for Y in (+HASH_OFFSET_M, -HASH_OFFSET_M):
                cv2.line(grid, to_img(-xspan, Y), to_img(+xspan, Y), (255, 120, 0), 1, cv2.LINE_AA)
            grid_png = out_dir / f"diag_{tag}_field.png"
            cv2.imwrite(str(grid_png), grid)
            print(f"  saved field-overlay (cyan = predicted field): {grid_png}")

        if len(corrs) < 6:
            print("  too few correspondences (<6) — need cleaner lines/hashes or a better hint")
        else:
            try:
                res = solve_pnp_from_correspondences(
                    corrs, image_size=(img.shape[1], img.shape[0]), max_reproj_px=1e9)
                print(f"  SOLVED: focal={res.intrinsics.fx:.0f}px  rms={res.rms_px:.2f}px  "
                      f"n={res.num_correspondences}")
                print("  (rms < ~5px = good; large rms => wrong hint side/direction or "
                      "noisy correspondences — flip side/increasing and retry)")
            except CalibrationError as e:
                print(f"  PnP failed: {e}")

    print(f"saved: {frame_png} , {mask_png} , {annot_png}")


if __name__ == "__main__":
    app()
