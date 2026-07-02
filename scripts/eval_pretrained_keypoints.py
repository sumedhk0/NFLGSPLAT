"""Evaluate a PRETRAINED Roboflow field-keypoint model on our frames.

Decision tool: can we skip hand-labeling + training by using an existing model?
Runs the model on frames, prints the detected keypoint CLASS NAMES + confidences,
and draws them so we can judge (a) does it detect well on our All-22 footage and
(b) do its keypoints carry yard-line identity that maps to NFL field coordinates.

Setup (run locally on Windows, where the frames + internet are):
    pip install inference-sdk opencv-python          # lightweight HTTP client, no torch
    # free API key from roboflow.com  → Settings → API Keys
    set ROBOFLOW_API_KEY=your_key
    # find the exact model id+version on the project's "Deploy" tab, e.g.:
    #   football-field-key-points-mvmjf/2
    python scripts/eval_pretrained_keypoints.py <frame_or_dir> --model-id football-field-key-points-mvmjf/2

Uses Roboflow's hosted inference (runs on their servers); images are sent to the
API. This is a one-off evaluation, not the production path.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def _collect_frames(path, count, out):
    """Frame paths to evaluate. Handles: a video (samples `count` fresh frames), a
    directory of images, or a single image."""
    p = Path(path)
    if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        from nfl_gsplat.landmarks.labeling import sample_frame_indices
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {p}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fdir = out / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in sample_frame_indices(total, count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                fp = fdir / f"f{i:05d}.png"
                cv2.imwrite(str(fp), frame)
                paths.append(fp)
        cap.release()
        print(f"Sampled {len(paths)} fresh frames from {total} in {p.name}.")
        return paths
    if p.is_dir():
        return sorted([*p.glob("*.png"), *p.glob("*.jpg")])
    return [p]


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_nfl_name(model_name, territory):
    """Map a model keypoint class (e.g. '30-top-hash', '20', '50-bottom-sl') to an
    NFL_LANDMARKS name, or None if unmappable (goalline/FG-POST/endzone)."""
    parts = model_name.split("-")
    yard = parts[0]
    if not yard.isdigit():
        return None
    yard = int(yard)
    base = "mid_50" if yard == 50 else f"{territory}_{yard}"
    rest = parts[1:]
    table = {
        (): ("left", "number"),                 # "30"        → top number
        ("bottom",): ("right", "number"),       # "30-bottom" → bottom number
        ("top", "hash"): ("left", "hash"),
        ("bottom", "hash"): ("right", "hash"),
        ("top", "sl"): ("left", "sideline"),
        ("bottom", "sl"): ("right", "sideline"),
    }
    key = tuple(rest)
    if key not in table:
        return None
    lr, typ = table[key]
    return f"{base}_{lr}_{typ}"


def _draw_field_grid(img, H):
    from nfl_gsplat.calibration.field_landmarks import (
        HALF_WIDTH_M, HASH_OFFSET_M, YARD_LINE_SPACING_M,
    )
    import numpy as np
    out = img.copy()

    def to_img(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (int(round(p[0])), int(round(p[1])))

    for k in range(-10, 11):
        X = k * YARD_LINE_SPACING_M
        cv2.line(out, to_img(X, +HALF_WIDTH_M), to_img(X, -HALF_WIDTH_M), (255, 200, 0), 1, cv2.LINE_AA)
    xs = 10 * YARD_LINE_SPACING_M
    for Y in (+HASH_OFFSET_M, -HASH_OFFSET_M):
        cv2.line(out, to_img(-xs, Y), to_img(xs, Y), (255, 120, 0), 1, cv2.LINE_AA)
    return out


def _extract_keypoints(pred):
    """(name, x, y, conf) list from one prediction, tolerant of dict/obj formats."""
    kps = _get(pred, "keypoints") or []
    out = []
    for kp in kps:
        out.append((
            _get(kp, "class_name") or _get(kp, "class") or "?",
            _get(kp, "x"), _get(kp, "y"), _get(kp, "confidence"),
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description="Eval a pretrained Roboflow keypoint model on our frames")
    ap.add_argument("images", help="a frame image or a directory of frames")
    ap.add_argument("--model-id", required=True, help="e.g. football-field-key-points-mvmjf/2")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--api-url", default="https://detect.roboflow.com")
    ap.add_argument("--out-dir", default="kp_eval")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence")
    ap.add_argument("--kp-conf", type=float, default=0.5, help="per-keypoint confidence to keep")
    ap.add_argument("--territory", default="home", choices=["home", "away"],
                    help="which side of the 50 the visible numbers are (global; mirror-resolved later)")
    ap.add_argument("--count", type=int, default=20, help="frames to sample if input is a video")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free at roboflow.com → Settings → API Keys) or pass --api-key")

    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_names: dict[str, int] = {}
    stats = []                                 # (n_confident, n_numbers, n_mapped, inliers)
    dumped = False
    for img_path in _collect_frames(args.images, args.count, out):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        clean = img.copy()                     # keep an unannotated copy for the grid overlay
        res = client.infer(str(img_path), model_id=args.model_id)
        r = res[0] if isinstance(res, list) else res
        if not dumped:                         # one-time structure dump to adapt parsing if needed
            print(f"[debug] response type: {type(r).__name__}; repr: {repr(r)[:400]}\n")
            dumped = True
        H, W = img.shape[:2]
        preds = _get(r, "predictions") or []
        kept = []
        for pred in preds:
            for (name, x, y, conf) in _extract_keypoints(pred):
                if x is None or y is None or conf is None:
                    continue
                # keep only confident, in-image keypoints (drop the skeleton's
                # "not visible" garbage points YOLO-pose emits off-frame at ~0 conf)
                if conf < args.kp_conf or not (0 <= x <= W and 0 <= y <= H):
                    continue
                kept.append((name, float(x), float(y), float(conf)))
                all_names[name] = all_names.get(name, 0) + 1
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), 2)
                cv2.putText(img, str(name), (int(x) + 6, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"kp_{img_path.name}"), img)
        print(f"\n{img_path.name}: {len(kept)} confident keypoints (>= {args.kp_conf}):")
        for (name, x, y, conf) in sorted(kept):
            print(f"    {name:18s} ({x:7.1f}, {y:7.1f})  conf {conf:.2f}")

        # Decisive test: map keypoints → NFL coords, fit a homography, overlay the
        # predicted field. If the cyan grid tracks the painted field, the pretrained
        # model is good enough to skip labeling/training.
        import numpy as np
        from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
        world, image_uv = [], []
        for (name, x, y, _conf) in kept:
            nfl = _to_nfl_name(name, args.territory)
            if nfl is None or nfl not in NFL_LANDMARKS:
                continue
            world.append(NFL_LANDMARKS[nfl][:2])
            image_uv.append([x, y])
        n_numbers = sum(1 for (nm, *_r) in kept if _to_nfl_name(nm, args.territory)
                        and "number" in _to_nfl_name(nm, args.territory))
        inliers = 0
        if len(world) >= 4:
            Hm, mask = cv2.findHomography(np.array(world, np.float64),
                                          np.array(image_uv, np.float64), cv2.RANSAC, 8.0)
            if Hm is not None:
                inl = mask.ravel().astype(bool)
                inliers = int(inl.sum())
                proj = cv2.perspectiveTransform(np.array(world, np.float64).reshape(-1, 1, 2),
                                                Hm).reshape(-1, 2)
                res = np.linalg.norm(proj - np.array(image_uv), axis=1)
                print(f"   HOMOGRAPHY from {len(world)} mapped kps: inliers {inliers}/{len(world)}"
                      f"  median(inliers) {np.median(res[inl]):.1f}px")
                grid = _draw_field_grid(clean, Hm)     # cyan grid on the CLEAN frame
                for (gx, gy) in image_uv:              # + the model's mapped points (green)
                    cv2.circle(grid, (int(gx), int(gy)), 6, (0, 200, 0), 2)
                cv2.imwrite(str(out / f"field_{img_path.name}"), grid)
        else:
            print(f"   only {len(world)} mappable keypoints (<4) — can't fit homography")
        stats.append((len(kept), n_numbers, len(world), inliers))

    print("\nKeypoint classes seen (name: count across frames):")
    for name, c in sorted(all_names.items()):
        print(f"  {name}: {c}")

    if stats:
        n = len(stats)
        avg_kp = sum(s[0] for s in stats) / n
        avg_num = sum(s[1] for s in stats) / n
        with_num = sum(1 for s in stats if s[1] >= 1)
        usable = sum(1 for s in stats if s[3] >= 6)        # >=6 inliers → redundant homography
        print(f"\n=== SUMMARY over {n} frames ===")
        print(f"  avg confident keypoints/frame: {avg_kp:.1f}")
        print(f"  avg NUMBER detections/frame:   {avg_num:.1f}  ({with_num}/{n} frames had >=1 number)")
        print(f"  frames with >=6 homography inliers (a redundant, trustworthy fit): {usable}/{n}")
        print("  → many frames with numbers = HYBRID viable (model numbers anchor identity,")
        print("    our classical hashes add density). Few numbers / <6 inliers = fine-tuning needed.")
    print(f"\noverlays → {out}")


if __name__ == "__main__":
    main()
