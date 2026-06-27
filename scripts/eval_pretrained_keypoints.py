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


def _iter_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted([*p.glob("*.png"), *p.glob("*.jpg")])
    return [p]


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free at roboflow.com → Settings → API Keys) or pass --api-key")

    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_names: dict[str, int] = {}
    dumped = False
    for img_path in _iter_images(args.images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = client.infer(str(img_path), model_id=args.model_id)
        r = res[0] if isinstance(res, list) else res
        if not dumped:                         # one-time structure dump to adapt parsing if needed
            print(f"[debug] response type: {type(r).__name__}; repr: {repr(r)[:400]}\n")
            dumped = True
        preds = _get(r, "predictions") or []
        n_kp = 0
        for pred in preds:
            for (name, x, y, conf) in _extract_keypoints(pred):
                if x is None or y is None:
                    continue
                n_kp += 1
                all_names[name] = all_names.get(name, 0) + 1
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), 2)
                cv2.putText(img, str(name), (int(x) + 6, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"kp_{img_path.name}"), img)
        print(f"{img_path.name}: {n_kp} keypoints")

    print("\nKeypoint classes seen (name: count across frames):")
    for name, c in sorted(all_names.items()):
        print(f"  {name}: {c}")
    print(f"\noverlays (yellow = detected keypoints) → {out}")
    print("JUDGE: (1) do the dots land on real lines/hashes/numbers on OUR footage? "
          "(2) do the class names carry yard identity (e.g. a specific yard line / hash) "
          "that we can map to NFL field coordinates? If both yes → we skip labeling/training.")


if __name__ == "__main__":
    main()
