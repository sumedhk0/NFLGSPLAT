"""Cache pretrained Roboflow field keypoints for a play video (run on Windows).

The ONLY step needing internet + ROBOFLOW_API_KEY. Everything downstream
(PACE or local) reads the JSON. One run per play, ever.

    set ROBOFLOW_API_KEY=...
    python scripts/03_roboflow_precompute.py "C:\\...\\sideline.mp4" ^
        --out "C:\\...\\play_dir\\roboflow_kps.json" ^
        --model-id football-field-key-points-mvmjf/2 [--stride 1] [--kp-conf 0.5]
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2

from nfl_gsplat.calibration.roboflow_precompute import run_precompute
from nfl_gsplat.utils.video import ffprobe_meta


def _video_frames(path, stride):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="<play_dir>/roboflow_kps.json")
    ap.add_argument("--model-id", default="football-field-key-points-mvmjf/2")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--api-url", default="https://detect.roboflow.com")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY or pass --api-key")

    # inference_sdk is a thin wrapper over Roboflow's hosted REST endpoint, and
    # it caps at Python <3.13 -- so on a current interpreter it simply cannot be
    # installed (measured: pip finds no candidate on 3.14). The endpoint itself
    # is stable and documented, so fall back to calling it directly rather than
    # pinning the whole pipeline to an old Python for one HTTP POST.
    try:
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)
        def _infer_one(path):
            return client.infer(path, model_id=args.model_id)
    except ImportError:
        import base64

        import requests

        def _infer_one(path):
            with open(path, "rb") as fh:
                payload = base64.b64encode(fh.read())
            resp = requests.post(
                f"{args.api_url.rstrip('/')}/{args.model_id}",
                params={"api_key": args.api_key},
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            if not resp.ok:
                # Never surface the response URL: the api_key rides in the
                # query string, so raise_for_status() prints the secret into
                # tracebacks and CI logs verbatim.
                detail = {402: "Roboflow account is out of hosted-inference "
                               "credits (HTTP 402). Add credits, or use "
                               "--mode learned with a local landmark model.",
                          401: "Roboflow rejected the API key (HTTP 401).",
                          403: "Roboflow denied access to this model (HTTP 403)."
                          }.get(resp.status_code,
                                f"Roboflow returned HTTP {resp.status_code}.")
                raise SystemExit(f"roboflow precompute: {detail}")
            return resp.json()

    def infer_fn(bgr):
        # hosted API takes a file path; write frame to a temp png
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            cv2.imwrite(tmp, bgr)
            res = _infer_one(tmp)
        finally:
            os.unlink(tmp)
        if isinstance(res, list):
            r = res[0] if res else {}
        else:
            r = res
        out = []
        for pred in (r.get("predictions") or []):
            for kp in (pred.get("keypoints") or []):
                name = kp.get("class_name") or kp.get("class")
                if name is None or kp.get("x") is None:
                    continue
                out.append((str(name), float(kp["x"]), float(kp["y"]),
                            float(kp.get("confidence", 0.0))))
        return out

    num_frames = ffprobe_meta(args.video).num_frames

    def _with_progress(frames_iter, total):
        import time
        t0 = time.time()
        for n, (idx, frame) in enumerate(frames_iter, 1):
            yield idx, frame
            if n % 50 == 0:
                rate = n / (time.time() - t0)
                left = (total // args.stride - n) / rate if rate > 0 else 0
                print(f"  frame {idx}/{total}  ({rate:.1f} f/s, ~{left / 60:.0f} min left)",
                      flush=True)

    n_hit = run_precompute(
        _with_progress(_video_frames(args.video, args.stride), num_frames),
        infer_fn=infer_fn,
        model_id=args.model_id, video_name=Path(args.video).name,
        num_frames=num_frames, out_json=args.out, kp_conf=args.kp_conf)
    print(f"cached keypoints for {n_hit} frames -> {args.out}")


if __name__ == "__main__":
    main()
