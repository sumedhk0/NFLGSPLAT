"""Multi-frame landmark labeling CLI.

Usage::

    python scripts/label_landmarks.py <clip> <out_dir> [--count N] [--names n1 n2 ...]

Opens <clip>, samples --count frames evenly, extracts each to <out>/frames/fNNNNN.png,
opens the click-loop GUI for each frame, and writes <out>/labels.json.

Controls per frame (same as annotate_gui):
    click: place   n/p: next/prev   u: undo   d: delete   s: save & next   q: quit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from nfl_gsplat.calibration.annotate_gui import DEFAULT_PRESET, annotate_frame
from nfl_gsplat.calibration.field_landmarks import list_landmark_names
from nfl_gsplat.landmarks.labeling import build_label_record, sample_frame_indices


def _extract_frame(cap: cv2.VideoCapture, frame_idx: int) -> tuple[int, int, object]:
    """Seek to frame_idx and return (width, height, bgr_ndarray)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    h, w = bgr.shape[:2]
    return w, h, bgr


def main():
    ap = argparse.ArgumentParser(description="Multi-frame NFL landmark labeling tool")
    ap.add_argument("clip", help="Input video file")
    ap.add_argument("out_dir", help="Output directory for frames + labels.json")
    ap.add_argument("--count", type=int, default=20,
                    help="Number of frames to sample (default: 20)")
    ap.add_argument("--names", nargs="+", default=None,
                    help="Explicit landmark names to annotate (overrides --yard-min/max)")
    ap.add_argument("--yard-min", type=float, default=None,
                    help="World-X min (m); auto-selects all landmarks in the window "
                         "(hashes + number anchors + sidelines) — use the SAME window at train time")
    ap.add_argument("--yard-max", type=float, default=None,
                    help="World-X max (m); pair with --yard-min")
    args = ap.parse_args()

    clip = Path(args.clip)
    out = Path(args.out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_names = list_landmark_names()
    if args.names:
        name_list = list(args.names)
    elif args.yard_min is not None and args.yard_max is not None:
        from nfl_gsplat.landmarks.schema import LandmarkSchema
        name_list = LandmarkSchema(yard_min=args.yard_min, yard_max=args.yard_max).class_names()
    else:
        name_list = list(DEFAULT_PRESET)
    for n in name_list:
        if n not in all_names:
            raise ValueError(f"unknown landmark name: {n!r}")
    print(f"Annotating {len(name_list)} landmark classes.")

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {clip}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, args.count)
    print(f"Sampling {len(frame_indices)} frames from {total_frames} total.")

    frame_records = []
    img_w = img_h = None

    for fi in frame_indices:
        w, h, bgr = _extract_frame(cap, fi)
        if img_w is None:
            img_w, img_h = w, h

        fname = f"f{fi:05d}.png"
        fpath = frames_dir / fname
        cv2.imwrite(str(fpath), bgr)

        print(f"\nFrame {fi} → {fname}  (press 's' to save and continue, 'q' to quit)")

        # TODO(bring-up): classical-homography prefill
        # Optionally compute a rough homography from already-placed points and
        # use it to project world coords to image coords as a prefill hint.
        prefill = None

        try:
            points = annotate_frame(
                bgr, name_list, prefill=prefill,
                window_title=f"Frame {fi} / {total_frames - 1} — s: save & next, q: quit"
            )
        except RuntimeError:
            print("Annotation aborted by user — stopping.")
            break

        if points:
            frame_records.append(build_label_record(fname, points))
            print(f"  saved {len(points)} point(s) for frame {fi}")
        else:
            print(f"  no points placed for frame {fi}, skipping record")

    cap.release()

    if img_w is None:
        raise RuntimeError("No frames were processed.")

    label_path = out / "labels.json"
    label_path.write_text(
        json.dumps({"image_size": [img_w, img_h], "frames": frame_records}, indent=2)
    )
    print(f"\nWrote {len(frame_records)} frame record(s) → {label_path}")


if __name__ == "__main__":
    main()
