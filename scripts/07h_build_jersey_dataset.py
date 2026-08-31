#!/usr/bin/env python
"""Cut a supervised jersey-number dataset out of the labelled helmet boxes.

Generic OCR is the weakest link in identity: measured, 87% of tracks got some
read but only 75% had the right number on top. It was never trained on this
problem -- small, motion-blurred, curved digits on a moving shoulder, seen from
sixty metres. A model trained on exactly that should do better, and the data to
train it exists: train_labels.csv carries 952k helmet boxes each tagged with the
player, and the player's jersey number is in the label ("H90" -> 90).

WHAT IS CUT. The label marks a HELMET; the number is on the torso below it, so a
body box is synthesised from the helmet's own size and the number band taken
from that. Same construction the OCR path uses, so the two are comparable.

HELD OUT BY PLAY, not by crop. Two crops of the same player one frame apart are
nearly the same picture, so splitting at random would put a near-duplicate of
every test crop in training and report a number that means nothing. Whole plays
go to one side or the other.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.data.helmet_dataset import load_labels

# The band of the synthesised body box that holds the number, and the size the
# crops are stored at. 64x64 is well above the ~20 px a distant jersey occupies,
# so nothing is thrown away by the resize.
BODY_HEIGHT_IN_HELMETS: float = 6.5
BODY_HALF_WIDTH_IN_HELMETS: float = 1.6
TORSO_TOP: float = 0.18
TORSO_BOTTOM: float = 0.55
CROP_PX: int = 64

# Below this the crop is a few pixels of mush and teaches the model noise.
MIN_HELMET_PX: int = 8


def person_boxes(model, image, conf: float = 0.25):
    """Person boxes in one frame, as ``[N, 4]`` xyxy."""
    res = model.predict(image, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4), np.float32)
    return res.boxes.xyxy.cpu().numpy().astype(np.float32)


def match_person(helmet, boxes):
    """The person box a helmet sits in, or None.

    A helmet is at the TOP of its player, so the right box is one that contains
    the helmet centre and starts near it -- not merely the nearest, which on a
    crowded field is often the player standing behind.
    """
    if len(boxes) == 0:
        return None
    left, top, width, height = helmet
    cx, cy = left + width / 2.0, top + height / 2.0
    contains = ((boxes[:, 0] <= cx) & (boxes[:, 2] >= cx)
                & (boxes[:, 1] <= cy + height) & (boxes[:, 3] >= cy))
    idx = np.flatnonzero(contains)
    if len(idx) == 0:
        return None
    # Among containers, the one whose top is closest to the helmet's top.
    best = idx[np.argmin(np.abs(boxes[idx, 1] - top))]
    return boxes[best]


def torso_crop_from_person(image, box, top=TORSO_TOP, bottom=TORSO_BOTTOM):
    """The number band of a real person box."""
    import cv2

    x1, y1, x2, y2 = [float(v) for v in box]
    h = y2 - y1
    ya = int(round(y1 + top * h))
    yb = int(round(y1 + bottom * h))
    xa, xb = int(round(x1)), int(round(x2))
    xa, ya = max(0, xa), max(0, ya)
    xb = min(image.shape[1], xb)
    yb = min(image.shape[0], yb)
    if xb - xa < 4 or yb - ya < 4:
        return None
    crop = image[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)


def torso_crop(image, left, top, width, height):
    """The number band implied by one helmet box, or None if unusable."""
    import cv2

    h_img, w_img = image.shape[:2]
    cx = left + width / 2.0
    half = BODY_HALF_WIDTH_IN_HELMETS * height
    x1 = int(max(0, round(cx - half)))
    x2 = int(min(w_img, round(cx + half)))
    y1 = int(max(0, round(top + TORSO_TOP * BODY_HEIGHT_IN_HELMETS * height)))
    y2 = int(min(h_img, round(top + TORSO_BOTTOM * BODY_HEIGHT_IN_HELMETS * height)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--out", type=Path, default=Path("data/helmet/jersey_crops.npz"))
    ap.add_argument("--frames-per-video", type=int, default=40)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--detector", action="store_true",
                    help="crop from real YOLO person boxes instead of a body "
                         "box synthesised from the helmet")
    args = ap.parse_args()

    detector = None
    if args.detector:
        from ultralytics import YOLO

        detector = YOLO("yolov8n.pt")

    labels = load_labels(args.root / "train_labels.csv")
    labels = labels[labels["label"].str.slice(1).str.isdigit()]
    videos = sorted(labels["video"].unique())
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    # Split on the PLAY so a near-duplicate frame cannot straddle the split.
    plays = sorted({tuple(v.split("_")[:2]) for v in videos})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(plays))
    n_test = max(1, int(round(args.holdout_frac * len(plays))))
    test_plays = {plays[i] for i in order[:n_test]}
    print(f"{len(plays)} plays -> {len(test_plays)} held out\n")

    crops, numbers, splits, heights, tracks = [], [], [], [], []
    for i, video in enumerate(videos, 1):
        sub = labels[labels["video"] == video]
        if sub.empty:
            continue
        key = tuple(video.split("_")[:2])
        is_test = key in test_plays
        frames = np.array(sorted(sub["frame"].unique()))
        pick = frames[np.linspace(0, len(frames) - 1,
                                  min(args.frames_per_video, len(frames))
                                  ).astype(int)]
        cap = cv2.VideoCapture(str(args.root / "video" / video))
        kept = 0
        for f in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - 1)
            ok, img = cap.read()
            if not ok:
                continue
            boxes = person_boxes(detector, img) if detector is not None else None
            for r in sub[sub["frame"] == f].itertuples():
                if r.height < MIN_HELMET_PX:
                    continue
                if boxes is not None:
                    person = match_person((r.left, r.top, r.width, r.height), boxes)
                    crop = (None if person is None
                            else torso_crop_from_person(img, person))
                else:
                    crop = torso_crop(img, r.left, r.top, r.width, r.height)
                if crop is None:
                    continue
                crops.append(crop)
                numbers.append(int(r.label[1:]))
                splits.append(1 if is_test else 0)
                # Helmet height is the only per-crop scale cue, and it decides
                # whether a number is legible at all -- recorded so accuracy can
                # be read against it instead of averaged over illegible crops.
                heights.append(float(r.height))
                # (video, player) IS the track. Identity consumes evidence per
                # track, never per crop, so without this the model cannot be
                # compared against the OCR path on the terms that matter.
                tracks.append(f"{video}|{r.label}")
                kept += 1
        cap.release()
        print(f"[{i:3d}/{len(videos)}] {video}  +{kept} crops "
              f"({'test' if is_test else 'train'})", flush=True)

    crops = np.stack(crops).astype(np.uint8)
    numbers = np.asarray(numbers, np.int16)
    splits = np.asarray(splits, np.int8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, crops=crops, numbers=numbers, splits=splits,
                        heights=np.asarray(heights, np.float32),
                        tracks=np.asarray(tracks))
    print(f"\n{len(crops)} crops -> {args.out}")
    print(f"   train {(splits == 0).sum()}  test {(splits == 1).sum()}")
    print(f"   distinct numbers {len(np.unique(numbers))}")


if __name__ == "__main__":
    main()
