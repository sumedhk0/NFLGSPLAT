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
    args = ap.parse_args()

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

    crops, numbers, splits, heights = [], [], [], []
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
            for r in sub[sub["frame"] == f].itertuples():
                if r.height < MIN_HELMET_PX:
                    continue
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
                kept += 1
        cap.release()
        print(f"[{i:3d}/{len(videos)}] {video}  +{kept} crops "
              f"({'test' if is_test else 'train'})", flush=True)

    crops = np.stack(crops).astype(np.uint8)
    numbers = np.asarray(numbers, np.int16)
    splits = np.asarray(splits, np.int8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, crops=crops, numbers=numbers, splits=splits,
                        heights=np.asarray(heights, np.float32))
    print(f"\n{len(crops)} crops -> {args.out}")
    print(f"   train {(splits == 0).sum()}  test {(splits == 1).sum()}")
    print(f"   distinct numbers {len(np.unique(numbers))}")


if __name__ == "__main__":
    main()
