#!/usr/bin/env python
"""Fine-tune a pretrained jersey reader and measure it the way identity uses it.

The from-scratch CNN measured 25.1% per track against the play's roster on
held-out plays; generic easyocr measures 75%. Its post-mortem named the missing
piece: a visual prior. This trains jersey_pretrained (ResNet-18, ImageNet
weights) on the same crops and scores BOTH readers on the same held-out plays
with the same per-track, roster-restricted rule, so the comparison is like for
like. Held out by PLAY: whole plays go to one side (splits column).

    per crop   top-1 against the play's roster
    per track  evidence pooled over the track's crops, then top-1

Numbers to beat: CNN 25.1% per track; easyocr 75% per track.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.identity import jersey_cnn, jersey_pretrained
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path,
                    default=Path("data/helmet/jersey_crops_det2.npz"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=0,
                    help="subsample the training crops (0 = all)")
    ap.add_argument("--cnn-weights", type=Path,
                    default=Path("data/weights/jersey_cnn.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/weights/jersey_resnet18.pt"))
    args = ap.parse_args()

    z = np.load(args.crops, allow_pickle=True)
    crops, numbers, splits, tracks = (z["crops"], z["numbers"].astype(int),
                                      z["splits"], z["tracks"])
    tr, te = splits == 0, splits == 1
    print(f"{args.crops.name}: {tr.sum()} train crops, {te.sum()} held-out "
          f"crops over {len(np.unique(tracks[te]))} held-out tracks")

    x_tr, y_tr = crops[tr], numbers[tr]
    if args.max_train and len(x_tr) > args.max_train:
        pick = np.random.default_rng(0).choice(len(x_tr), args.max_train,
                                               replace=False)
        x_tr, y_tr = x_tr[pick], y_tr[pick]

    model = jersey_pretrained.train(x_tr, y_tr, epochs=args.epochs,
                                    batch=args.batch,
                                    val=(crops[te][:4000], numbers[te][:4000]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jersey_cnn.save(model, args.out)
    print(f"saved -> {args.out}")

    per_track, per_crop, n = jersey_pretrained.track_accuracy(
        model, crops[te], numbers[te], tracks[te])
    print("\nheld-out plays, top-1 against the play's roster")
    print(f"   pretrained resnet18   per crop {per_crop:5.1%}   per track "
          f"{per_track:5.1%}   ({n} tracks)")

    if args.cnn_weights.exists():
        try:
            old = jersey_cnn.load(args.cnn_weights)
            pt2, pc2, n2 = jersey_pretrained.track_accuracy(
                old, crops[te], numbers[te], tracks[te])
            print(f"   from-scratch cnn      per crop {pc2:5.1%}   per track "
                  f"{pt2:5.1%}   ({n2} tracks)   [same harness]")
        except Exception as exc:                        # noqa: BLE001
            print(f"   from-scratch cnn      could not be scored: {exc}")
    print("   easyocr (measured earlier)              per track 75%")


if __name__ == "__main__":
    main()
