#!/usr/bin/env python
"""Are jersey numbers more or less readable after the appearance fit?

Acceptance criterion 2 of docs/DESIGN_photometric_appearance.md: the fit
must not make the numbers on rendered bodies less legible than the median
texture did -- numbers are what a viewer looks for. Two render directories
of the SAME frames (05d with and without --fitted-appearance) are read with
easyocr, digits only; per frame the count of digit reads above a confidence
and their summed confidence are compared. No ground truth: a read is a read,
which is fair to both sides since the frames, camera and bodies are identical
and only the appearance differs.

    python scripts/05j_render_legibility.py --a <render_abs> --b <render_fitted>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reads_in(reader, path, *, min_conf: float, upscale: float):
    import cv2

    img = cv2.imread(str(path))
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    out = [(t, float(c)) for _b, t, c in reader.readtext(img, allowlist="0123456789")
           if c >= min_conf and 1 <= len(t) <= 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="render dir A (e.g. median texture)")
    ap.add_argument("--b", type=Path, required=True, help="render dir B (e.g. fitted)")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import easyocr

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    frames = sorted(p.name for p in args.a.glob("frame_*.png") if (args.b / p.name).exists())
    if args.limit:
        frames = frames[:: max(1, len(frames) // args.limit)][:args.limit]
    if not frames:
        raise SystemExit("no frame_*.png shared by the two directories")
    na, nb, ca, cb = [], [], [], []
    for name in frames:
        ra = reads_in(reader, args.a / name, min_conf=args.min_conf, upscale=args.upscale)
        rb = reads_in(reader, args.b / name, min_conf=args.min_conf, upscale=args.upscale)
        na.append(len(ra)); nb.append(len(rb))
        ca.append(sum(c for _t, c in ra)); cb.append(sum(c for _t, c in rb))
    na, nb, ca, cb = map(np.asarray, (na, nb, ca, cb))
    print(f"{len(frames)} frames; digit reads >= {args.min_conf}: A {na.sum()} (mean {na.mean():.2f}/frame), "
          f"B {nb.sum()} (mean {nb.mean():.2f}/frame); summed confidence A {ca.sum():.1f}, B {cb.sum():.1f}")
    print(f"frames where B reads more: {(nb > na).sum()}, fewer: {(nb < na).sum()}, same: {(nb == na).sum()}")
    verdict = "B at least as legible" if nb.sum() >= na.sum() else "B LESS legible -- do not ship"
    print(verdict)


if __name__ == "__main__":
    main()
