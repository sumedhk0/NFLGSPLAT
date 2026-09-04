"""05l: the field texture from the footage (field.footage_texture).

WHAT: warps every n-th frame of each calibrated camera onto the ground
plane, takes the per-texel median over the play (players move out of it),
and composites over the procedural field where too few frames saw the
ground. Writes ``<play-dir>/field_texture.npz`` (texture, count, res_m)
and a PNG preview for the eye.

USAGE:
  python scripts/05l_field_from_footage.py --play-dir data/all22/<game>/play_001
  05k --field-texture <play-dir>/field_texture.npz renders on it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/field_texture.npz")
    ap.add_argument("--preview", type=Path, default=None, help="PNG of the composite (default next to --out)")
    ap.add_argument("--res-m", type=float, default=0.12, help="texels; 05k's --field-res-m must match")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--cams", nargs="+", default=["sideline", "endzone"])
    ap.add_argument("--min-count", type=int, default=None)
    ap.add_argument("--min-px2", type=float, default=None)
    args = ap.parse_args()

    import imageio.v2 as imageio

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.field import footage_texture as ft
    from nfl_gsplat.field.procedural_field import render_field_texture

    P = args.play_dir
    tracks = load_camera_track(P / "cameras.npz")
    videos = {c: P / f"{c}.mp4" for c in args.cams if (P / f"{c}.mp4").exists() and c in tracks}
    if not videos:
        print(f"no camera of {args.cams} has both a video and a track in {P}", file=sys.stderr)
        sys.exit(2)
    kw = {}
    if args.min_count is not None:
        kw["min_count"] = args.min_count
    if args.min_px2 is not None:
        kw["min_px2"] = args.min_px2
    t0 = time.time()
    median, count = ft.footage_texture(videos, tracks, res_m=args.res_m, stride=args.stride, **kw)
    turf, paint = ft.footage_colours(median, count, min_count=kw.get("min_count", ft.MIN_COUNT))
    # footage_colours are BGR (the texture's order); render_field_texture takes RGB.
    procedural = (render_field_texture(args.res_m, turf_rgb=turf[::-1], paint_rgb=paint[::-1])
                  if turf is not None else render_field_texture(args.res_m))
    print(f"footage turf {turf}, paint {paint}; the procedural remainder is drawn in them")
    comp = ft.composite(procedural, median, count, min_count=kw.get("min_count", ft.MIN_COUNT))
    seen = float((count >= kw.get("min_count", ft.MIN_COUNT)).mean())
    out = args.out or P / "field_texture.npz"
    ft.save_texture(out, comp, count, res_m=args.res_m)
    preview = args.preview or out.with_suffix(".png")
    imageio.imwrite(preview, np.ascontiguousarray(comp[:, :, ::-1]))       # BGR -> RGB for the eye
    print(f"field texture {comp.shape[1]}x{comp.shape[0]} at {args.res_m} m; footage covers "
          f"{100 * seen:.0f}% of the extent (median count where seen "
          f"{np.median(count[count > 0]):.0f} frames); {time.time() - t0:.0f} s -> {out}, preview {preview}")


if __name__ == "__main__":
    main()
