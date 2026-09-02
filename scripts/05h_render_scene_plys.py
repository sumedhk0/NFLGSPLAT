#!/usr/bin/env python
"""Render per-frame Gaussian scenes (05d --ply-dir) with gsplat, from an orbit.

05d's preview splatter is CPU, sorts every Gaussian per frame, and is enough
to check placement and colour. This is the real thing: gsplat's rasteriser on a
GPU, over the same per-frame scene PLYs (field + bodies), from a virtual camera
that orbits the formation's centroid. Output is a PNG per frame plus a GIF.

Locally the CUDA backend is unavailable (no toolkit to JIT against), so
``--backend cpu`` falls back to the preview splatter over the same loop --
which is how this script is smoke-tested here before it runs on PACE
(scripts/render_play.sbatch).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.compositing.merge_ply import load_gaussian_ply
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at, render_gaussians_cpu
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
_SH_C0 = 0.28209479177387814


def orbit_camera(centre, i, n, *, radius_m: float, height_m: float, turns: float):
    """Camera position on an orbit about ``centre``, frame i of n."""
    ang = 2.0 * np.pi * turns * i / max(n, 1) + np.pi / 2.0
    pos = np.array([centre[0] + radius_m * np.cos(ang),
                    centre[1] + radius_m * np.sin(ang), height_m])
    return look_at(pos, np.array([centre[0], centre[1], 0.0]))


def render_gsplat(batch, K, R, t, width, height, device="cuda:0"):
    import torch
    from gsplat import rasterization

    dev = torch.device(device)
    means = torch.tensor(batch.xyz, dtype=torch.float32, device=dev)
    quats = torch.tensor(batch.rot, dtype=torch.float32, device=dev)          # wxyz
    scales = torch.exp(torch.tensor(batch.scale, dtype=torch.float32, device=dev))
    opac = torch.sigmoid(torch.tensor(batch.opacity, dtype=torch.float32, device=dev))
    colors = torch.clamp(0.5 + _SH_C0 * torch.tensor(batch.sh[:, :, 0], dtype=torch.float32,
                                                     device=dev), 0.0, 1.0)
    viewmat = torch.eye(4, dtype=torch.float32, device=dev)
    viewmat[:3, :3] = torch.tensor(R, dtype=torch.float32)
    viewmat[:3, 3] = torch.tensor(t, dtype=torch.float32)
    Ks = torch.tensor(K, dtype=torch.float32, device=dev)[None]
    img, _alpha, _meta = rasterization(means, quats, scales, opac, colors,
                                       viewmat[None], Ks, width, height,
                                       backgrounds=torch.tensor([[0.06, 0.06, 0.08]],
                                                                device=dev))
    return (255 * img[0].clamp(0, 1).cpu().numpy()).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=("gsplat", "cpu"), default="gsplat")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov-deg", type=float, default=50.0)
    ap.add_argument("--radius-m", type=float, default=30.0)
    ap.add_argument("--height-m", type=float, default=12.0)
    ap.add_argument("--turns", type=float, default=0.5,
                    help="orbit turns over the whole sequence")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import imageio.v2 as imageio

    plys = sorted(args.ply_dir.glob("scene_*.ply"))
    if args.limit:
        plys = plys[:args.limit]
    if not plys:
        raise SystemExit(f"no scene_*.ply in {args.ply_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = intrinsics(args.width, args.height, fov_deg=args.fov_deg)

    # The orbit centre: the bodies' centroid on the first frame. Bodies are
    # everything that is not the field, and the field is the low, flat bulk;
    # the median of points above 0.2 m is the formation.
    first = load_gaussian_ply(plys[0])
    above = first.xyz[first.xyz[:, 2] > 0.2]
    centre = np.median(above[:, :2], axis=0) if len(above) else np.zeros(2)
    _LOG.info("%d frames; orbit centre %s; backend %s", len(plys),
              np.round(centre, 1), args.backend)

    t0 = time.time()
    written = []
    for i, ply in enumerate(plys):
        batch = load_gaussian_ply(ply)
        R, t = orbit_camera(centre, i, len(plys), radius_m=args.radius_m,
                            height_m=args.height_m, turns=args.turns)
        if args.backend == "gsplat":
            img = render_gsplat(batch, K, R, t, args.width, args.height, args.device)
        else:
            img = render_gaussians_cpu(batch, K, R, t, width=args.width,
                                       height=args.height)[..., ::-1]     # BGR -> RGB
        out = args.out_dir / (ply.stem.replace("scene", "frame") + ".png")
        imageio.imwrite(out, img)
        written.append(out)
        if i % 10 == 0:
            _LOG.info("rendered %d/%d (%.2f f/s)", i + 1, len(plys),
                      (i + 1) / max(1e-6, time.time() - t0))
    gif = args.out_dir / "orbit.gif"
    imageio.mimsave(gif, [imageio.imread(p) for p in written], fps=10, loop=0)
    _LOG.info("wrote %d frames + %s (%.0f s)", len(written), gif, time.time() - t0)


if __name__ == "__main__":
    main()
