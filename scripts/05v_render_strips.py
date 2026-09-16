"""Per-id strips and contact sheets cut from the RENDERED frames -- the viewer's footage -- so a body's
motion is judged where the user judges it (the loop's review step; the 05q strips judge the same body
against the real footage).

  05v_render_strips.py --play-dir P --render-dir P/render_hifi_vNN --out DIR \
      --spec PID:START:COUNT:STEP [--spec ...]            per-id crops following the render's camera
      --sheet 300 310 320 ... [--sheet-crop cx,cy,w,h]     a contact sheet of whole frames

The crop follows the id through the render's own follow camera (the same follow_path / look_at /
intrinsics as 05k; pass the eye offset, fov and size the render used). A yellow tick marks the id's
drawn pelvis so the right body is read in a crowd; a frame the render skipped (stride) falls to its
neighbour; a frame where the id is absent is a grey tile that says so. Read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def project_point(K, eye, target, xyz):
    """Pixel of world ``xyz`` for a camera at ``eye`` looking at ``target`` (None when behind it)."""
    from nfl_gsplat.compositing.preview_cpu import look_at

    R, t = look_at(np.asarray(eye, float), np.asarray(target, float))
    c = R @ np.asarray(xyz, float) + t
    if c[2] <= 0:
        return None
    u = np.asarray(K, float) @ (c / c[2])
    return float(u[0]), float(u[1])


def crop_box(u, v, w, h, above: float = 0.55):
    """Integer crop (x0, y0, x1, y1) of ``w x h`` with the pelvis pixel ``above`` of the way down."""
    x0, y0 = int(round(u - w / 2)), int(round(v - h * above))
    return x0, y0, x0 + int(w), y0 + int(h)


def frame_path(render_dir: Path, f: int):
    """The rendered PNG for frame ``f`` or its nearest rendered neighbour (stride); None if neither."""
    for g in (f, f + 1, f - 1):
        p = render_dir / f"frame_{g:05d}.png"
        if p.exists():
            return g, p
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--render-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", action="append", default=[], help="PID:START:COUNT:STEP")
    ap.add_argument("--sheet", type=int, nargs="*", default=None, help="frames of a contact sheet")
    ap.add_argument("--sheet-crop", default=None, help="cx,cy,w,h centre crop of every sheet tile (source px)")
    ap.add_argument("--sheet-cols", type=int, default=4)
    ap.add_argument("--sheet-width", type=int, default=480)
    ap.add_argument("--w", type=int, default=160, help="strip crop width, source px")
    ap.add_argument("--h", type=int, default=220)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--eye-offset", type=float, nargs=3, default=(2.0, -26.0, 10.0), help="as passed to 05k")
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--poses-refit", type=Path, default=None)
    args = ap.parse_args()
    from PIL import Image, ImageDraw

    args.out.mkdir(parents=True, exist_ok=True)
    if args.sheet:
        tiles = []
        for f in args.sheet:
            g, p = frame_path(args.render_dir, f)
            if p is None:
                continue
            im = Image.open(p).convert("RGB")
            if args.sheet_crop:
                cx, cy, w, h = (int(v) for v in args.sheet_crop.split(","))
                im = im.crop((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2))
            im = im.resize((args.sheet_width, int(im.height * args.sheet_width / im.width)), Image.LANCZOS)
            ImageDraw.Draw(im).text((6, 4), f"f{g}", fill=(255, 255, 0))
            tiles.append(im)
        if tiles:
            W, H = tiles[0].size
            rows = (len(tiles) + args.sheet_cols - 1) // args.sheet_cols
            sheet = Image.new("RGB", (args.sheet_cols * W, rows * H), (0, 0, 0))
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % args.sheet_cols) * W, (i // args.sheet_cols) * H))
            out = args.out / f"sheet_f{args.sheet[0]}_{args.sheet[-1]}.png"
            sheet.save(out)
            print(f"wrote {out} ({sheet.size[0]}x{sheet.size[1]}, {len(tiles)} frames)")
    if not args.spec:
        return

    import smplx

    from nfl_gsplat.compositing.preview_cpu import intrinsics
    from nfl_gsplat.render.camera_path import follow_path
    from nfl_gsplat.render.play_timeline import load_play_timeline

    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10,
                         use_pca=False, batch_size=1)
    tl, _tracks, _df, frames_all, _poses = load_play_timeline(args.play_dir, model, poses_refit=args.poses_refit)
    path = follow_path(frames_all, {f: np.mean([s.xy for s in tl.states[f]], axis=0)
                                    for f in frames_all if tl.states.get(f)}, eye_offset=tuple(args.eye_offset))
    K = intrinsics(args.width, args.height, fov_deg=args.fov)
    W, H = int(args.w * args.zoom), int(args.h * args.zoom)
    for spec in args.spec:
        pid, start, count, step = (int(v) for v in spec.split(":"))
        tiles = []
        for i in range(count):
            f, p = frame_path(args.render_dir, start + i * step)
            if p is None:
                continue
            st = next((s for s in tl.states.get(f, ()) if int(s.pid) == pid), None)
            uv = project_point(K, *path[f], (st.xy[0], st.xy[1], 1.0)) if (st is not None and f in path) else None
            if uv is None:
                tile = Image.new("RGB", (W, H), (40, 40, 40))
                ImageDraw.Draw(tile).text((6, 4), f"f{f} id {pid} absent", fill=(255, 80, 80))
                tiles.append(tile)
                continue
            x0, y0, x1, y1 = crop_box(uv[0], uv[1], args.w, args.h)
            crop = Image.open(p).convert("RGB").crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS)
            d = ImageDraw.Draw(crop)
            cu, cv = (uv[0] - x0) * args.zoom, (uv[1] - y0) * args.zoom
            d.line([(cu - 6, cv), (cu + 6, cv)], fill=(255, 255, 0), width=2)
            d.text((6, 4), f"f{f}", fill=(255, 255, 0))
            tiles.append(crop)
        if not tiles:
            print(f"id {pid}: no rendered frames from {start}")
            continue
        cols = min(len(tiles), 8)
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * W, (i // cols) * H))
        out = args.out / f"render_{pid}_f{start}.png"
        sheet.save(out)
        print(f"wrote {out} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
