"""Bring-up diagnostic for automatic field calibration.

Extracts one frame from a play's clip, runs the current line detector + white
mask, and prints detected yard-line x-positions so you can pick a reference
x for the calib_hints block in meta.yaml.

    python scripts/diag_calib.py --play-dir data/2025/week_04/SEA_at_AZ/play_001 --frame 0

Saves <out-dir>/diag_<cam>_f<NNNNN>.png and diag_<cam>_f<NNNNN>_mask.png;
prints detected yard-line x-positions. No display / GPU required.
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    play_dir: Path = typer.Option(..., "--play-dir"),
    frame: int = typer.Option(0, "--frame"),
    cam: str = typer.Option("sideline", "--cam"),
    out_dir: Path = typer.Option(Path("/tmp"), "--out-dir"),
) -> None:
    import cv2

    from nfl_gsplat.calibration.field_detect import (
        FieldDetectConfig, _white_mask, detect_lines,
    )

    video = Path(play_dir) / f"{cam}.mp4"
    if not video.exists():
        raise SystemExit(f"missing video {video}")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    okk, img = cap.read()
    cap.release()
    if not okk:
        raise SystemExit(f"could not read frame {frame} from {video}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FieldDetectConfig()
    tag = f"{cam}_f{int(frame):05d}"
    frame_png = out_dir / f"diag_{tag}.png"
    mask_png = out_dir / f"diag_{tag}_mask.png"
    cv2.imwrite(str(frame_png), img)
    cv2.imwrite(str(mask_png), _white_mask(img, cfg))

    print(f"frame {frame} of {video.name}: shape {img.shape}")
    lines = detect_lines(img, cfg)
    xs = sorted(round(0.5 * (s.p0[0] + s.p1[0])) for s in lines)
    print(f"yard lines detected: {len(lines)}")
    print(f"line x-positions: {xs}")
    print("  (use one of these x-positions as ref_x in meta.yaml calib_hints)")
    print(f"saved: {frame_png} , {mask_png}")


if __name__ == "__main__":
    app()
