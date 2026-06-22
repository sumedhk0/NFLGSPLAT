"""Bring-up diagnostic for automatic field calibration.

Extracts one frame from a play's clip, runs the current line detector + white
mask, and dumps raw PaddleOCR over the whole frame so we can see what the field
markings / painted numbers look like and how well they're detected/read.

    python scripts/diag_calib.py --play-dir data/2025/week_04/SEA_at_AZ/play_001 --frame 0

Saves <out-dir>/diag_frame.png and diag_whitemask.png; prints detected yard-line
x-positions and every OCR'd text string. No display / GPU required.
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
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="run PaddleOCR (disable if it hangs/errs)"),
) -> None:
    import cv2
    import numpy as np

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
    print(f"saved: {frame_png} , {mask_png}")

    if not ocr:
        return
    print("=== raw PaddleOCR (text, conf, center) ===")
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu=False)
        res = engine.ocr(img, cls=False)
        rows = res[0] if res else None
        if not rows:
            print("  (no text detected)")
        for box, (text, conf) in rows or []:
            cx = int(np.mean([p[0] for p in box]))
            cy = int(np.mean([p[1] for p in box]))
            print(f"  '{text}'  conf={conf:.2f}  at ({cx},{cy})")
    except Exception as e:  # noqa: BLE001 - diagnostic; surface any OCR setup issue
        print(f"  OCR failed: {type(e).__name__}: {e}")
        print("  (run on the login node first so PaddleOCR caches its models, then retry)")


if __name__ == "__main__":
    app()
