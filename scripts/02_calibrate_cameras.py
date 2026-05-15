"""Annotate landmarks + solve PnP for each camera of a game.

Usage::

    python scripts/02_calibrate_cameras.py --game game_001
    python scripts/02_calibrate_cameras.py --game game_001 --camera sideline --annotate

- Without ``--annotate``: assumes ``data/annotations/{game}/{cam}_landmarks.json``
  already exists; runs PnP and writes ``outputs/{game}/calib/cameras.json``.
- With ``--annotate``: opens the OpenCV annotation GUI first.

Respects the pipeline's fail-loud error philosophy: missing annotations raise
:class:`SetupError` with the exact next command to run.
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.calibration.annotate_gui import annotate_frame
from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.io import write_json
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.video import iter_frames

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _first_frame(video_path: Path):
    for _, img in iter_frames(video_path, start_frame=0, stride=1):
        return img
    raise RuntimeError(f"no frames readable from {video_path}")


@app.command()
def main(
    game: str = typer.Option(..., help="game_id, e.g. game_001"),
    camera: list[str] = typer.Option(["sideline", "endzone"], "-c", "--camera"),
    annotate: bool = typer.Option(False, help="open GUI annotator before solving"),
    data_root: Path = typer.Option(Path("data"), help="root of raw/annotations dirs"),
    out_root: Path = typer.Option(Path("outputs"), help="root of outputs dir"),
    max_reproj_px: float = typer.Option(5.0, help="PnP reprojection tolerance"),
) -> None:
    raw_dir = data_root / "raw" / game
    ann_dir = data_root / "annotations" / game
    out_dir = out_root / game / "calib"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_out: dict = {"reprojection_error_px": {}}

    for cam in camera:
        video = raw_dir / f"{cam}.mp4"
        ann_path = ann_dir / f"{cam}_landmarks.json"

        if not video.exists():
            raise SetupError(
                f"missing video {video}. Place the broadcast clip at that path "
                "(see SETUP.md §2)."
            )

        if annotate or not ann_path.exists():
            _LOG.info(f"launching annotation GUI for {cam}")
            frame = _first_frame(video)
            annotate_frame(frame, ann_path)

        if not ann_path.exists():
            raise SetupError(
                f"landmarks not annotated for {cam}. Rerun with "
                f"  python scripts/02_calibrate_cameras.py --game {game} --camera {cam} --annotate"
            )

        # Use the first frame's resolution as the calibration reference.
        frame = _first_frame(video)
        H, W = frame.shape[:2]
        result = solve_pnp_from_annotations(
            ann_path, image_size=(W, H), max_reproj_px=max_reproj_px,
        )
        cameras_out[cam] = {
            "K": result.K.tolist(), "R": result.R.tolist(), "t": result.t.tolist(),
            "width": W, "height": H,
        }
        cameras_out["reprojection_error_px"][cam] = float(result.rms_px)
        _LOG.info(f"{cam}: rms {result.rms_px:.2f} px from {result.num_landmarks} landmarks")

    out_path = out_dir / "cameras.json"
    write_json(out_path, cameras_out)
    _LOG.info(f"wrote {out_path}")


if __name__ == "__main__":
    app()
