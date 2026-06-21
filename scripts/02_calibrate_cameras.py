"""Annotate landmarks + solve PnP for each camera of a play.

Usage::

    python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001
    python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001 --camera sideline --annotate

- Without ``--annotate``: assumes ``{play_dir}/{cam}_landmarks.json`` already
  exists; runs PnP and writes ``{play_dir}/cameras.json``.
- With ``--annotate``: opens the OpenCV annotation GUI first.

Respects the pipeline's fail-loud error philosophy: missing annotations raise
:class:`SetupError` with the exact next command to run.
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.calibration.annotate_gui import annotate as run_annotator
from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
from nfl_gsplat.errors import SetupError
from nfl_gsplat.paths import PlayDir
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
    play_dir: Path = typer.Option(..., help="path to the play folder, e.g. data/2024/week_01/NO_at_ATL/play_001"),
    camera: list[str] = typer.Option(["sideline", "endzone"], "-c", "--camera"),
    annotate: bool = typer.Option(False, help="open GUI annotator before solving"),
    max_reproj_px: float = typer.Option(5.0, help="PnP reprojection tolerance"),
) -> None:
    pd = PlayDir.from_dir(play_dir)
    pd.dir.mkdir(parents=True, exist_ok=True)

    cameras_out: dict = {"reprojection_error_px": {}}

    for cam in camera:
        video = pd.video(cam)
        ann_path = pd.dir / f"{cam}_landmarks.json"

        if not video.exists():
            raise SetupError(
                f"missing video {video}. Place the broadcast clip at that path "
                "(see SETUP.md §2)."
            )

        if annotate or not ann_path.exists():
            _LOG.info(f"launching annotation GUI for {cam}")
            run_annotator(video, ann_path)

        if not ann_path.exists():
            raise SetupError(
                f"landmarks not annotated for {cam}. Rerun with "
                f"  python scripts/02_calibrate_cameras.py --play-dir {play_dir} --camera {cam} --annotate"
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

    write_json(pd.cameras_json, cameras_out)
    _LOG.info(f"wrote {pd.cameras_json}")


if __name__ == "__main__":
    app()
