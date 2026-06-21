"""Per-frame calibration (batch, headless): keyframes + clips -> cameras.npz.

    python scripts/02b_track_calibration.py --play-dir data/2025/week_04/SEA_at_AZ/play_001

Reads each camera's {cam}_keyframes.json (from 02_calibrate_cameras.py), tracks
the field homography across the clip (players masked), and writes cameras.npz.
Fails loud if tracking loses lock (add a keyframe + re-run).
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.calibration.run_tracking import build_camera_npz
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _yolo_masks_provider(cam: str):
    """frame -> player bboxes to exclude from field-feature tracking. The real
    per-frame YOLO/tracks.parquet masking is wired at bring-up; default empty."""
    return lambda frame: []


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    pd = PlayDir.from_dir(play_dir)
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}
    out = build_camera_npz(play_dir=pd.dir, videos=videos, fps=meta.fps,
                           masks_provider=_yolo_masks_provider)
    _LOG.info(f"wrote per-frame calibration → {out}")


if __name__ == "__main__":
    app()
