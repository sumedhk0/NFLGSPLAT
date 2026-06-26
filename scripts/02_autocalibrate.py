"""Automatic per-frame field calibration → cameras.npz (headless, no display).

    python scripts/02_autocalibrate.py --play-dir data/2025/week_04/SEA_at_AZ/play_001

Detects + identifies field markings each frame and solves the camera per frame
(no manual annotation, no keyframes). Fails loud if a long run of frames can't be
registered. Replaces the manual 02_calibrate + 02b path (kept as fallback).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from nfl_gsplat.calibration.run_autocalib import build_autocalib_npz, build_autocalib_npz_learned
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


class CalibMode(str, Enum):
    hint = "hint"
    learned = "learned"


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         mode: CalibMode = typer.Option(CalibMode.hint, "--mode",
                                        help="'hint' (default) or 'learned' (requires --model-ckpt)."),
         model_ckpt: Optional[Path] = typer.Option(None, "--model-ckpt",
                                                    help="Path to LandmarkNet checkpoint (learned mode only)."),
         yard_min: float = typer.Option(-25.0, "--yard-min",
                                        help="World-X lower bound for landmark schema (learned mode)."),
         yard_max: float = typer.Option(25.0, "--yard-max",
                                        help="World-X upper bound for landmark schema (learned mode)."),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    pd = PlayDir.from_dir(play_dir)
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}

    if mode is CalibMode.learned:
        if model_ckpt is None:
            raise typer.BadParameter("--model-ckpt is required in learned mode.")
        # TODO(bring-up): per-game model_ckpt + yard window in meta.yaml
        out = build_autocalib_npz_learned(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            model_ckpt=model_ckpt, yard_min=yard_min, yard_max=yard_max,
        )
    else:
        # TODO(bring-up): wire tracks.parquet player boxes via masks_provider to de-clutter lines
        out = build_autocalib_npz(
            play_dir=pd.dir, videos=videos, fps=meta.fps, hints=meta.calib_hints,
        )
    _LOG.info(f"wrote automatic per-frame calibration → {out}")


if __name__ == "__main__":
    app()
