"""Automatic per-frame field calibration → cameras.npz (headless, no display).

    python scripts/02_autocalibrate.py --play-dir data/2025/week_04/SEA_at_AZ/play_001

Detects + identifies field markings each frame and solves the camera per frame
(no manual annotation, no keyframes). Fails loud if a long run of frames can't be
registered. Replaces the manual 02_calibrate + 02b path (kept as fallback).
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.calibration.run_autocalib import build_autocalib_npz
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    pd = PlayDir.from_dir(play_dir)
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}
    out = build_autocalib_npz(play_dir=pd.dir, videos=videos, fps=meta.fps)
    _LOG.info(f"wrote automatic per-frame calibration → {out}")


if __name__ == "__main__":
    app()
