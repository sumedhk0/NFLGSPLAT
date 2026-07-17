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

from nfl_gsplat.calibration.run_autocalib import (
    build_autocalib_npz, build_autocalib_npz_learned, build_autocalib_npz_pretrained,
)
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


class CalibMode(str, Enum):
    hint = "hint"
    learned = "learned"
    pretrained = "pretrained"
    cross_endzone = "cross-endzone"


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         mode: CalibMode = typer.Option(CalibMode.hint, "--mode",
                                        help="'hint' (default), 'learned' (requires --model-ckpt), "
                                             "or 'pretrained' (requires --roboflow-kps)."),
         model_ckpt: Optional[Path] = typer.Option(None, "--model-ckpt",
                                                    help="Path to LandmarkNet checkpoint (learned mode only)."),
         yard_min: float = typer.Option(-25.0, "--yard-min",
                                        help="World-X lower bound for landmark schema (learned mode)."),
         yard_max: float = typer.Option(25.0, "--yard-max",
                                        help="World-X upper bound for landmark schema (learned mode)."),
         roboflow_kps: Optional[Path] = typer.Option(None, "--roboflow-kps",
             help="Path to roboflow_kps.json (pretrained mode; from scripts/03_roboflow_precompute.py)."),
         territory: str = typer.Option("away", "--territory",
             help="Which side of the 50 the visible yard numbers belong to (pretrained mode)."),
         cameras: str = typer.Option("sideline,endzone", "--cameras",
             help="Comma-separated camera names present in the play dir (e.g. 'sideline')."),
         rotate: str = typer.Option("", "--rotate",
             help="Per-camera view rotation override, e.g. 'endzone=90' or "
                  "'endzone=90,sideline=0'. Default: endzone->90, others->0."),
         player_boxes: Optional[Path] = typer.Option(None, "--player-boxes",
             help="Path to tracks.parquet for player masking (pretrained mode; "
                  "default <play_dir>/tracks.parquet if present)."),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    rotations = {}
    for pair in (p.strip() for p in rotate.split(",") if p.strip()):
        cam_name, _, deg_s = pair.partition("=")
        if deg_s not in ("0", "90", "180", "270"):
            raise typer.BadParameter(
                f"--rotate {pair!r}: rotation must be 0/90/180/270.")
        rotations[cam_name.strip()] = int(deg_s)
    pd = PlayDir.from_dir(play_dir, cameras=tuple(c.strip() for c in cameras.split(",") if c.strip()))
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}

    if mode is CalibMode.cross_endzone:
        from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
        out = build_endzone_from_sideline(
            play_dir=pd.dir, tracks_path=pd.dir / "tracks.parquet",
            cameras_npz=pd.dir / "cameras.npz", endzone_video=pd.video("endzone"),
            fps=meta.fps, rotations=rotations or None)
    elif mode is CalibMode.pretrained:
        # Per-camera keypoint caches: --roboflow-kps names a single-camera
        # cache explicitly; otherwise each camera uses the convention
        # <play_dir>/roboflow_kps_{cam}.json (falling back to the legacy
        # <play_dir>/roboflow_kps.json for a single camera).
        if roboflow_kps is not None:
            if len(videos) != 1:
                raise typer.BadParameter(
                    "--roboflow-kps only applies to a single camera; with "
                    "multiple cameras place roboflow_kps_{cam}.json files "
                    "in the play dir.")
            kps_json = {next(iter(videos)): roboflow_kps}
        else:
            kps_json = {}
            for cam in videos:
                p = pd.dir / f"roboflow_kps_{cam}.json"
                if not p.exists() and len(videos) == 1:
                    p = pd.dir / "roboflow_kps.json"
                kps_json[cam] = p
        from nfl_gsplat.calibration.player_masks import boxes_provider_from_tracks
        boxes_path = player_boxes if player_boxes is not None else (pd.dir / "tracks.parquet")
        if boxes_path.exists():
            masks_provider = boxes_provider_from_tracks(boxes_path)
        else:
            _LOG.warning("no player tracks at %s — running calibration UNMASKED "
                         "(endzone likely fails; run scripts/03b_detect_players.py)",
                         boxes_path)
            masks_provider = None
        out = build_autocalib_npz_pretrained(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            kps_json=kps_json, territory=territory,
            rotations=rotations or None, masks_provider=masks_provider,
        )
    elif mode is CalibMode.learned:
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
