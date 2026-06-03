"""Ball stage CLI: detect the football per camera → 3D Kalman → ``ball.npz``.

Detection (football-tuned YOLO) is the env-gated seam; assembling per-frame
detections and running the gravity-prior Kalman filter is pure and tested.
The renderer orients the canonical football asset along the filtered velocity.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def detections_to_frames(
    detections_by_cam: Mapping[str, pd.DataFrame],
    window_start: int,
    window_end: int,
) -> list[dict[str, np.ndarray]]:
    """Assemble ``[{cam: uv[2]}, ...]`` over the window for :func:`run_kalman`.

    Each per-camera ``DataFrame`` has the ball-detection schema (``frame``, ``u``,
    ``v``, ...). A frame slot gets a camera entry only where that camera detected
    the ball; absent cameras mean "no detection" (the filter coasts on gravity).
    Pure — no video or model.
    """
    T = window_end - window_start + 1
    frames: list[dict[str, np.ndarray]] = [dict() for _ in range(T)]
    for cam, df in detections_by_cam.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            slot = int(r["frame"]) - window_start
            if 0 <= slot < T:
                frames[slot][cam] = np.array([float(r["u"]), float(r["v"])], dtype=np.float64)
    return frames


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    from pathlib import Path

    import typer

    from nfl_gsplat.ball.ball_io import build_and_write_ball_track
    from nfl_gsplat.ball.detect_ball import BallDetectConfig, detect_ball
    from nfl_gsplat.ball.kalman_3d import BallKalmanConfig
    from nfl_gsplat.calibration.cameras_io import load_cameras
    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import play_paths
    from nfl_gsplat.utils.plays import load_plays

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(game: str = typer.Option(...), play: str = typer.Option(...),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pp = play_paths(cfg, game, play)
        manifest = load_plays(pp.game.plays_yaml)
        window = manifest.window(play)
        cameras = load_cameras(pp.game.calib_json)

        weights = str(Path(str(cfg.paths.weights)) / str(cfg.ball.yolo_weights))
        det_cfg = BallDetectConfig(weights=weights,
                                   min_conf=float(cfg.ball.min_detection_conf),
                                   device=str(cfg.pose.get("device", "cuda:0")))
        detections: dict[str, pd.DataFrame] = {
            cam: detect_ball(pp.game.raw_video(cam), cam, det_cfg) for cam in cameras
        }
        frames = detections_to_frames(detections, window.start_frame, window.end_frame)
        kal_cfg = BallKalmanConfig(fps=manifest.fps)
        build_and_write_ball_track(pp.ball, frames, cameras, kal_cfg)
        n_seen: Sequence[int] = [len(f) for f in frames]
        _LOG.info(f"ball: {sum(1 for n in n_seen if n)} of {len(frames)} frames had a "
                  f"detection → {pp.ball}")

    app()


if __name__ == "__main__":
    _main()
