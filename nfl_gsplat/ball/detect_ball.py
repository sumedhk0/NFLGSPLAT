"""YOLOv8 football detection with a lazy ``ultralytics`` import.

Uses a fine-tuned football weights file (fetched by
``scripts/01_download_models.sh``) — generic COCO YOLO is not reliable on
the football-specific class. Output schema is the small DataFrame::

    frame  cam  conf  u  v  bbox_x1  bbox_y1  bbox_x2  bbox_y2

One row per frame where the ball is detected in ``cam``. This is consumed by
:mod:`kalman_3d` which fuses per-camera 2D detections into a 3D trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

BALL_COLUMNS: list[str] = [
    "frame", "cam", "conf", "u", "v",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


@dataclass(frozen=True)
class BallDetectConfig:
    weights: str = "data/body_models/ball_yolov8.pt"
    min_conf: float = 0.25
    device: str = "cuda:0"
    football_class_id: int = 0


def empty_ball_detections() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(dtype="int64" if c == "frame"
                     else ("object" if c == "cam" else "float64"))
        for c in BALL_COLUMNS
    })


def detect_ball(video: Path | str, cam: str, cfg: BallDetectConfig) -> pd.DataFrame:
    """Detect the football in each frame of ``video``; return a DataFrame."""
    weights = Path(cfg.weights)
    if not weights.exists():
        raise SetupError(
            f"football YOLO weights missing at {weights}. "
            "Run scripts/01_download_models.sh — see SETUP.md §4."
        )
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise SetupError(
            "ultralytics not installed — activate the `nfl_smplx` conda env. "
            "See SETUP.md §1."
        ) from e

    model = YOLO(str(weights))
    rows: list[dict] = []
    for frame_idx, res in enumerate(model.predict(
            source=str(video), stream=True, conf=cfg.min_conf,
            classes=[cfg.football_class_id], device=cfg.device, verbose=False)):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        # Keep only the highest-confidence detection per frame; multiple footballs
        # on a broadcast feed are false positives 99% of the time.
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        b = boxes[best]
        rows.append({
            "frame": int(frame_idx),
            "cam": cam,
            "conf": float(confs[best]),
            "u": float(0.5 * (b[0] + b[2])),
            "v": float(0.5 * (b[1] + b[3])),
            "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
            "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
        })
    _LOG.info(f"detect_ball({cam}): {len(rows)} frames with ball detected")
    return pd.DataFrame(rows, columns=BALL_COLUMNS) if rows else empty_ball_detections()
