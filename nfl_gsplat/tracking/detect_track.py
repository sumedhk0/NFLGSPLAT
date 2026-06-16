"""Per-camera person detection + tracking via Ultralytics YOLOv8 + BoT-SORT.

Output schema (Parquet-friendly DataFrame):
- ``frame``              int
- ``cam``                str  (e.g. "sideline")
- ``track_id``           int  (per-camera; not yet cross-cam)
- ``global_player_id``   int  (filled by :mod:`cross_cam_reid`; -1 here)
- ``bbox_x1, y1, x2, y2`` float (pixels)
- ``conf``               float
- ``foot_u, foot_v``     float (pixels)  — bottom-center of bbox
- ``jersey_number_ocr``  int   (filled by :mod:`jersey_ocr`; -1 here)

``ultralytics`` is imported lazily so callers without the ``nfl_smplx`` env
can still build DataFrames by hand (useful for tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import foot_point_from_bbox
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

TRACK_COLUMNS: list[str] = [
    "frame", "cam", "track_id", "global_player_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "conf", "foot_u", "foot_v", "jersey_number_ocr",
]


@dataclass(frozen=True)
class TrackingConfig:
    yolo_weights: str = "yolov8x.pt"
    tracker: str = "botsort.yaml"
    person_class_id: int = 0
    min_detection_conf: float = 0.35
    device: str = "cuda:0"


def empty_tracks() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_column_dtype(c)) for c in TRACK_COLUMNS})


def _column_dtype(c: str) -> str:
    if c in ("frame", "track_id", "global_player_id", "jersey_number_ocr"):
        return "int64"
    if c == "cam":
        return "object"
    return "float64"


def detect_and_track(
    video: Path | str,
    cam: str,
    cfg: TrackingConfig,
) -> pd.DataFrame:
    """Run YOLOv8 + BoT-SORT on ``video`` and return a TRACK_COLUMNS DataFrame.

    Lazily imports ``ultralytics``; raises :class:`SetupError` if absent.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise SetupError(
            "ultralytics not installed — activate the `nfl_smplx` conda env "
            "(`conda activate nfl_smplx`). See SETUP.md §1."
        ) from e

    model = YOLO(cfg.yolo_weights)
    rows: list[dict] = []
    results: Iterator = model.track(
        source=str(video),
        stream=True,
        tracker=cfg.tracker,
        classes=[cfg.person_class_id],
        conf=cfg.min_detection_conf,
        device=cfg.device,
        persist=True,
        verbose=False,
    )
    for frame_idx, res in enumerate(results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id
        if ids is None:
            continue
        ids = ids.int().cpu().numpy()
        for (b, c, tid) in zip(xyxy, confs, ids):
            u, v = foot_point_from_bbox(np.asarray(b, dtype=np.float64))
            rows.append({
                "frame": int(frame_idx),
                "cam": cam,
                "track_id": int(tid),
                "global_player_id": -1,
                "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
                "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                "conf": float(c),
                "foot_u": float(u), "foot_v": float(v),
                "jersey_number_ocr": -1,
            })
    _LOG.info(f"detect_and_track({cam}): {len(rows)} detections, "
              f"{pd.Series([r['track_id'] for r in rows]).nunique() if rows else 0} unique tracks")
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return _coerce_dtypes(df)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in TRACK_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(_column_dtype(c), copy=False)
    return df


def window_tracks(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    """Keep only detections inside the play's inclusive ``[start, end]`` window.

    Detection runs over the whole per-game video (BoT-SORT needs continuity);
    this slices the result down to the play. Pure — unit-tested.
    """
    if df.empty:
        return df.copy()
    keep = (df["frame"] >= int(start_frame)) & (df["frame"] <= int(end_frame))
    return df.loc[keep].reset_index(drop=True)


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import PlayDir

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(play_dir: Path = typer.Option(..., "--play-dir"),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pdir = PlayDir.from_dir(play_dir)
        tracker = str(cfg.tracking.tracker)
        tcfg = TrackingConfig(
            yolo_weights=str(cfg.tracking.yolo_weights),
            tracker=tracker if tracker.endswith(".yaml") else f"{tracker}.yaml",
            min_detection_conf=float(cfg.tracking.min_detection_conf),
            device=str(cfg.pose.get("device", "cuda:0")),
        )
        dfs = [detect_and_track(pdir.video(cam), cam, tcfg) for cam in pdir.cameras]
        df = pd.concat(dfs, ignore_index=True) if dfs else empty_tracks()
        pdir.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pdir.tracks, index=False)
        _LOG.info(f"detect_track: {len(df)} detections → {pdir.tracks}")

    app()


if __name__ == "__main__":
    _main()
