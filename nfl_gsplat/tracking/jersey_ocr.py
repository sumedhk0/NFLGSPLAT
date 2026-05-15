"""Optional jersey-number OCR using PaddleOCR + majority vote per track.

Strategy: for each ``(cam, track_id)``, pick the top-K frames by bbox area
as OCR candidates (bigger = more readable), crop, binarize, run PaddleOCR,
filter results to 1–2 digit strings, majority-vote. Write the result into
``df['jersey_number_ocr']`` (−1 when no confident result).

The pipeline can skip this stage entirely via ``cfg.tracking.jersey_ocr_enabled=false``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class JerseyOCRConfig:
    top_k_frames: int = 8
    min_bbox_h_px: int = 80
    min_ocr_conf: float = 0.5
    use_gpu: bool = True


def _lazy_ocr_engine(use_gpu: bool):
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as e:
        raise SetupError(
            "paddleocr not installed — activate the `nfl_smplx` conda env. See SETUP.md §1."
        ) from e
    return PaddleOCR(
        use_angle_cls=False, lang="en", show_log=False,
        use_gpu=use_gpu,
    )


def _read_frame(video: Path | str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def _ocr_crop(engine, crop: np.ndarray, min_conf: float) -> int | None:
    """Return digit integer 0..99 or None."""
    result = engine.ocr(crop, cls=False)
    if not result or not result[0]:
        return None
    best: tuple[float, str] | None = None
    for line in result[0]:
        text = line[1][0]
        conf = float(line[1][1])
        if conf < min_conf:
            continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if 1 <= len(digits) <= 2:
            if best is None or conf > best[0]:
                best = (conf, digits)
    return int(best[1]) if best else None


def vote_jersey_numbers(
    df: pd.DataFrame,
    video_paths: dict[str, Path | str],
    cfg: JerseyOCRConfig,
) -> pd.DataFrame:
    """Run OCR + majority vote for each ``(cam, track_id)`` group and write
    the winning digit into ``jersey_number_ocr``."""
    if df.empty:
        return df.copy()

    engine = _lazy_ocr_engine(cfg.use_gpu)
    out = df.copy()

    for (cam, tid), group in df.groupby(["cam", "track_id"]):
        video = video_paths.get(cam)
        if video is None:
            continue
        g = group.copy()
        g["h"] = g["bbox_y2"] - g["bbox_y1"]
        g = g[g["h"] >= cfg.min_bbox_h_px]
        g = g.nlargest(cfg.top_k_frames, "h")

        votes: Counter[int] = Counter()
        for _, row in g.iterrows():
            frame = _read_frame(video, int(row["frame"]))
            if frame is None:
                continue
            x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = frame[y1:y2, x1:x2]
            digit = _ocr_crop(engine, crop, cfg.min_ocr_conf)
            if digit is not None:
                votes[digit] += 1

        if not votes:
            continue
        winner, _ = votes.most_common(1)[0]
        mask = (out["cam"] == cam) & (out["track_id"] == tid)
        out.loc[mask, "jersey_number_ocr"] = int(winner)
        _LOG.info(f"jersey OCR: ({cam}, track {tid}) → #{winner}  "
                  f"(votes={dict(votes)})")

    return out
