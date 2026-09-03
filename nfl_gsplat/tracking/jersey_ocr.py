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
    backend: str = "auto"   # auto | paddle | rapidocr | easyocr
    # OCR the torso band rather than the whole player, upscaled: measured on
    # real SEA@AZ crops, feeding the full unscaled box read ~2% of crops, while
    # the upscaled torso band read ~30%. The number sits on the upper back /
    # chest, so everything below the waist is noise that drags detection.
    torso_top_frac: float = 0.15
    torso_bot_frac: float = 0.55
    upscale: float = 2.5
    # Crops are picked by box height; with a facing function (identity.facing,
    # from the pose stage) rows scoring below this are dropped FIRST, so the
    # budget goes to chests and backs, not profiles. 0 = off. Unposed rows
    # (NaN) are kept.
    min_facing: float = 0.0
    # One vote per PLAYER over both views instead of one per (view, track):
    # the ids are shared across views (08b), the endzone sees backs and the
    # sideline profiles, and a number read in either names the player.
    pool_views: bool = False


def _build_paddle(use_gpu: bool):
    from paddleocr import PaddleOCR  # type: ignore
    engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu=use_gpu)

    def read(crop):
        result = engine.ocr(crop, cls=False)
        if not result or not result[0]:
            return []
        return [(line[1][0], float(line[1][1])) for line in result[0]]
    return read


def _build_rapidocr(use_gpu: bool):
    """PP-OCR models on onnxruntime — same recognizer family as paddleocr, but
    with no paddlepaddle dependency (paddle ships no wheels for some Pythons)."""
    from rapidocr import RapidOCR  # type: ignore
    engine = RapidOCR()

    def read(crop):
        out = engine(crop)
        if out is None:
            return []
        # rapidocr >=2 returns an object with .txts/.scores; older returns
        # (list[[box, text, score]], elapse). Support both.
        txts = getattr(out, "txts", None)
        if txts is not None:
            scores = getattr(out, "scores", None) or []
            return [(t, float(s)) for t, s in zip(txts, scores)]
        res = out[0] if isinstance(out, tuple) else out
        return [(r[1], float(r[2])) for r in (res or [])]
    return read


def _build_easyocr(use_gpu: bool):
    import easyocr  # type: ignore
    engine = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def read(crop):
        # allowlist digits: jersey numbers are numeric, and constraining the
        # charset stops letters on the uniform (names, logos) from winning.
        return [(t, float(c)) for _box, t, c in
                engine.readtext(crop, allowlist="0123456789")]
    return read


# Insertion order = preference for backend="auto". easyocr outranks rapidocr:
# measured on 60 real player crops (RTX 4080, play_002), easyocr on CUDA ran
# 37.7 crops/s with 18/60 reads vs rapidocr's 1.3 crops/s with 14/60 — rapidocr
# rides onnxruntime, whose CUDA provider does not engage here (setting
# EngineConfig.onnxruntime.use_cuda changed nothing), so it stays CPU-bound.
# 1.3 crops/s makes a multi-play precompute impractical (~40 min/play).
_BACKEND_BUILDERS = {
    "paddle": _build_paddle,
    "easyocr": _build_easyocr,
    "rapidocr": _build_rapidocr,
}


def _lazy_ocr_engine(use_gpu: bool, backend: str = "auto"):
    """Return ``reader(crop) -> [(text, conf), ...]`` for the chosen backend.

    ``auto`` tries each backend in preference order and uses the first that
    imports, so the same code runs on PACE (paddle) and on machines where
    paddlepaddle has no wheels (rapidocr / easyocr)."""
    if backend != "auto" and backend not in _BACKEND_BUILDERS:
        raise SetupError(
            f"unknown jersey-OCR backend {backend!r} — pick one of "
            f"{sorted(_BACKEND_BUILDERS)} or 'auto'.")

    names = list(_BACKEND_BUILDERS) if backend == "auto" else [backend]
    failures = []
    for name in names:
        try:
            return _BACKEND_BUILDERS[name](use_gpu)
        except ImportError as e:
            failures.append(f"{name} ({e})")
    raise SetupError(
        "no jersey-OCR backend available — tried: " + "; ".join(failures) +
        ". Install one: `pip install rapidocr onnxruntime-gpu` (no paddle "
        "needed), `pip install easyocr`, or use the paddleocr `nfl_smplx` env.")


def _read_frame(video: Path | str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def jersey_crop(frame: np.ndarray, box, cfg: JerseyOCRConfig) -> np.ndarray | None:
    """Upscaled torso band of a player box — the region the number sits on.

    ``box`` is (x1, y1, x2, y2) in frame pixels; returns None when the band is
    degenerate. Cropping to the torso and upscaling is what makes small,
    far-from-camera numbers legible (see JerseyOCRConfig)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    bh = y2 - y1
    ty1 = y1 + int(cfg.torso_top_frac * bh)
    ty2 = y1 + int(cfg.torso_bot_frac * bh)
    band = frame[ty1:ty2, x1:x2]
    if band.size == 0 or band.shape[0] < 2 or band.shape[1] < 2:
        return None
    if cfg.upscale and cfg.upscale != 1.0:
        band = cv2.resize(band, None, fx=cfg.upscale, fy=cfg.upscale,
                          interpolation=cv2.INTER_CUBIC)
    return band


def _ocr_crop(reader, crop: np.ndarray, min_conf: float) -> int | None:
    """Highest-confidence 1-2 digit reading as an int 0..99, else None.
    ``reader`` is a backend adapter returning ``[(text, conf), ...]``."""
    best: tuple[float, str] | None = None
    for text, conf in reader(crop) or []:
        if conf < min_conf:
            continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if 1 <= len(digits) <= 2 and (best is None or conf > best[0]):
            best = (conf, digits)
    return int(best[1]) if best else None


def vote_jersey_numbers(
    df: pd.DataFrame,
    video_paths: dict[str, Path | str],
    cfg: JerseyOCRConfig,
    facing_of=None,
) -> pd.DataFrame:
    """Run OCR + majority vote per ``(cam, track_id)`` -- or per ``track_id``
    over both views with ``cfg.pool_views`` -- and write the winning digit
    into ``jersey_number_ocr``. ``facing_of(cam, frame, track_id)`` (see
    identity.facing) gates the crops when ``cfg.min_facing`` > 0."""
    if df.empty:
        return df.copy()

    reader = _lazy_ocr_engine(cfg.use_gpu, cfg.backend)
    out = df.copy()

    keys = ["track_id"] if cfg.pool_views else ["cam", "track_id"]
    for key, group in df.groupby(keys):
        tid = int(key[-1])
        votes: Counter[int] = Counter()
        for cam, sub in group.groupby("cam"):
            video = video_paths.get(cam)
            if video is None:
                continue
            fo = (None if facing_of is None
                  else (lambda frame, c=cam: facing_of(c, int(frame), tid)))
            for _, row in _pick_rows(sub, cfg, fo).iterrows():
                frame = _read_frame(video, int(row["frame"]))
                if frame is None:
                    continue
                crop = jersey_crop(frame, (row["bbox_x1"], row["bbox_y1"],
                                           row["bbox_x2"], row["bbox_y2"]), cfg)
                if crop is None:
                    continue
                digit = _ocr_crop(reader, crop, cfg.min_ocr_conf)
                if digit is not None:
                    votes[digit] += 1

        if not votes:
            continue
        winner, _ = votes.most_common(1)[0]
        mask = out["track_id"] == tid
        if not cfg.pool_views:
            mask &= out["cam"] == key[0]
        out.loc[mask, "jersey_number_ocr"] = int(winner)
        _LOG.info(f"jersey OCR: ({'both' if cfg.pool_views else key[0]}, track {tid}) "
                  f"-> #{winner}  (votes={dict(votes)})")

    return out


def _pick_rows(g: pd.DataFrame, cfg: JerseyOCRConfig, facing_of) -> pd.DataFrame:
    """The crops one (view, track) gets: tall enough, facing the lens when a
    facing function is given (NaN = unposed, kept), then the top-k by height."""
    g = g.copy()
    g["h"] = g["bbox_y2"] - g["bbox_y1"]
    g = g[g["h"] >= cfg.min_bbox_h_px]
    if facing_of is not None and cfg.min_facing > 0 and len(g):
        score = np.array([facing_of(f) for f in g["frame"]], float)
        g = g[np.isnan(score) | (score >= cfg.min_facing)]
    return g.nlargest(cfg.top_k_frames, "h")


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import PlayDir

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(play_dir: Path = typer.Option(..., "--play-dir"),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        if not bool(cfg.tracking.jersey_ocr_enabled):
            _LOG.info("jersey OCR disabled (tracking.jersey_ocr_enabled=false); skipping")
            return
        pdir = PlayDir.from_dir(play_dir)
        df = pd.read_parquet(pdir.tracks)
        video_paths = {cam: pdir.video(cam) for cam in pdir.cameras}
        out = vote_jersey_numbers(df, video_paths, JerseyOCRConfig(use_gpu=True))
        out.to_parquet(pdir.tracks, index=False)

    app()


if __name__ == "__main__":
    _main()
