"""Jersey OCR backend adapters + digit parsing. No OCR engine required:
readers are plain callables returning [(text, conf), ...]."""
import numpy as np
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.tracking import jersey_ocr as jo


def _crop():
    return np.zeros((40, 20, 3), np.uint8)


def test_ocr_crop_picks_highest_confidence_digits():
    reader = lambda _c: [("58", 0.7), ("12", 0.9), ("3", 0.6)]
    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) == 12


def test_ocr_crop_rejects_below_min_conf():
    reader = lambda _c: [("58", 0.3)]
    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) is None


def test_ocr_crop_rejects_non_two_digit_text():
    # 3+ digits (e.g. a scoreboard number) and pure letters are not jerseys
    reader = lambda _c: [("123", 0.99), ("ABC", 0.99)]
    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) is None


def test_ocr_crop_strips_surrounding_noise():
    # OCR often returns the number glued to stray glyphs
    reader = lambda _c: [("#58", 0.9)]
    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) == 58


def test_ocr_crop_empty_result_is_none():
    assert jo._ocr_crop(lambda _c: [], _crop(), min_conf=0.5) is None


def test_jersey_crop_takes_upscaled_torso_band():
    frame = np.zeros((400, 300, 3), np.uint8)
    cfg = jo.JerseyOCRConfig(torso_top_frac=0.2, torso_bot_frac=0.6, upscale=2.0)
    out = jo.jersey_crop(frame, (100, 100, 180, 300), cfg)   # box 80x200
    # band = rows 100+0.2*200=140 .. 100+0.6*200=220 -> 80 tall, 80 wide, x2
    assert out.shape[:2] == (160, 160)


def test_jersey_crop_clips_to_frame_and_rejects_degenerate():
    frame = np.zeros((100, 100, 3), np.uint8)
    cfg = jo.JerseyOCRConfig()
    assert jo.jersey_crop(frame, (10, 10, 12, 12), cfg) is None    # too small
    out = jo.jersey_crop(frame, (-20, -20, 90, 90), cfg)           # clipped, still valid
    assert out is not None and out.shape[0] > 0


def test_unknown_backend_fails_loud():
    with pytest.raises(SetupError, match="backend"):
        jo._lazy_ocr_engine(use_gpu=False, backend="not-a-backend")


def test_auto_backend_reports_all_candidates_when_none_installed(monkeypatch):
    # Every backend import fails -> one actionable error naming what to install.
    monkeypatch.setattr(jo, "_BACKEND_BUILDERS", {
        "paddle": lambda _g: (_ for _ in ()).throw(ImportError("no paddle")),
        "rapidocr": lambda _g: (_ for _ in ()).throw(ImportError("no rapidocr")),
    })
    with pytest.raises(SetupError) as exc:
        jo._lazy_ocr_engine(use_gpu=False, backend="auto")
    msg = str(exc.value)
    assert "rapidocr" in msg and "paddle" in msg


def test_auto_backend_uses_first_importable(monkeypatch):
    monkeypatch.setattr(jo, "_BACKEND_BUILDERS", {
        "paddle": lambda _g: (_ for _ in ()).throw(ImportError("no paddle")),
        "rapidocr": lambda _g: (lambda _c: [("42", 0.95)]),
    })
    reader = jo._lazy_ocr_engine(use_gpu=False, backend="auto")
    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) == 42
