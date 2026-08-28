"""Jersey OCR backend adapters + digit parsing. No OCR engine required:
readers are plain callables returning [(text, conf), ...]."""
import collections

import numpy as np
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity.jersey_ocr import read_jerseys
from nfl_gsplat.tracking import jersey_ocr as jo


def _crop():
    return np.zeros((40, 20, 3), np.uint8)


def test_ocr_crop_picks_highest_confidence_digits():
    def reader(_c):
        return [("58", 0.7), ("12", 0.9), ("3", 0.6)]

    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) == 12


def test_ocr_crop_rejects_below_min_conf():
    def reader(_c):
        return [("58", 0.3)]

    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) is None


def test_ocr_crop_rejects_non_two_digit_text():
    # 3+ digits (e.g. a scoreboard number) and pure letters are not jerseys
    def reader(_c):
        return [("123", 0.99), ("ABC", 0.99)]

    assert jo._ocr_crop(reader, _crop(), min_conf=0.5) is None


def test_ocr_crop_strips_surrounding_noise():
    # OCR often returns the number glued to stray glyphs
    def reader(_c):
        return [("#58", 0.9)]

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


# --- the yield knobs must actually be knobs ------------------------------------
# The right values differ per feed by an order of magnitude: the endzone is
# zoomed ~10x tighter than the sideline and read jerseys 5-10x better on the same
# play. A floor tuned for one feed silently discards the other's evidence, so
# these are parameters and these tests hold them to it.


class _StubReader:
    """Returns one fixed read, at a fixed confidence, for every crop."""

    def __init__(self, text="18", conf=0.22):
        self.text, self.conf = text, conf
        self.crops = []

    def readtext(self, crop, allowlist=None):
        self.crops.append(crop.shape)
        return [((0, 0), self.text, self.conf)]


class _Row:
    def __init__(self, track_id, box):
        self.track_id = track_id
        (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2) = box


def _frames(h=200, w=120):
    yield 0, np.full((h, w, 3), 120, np.uint8)


def test_a_read_below_the_confidence_floor_is_dropped():
    reader = _StubReader(conf=0.22)
    wanted = {0: [_Row(1, (10, 10, 90, 190))]}
    votes, _ = read_jerseys(_frames(), wanted, reader=reader, min_conf=0.30)
    assert votes == {}


def test_lowering_the_floor_keeps_the_same_read():
    reader = _StubReader(conf=0.22)
    wanted = {0: [_Row(1, (10, 10, 90, 190))]}
    votes, _ = read_jerseys(_frames(), wanted, reader=reader, min_conf=0.15)
    assert votes[1] == collections.Counter({18: 1})


def test_a_box_below_the_height_floor_is_never_read():
    reader = _StubReader()
    wanted = {0: [_Row(1, (10, 10, 90, 45))]}          # 35 px tall
    votes, _ = read_jerseys(_frames(), wanted, reader=reader, min_conf=0.1,
                            min_box_h=40)
    assert votes == {} and reader.crops == []
    votes, _ = read_jerseys(_frames(), wanted, reader=reader, min_conf=0.1,
                            min_box_h=28)
    assert votes[1] == collections.Counter({18: 1})


def test_upscale_reaches_the_reader():
    """The crop handed to OCR must actually be resized by the factor given."""
    small, big = _StubReader(), _StubReader()
    wanted = {0: [_Row(1, (10, 10, 90, 190))]}
    read_jerseys(_frames(), wanted, reader=small, min_conf=0.1, upscale=2.0)
    read_jerseys(_frames(), wanted, reader=big, min_conf=0.1, upscale=4.0)
    assert big.crops[0][0] == 2 * small.crops[0][0]
    assert big.crops[0][1] == 2 * small.crops[0][1]
