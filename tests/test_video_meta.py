"""ffprobe_meta parses by key, not column order (regression for a real bug).

ffprobe returns the requested stream entries in the stream's natural order, not
the order asked for — broadcast clips emit ``duration`` before ``nb_frames``.
Positional parsing read duration as nb_frames and nb_frames as duration, turning
a 1302-frame / 21.7s clip into 78042 "frames" (1302 * 59.94). These tests pin
the key-based parse.
"""
from __future__ import annotations

from nfl_gsplat.utils import video


def _patch(monkeypatch, blob: str):
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(video.subprocess, "check_output", lambda *a, **k: blob)


def test_ffprobe_meta_key_parse_handles_swapped_order(monkeypatch):
    # Note: duration emitted BEFORE nb_frames (the real ffprobe ordering).
    blob = (
        "width=1920\n"
        "height=1080\n"
        "r_frame_rate=60000/1001\n"
        "duration=21.721522\n"
        "nb_frames=1302\n"
    )
    _patch(monkeypatch, blob)
    m = video.ffprobe_meta("clip.mp4")
    assert (m.width, m.height) == (1920, 1080)
    assert abs(m.fps - 59.94005994) < 1e-3
    assert m.num_frames == 1302          # not 78042


def test_ffprobe_meta_falls_back_to_duration_when_nb_frames_na(monkeypatch):
    blob = (
        "width=1280\n"
        "height=720\n"
        "r_frame_rate=30/1\n"
        "nb_frames=N/A\n"
        "duration=10.0\n"
    )
    _patch(monkeypatch, blob)
    m = video.ffprobe_meta("clip.mp4")
    assert m.num_frames == 300           # round(10.0 * 30)
