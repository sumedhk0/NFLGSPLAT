"""Reading video must not depend on one decoder being loadable.

pyav is a compiled extension. On this machine Windows Application Control began
blocking its DLL mid-session, and because the reader named that plugin
explicitly, every video test failed at once with an ImportError that reads like
a missing package rather than a blocked one.
"""
import imageio.v3 as iio
import numpy as np
import pytest

from nfl_gsplat.utils import video


def test_plugin_order_prefers_pyav_but_does_not_require_it():
    assert video._VIDEO_PLUGINS[0] == "pyav"
    assert len(video._VIDEO_PLUGINS) > 1


def test_falls_back_when_the_preferred_decoder_cannot_load(monkeypatch, tmp_path):
    """A blocked decoder must be stepped over, not fatal."""
    calls = []

    def fake(path, plugin=None):
        calls.append(plugin)
        if plugin == "pyav":
            raise ImportError("The `pyav` plugin is not installed.")
        return iter([np.zeros((4, 4, 3), np.uint8)])

    monkeypatch.setattr(iio, "imiter", fake)
    got = video._imiter(str(tmp_path / "clip.mp4"))
    assert next(iter(got)).shape == (4, 4, 3)
    assert calls[0] == "pyav"          # tried first
    assert len(calls) > 1              # and moved on


def test_says_so_when_no_decoder_works(monkeypatch, tmp_path):
    def broken(path, plugin=None):
        raise ImportError("nope")

    monkeypatch.setattr(iio, "imiter", broken)
    with pytest.raises(RuntimeError, match="no imageio plugin"):
        video._imiter(str(tmp_path / "clip.mp4"))
