"""05r's frame selection: a sideline frame is fitted only where no record of the id lies within the reach."""
import importlib.util
from pathlib import Path

import pytest


def _load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "05r_refit_endzone_only.py"
    spec = importlib.util.spec_from_file_location("refit_endzone_only_05r", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:                                   # the fit's own imports (smplx env)
        pytest.skip(f"05r imports: {e}")
    return mod


def test_select_frames_skips_frames_a_record_reaches():
    m = _load()
    assert m.select_frames([10, 20], [8, 12, 14, 16, 24], 3) == [14, 16, 24]
    assert m.select_frames([], [2, 4], 3) == [2, 4]
