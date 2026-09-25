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


def test_swallowed_names_the_man_a_wide_box_took():
    """A wide box is merged only when a man it overlapped has lost his own box, or shares it (play 1: Madubuike,
    lunging between #65 and #74, is wide on his own while #65 keeps his box beside him; at 518-522 #65's box is gone
    into his and the keypoints are #65's)."""
    m = _load()
    me = (100.0, 100.0, 300.0, 200.0)
    beside = (40.0, 90.0, 130.0, 260.0)                       # IoU with me ~0.09: a man next to him, boxed on his own
    over = (120.0, 90.0, 260.0, 230.0)                        # IoU with me ~0.55: engaged, overlapping
    box_of = {10: {4: me, 76: beside}, 11: {4: me, 76: beside}}
    assert m.swallowed(box_of, 4, 11) is None                  # beside him and still boxed: not merged
    box_of = {9: {4: me, 170: over}, 10: {4: me, 170: over}, 11: {4: me}}
    assert m.swallowed(box_of, 4, 11) == 170                   # overlapped him a frame ago, no box now: taken
    assert m.swallowed(box_of, 4, 11, window=0) is None        # ... unless the look-back is off
    box_of = {11: {4: me, 76: (105.0, 100.0, 300.0, 205.0)}}
    assert m.swallowed(box_of, 4, 11) == 76                    # one box under two ids (a twin): shared
    box_of = {2: {4: me, 170: over}, 11: {4: me}}
    assert m.swallowed(box_of, 4, 11) is None                  # overlapped him long ago: outside the window
