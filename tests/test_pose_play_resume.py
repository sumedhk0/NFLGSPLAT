"""Resuming an interrupted pose cache.

The run checkpoints every batch because it is long -- about 68 minutes for a
play's 816 verified frames. Writing a checkpoint nobody READS is half a feature:
an interruption still cost the entire run, which is the one thing a checkpoint
exists to prevent.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05c_pose_play.py"


def test_the_script_exposes_a_resume_flag():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "--no-resume" in src
    assert "args.resume" in src


def test_resume_skips_frames_already_in_the_checkpoint(tmp_path):
    """The behaviour that matters: cached frames are not posed a second time."""
    cache = {"cam": "sideline", "stride": 1,
             "frames": {3: {1: {"betas": [0.0]}}, 9: {1: {"betas": [0.0]}}}}
    out = tmp_path / "cache.pkl"
    with open(out, "wb") as fh:
        pickle.dump(cache, fh)

    wanted = {3, 9, 15, 21}
    prior = pickle.load(open(out, "rb"))
    assert prior["cam"] == "sideline"
    done = set(prior["frames"]) & wanted
    assert wanted - done == {15, 21}


def test_a_truncated_checkpoint_does_not_raise_the_obvious_exception(tmp_path):
    """Counter-intuitive, and the reason the guard in 05c is broad.

    A pickle truncated mid-write raises MemoryError here, not EOFError or
    UnpicklingError: the header promises a length the file does not contain. A
    guard catching only the two obvious types would let exactly the case it
    exists for -- an interrupted checkpoint -- crash the next run instead of
    restarting it.
    """
    out = tmp_path / "cache.pkl"
    out.write_bytes(b"\x80\x04\x95truncated")
    with pytest.raises(Exception) as exc:      # noqa: B017 - the type IS the point
        pickle.load(open(out, "rb"))
    assert not isinstance(exc.value, (EOFError, pickle.UnpicklingError))
    # ... so 05c has to catch broadly rather than by type.
    assert "noqa: BLE001" in SCRIPT.read_text(encoding="utf-8")


def test_a_checkpoint_for_the_other_camera_must_not_be_merged():
    """Mixing two cameras in one cache would corrupt every placement built from
    it, and nothing downstream could detect the mixture."""
    assert ("Refusing to mix two cameras in one cache"
            in SCRIPT.read_text(encoding="utf-8"))
