"""Tests for the nerfstudio transforms.json builder.

The confidence filter is the load-bearing part. A CameraTrack returns a
well-formed K/R/t for EVERY frame index, including ones it never solved -- those
are filled from the nearest valid neighbour. Nothing about the returned pose says
"interpolated", so an unverified frame silently becomes training data.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.errors import SetupError
from nfl_gsplat.field.build_transforms import build_transforms_json


def _track(conf):
    n = len(conf)
    return CameraTrack(
        K=np.tile(np.eye(3), (n, 1, 1)) * 1000.0,
        R=np.tile(np.eye(3), (n, 1, 1)),
        t=np.zeros((n, 3)),
        conf=np.asarray(conf, float),
        width=1920, height=1080)


def test_unverified_frames_are_dropped():
    """A conf=0 frame carries a neighbour's pose, measured 8 px off the paint
    during an endzone pan. It must not reach the training set."""
    track = _track([1.0, 0.0, 1.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transforms.json"
        frames = [(i, Path(tmp) / f"r00_{i:06d}.png") for i in range(4)]
        build_transforms_json({"endzone": track}, {"endzone": frames}, out)
        got = json.loads(out.read_text())
    kept = sorted(int(Path(f["file_path"]).stem.split("_")[-1])
                  for f in got["frames"])
    assert kept == [0, 2], f"unverified frames leaked into transforms: {kept}"


def test_min_conf_none_keeps_everything():
    """The escape hatch must actually escape -- training on unverified poses is
    a deliberate choice an operator is allowed to make."""
    track = _track([1.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transforms.json"
        frames = [(i, Path(tmp) / f"r00_{i:06d}.png") for i in range(2)]
        build_transforms_json({"endzone": track}, {"endzone": frames}, out,
                              min_conf=None)
        got = json.loads(out.read_text())
    assert len(got["frames"]) == 2


def test_all_unverified_fails_loud():
    """Zero usable frames must raise, not write an empty training set."""
    track = _track([0.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transforms.json"
        frames = [(i, Path(tmp) / f"r00_{i:06d}.png") for i in range(2)]
        with pytest.raises(SetupError, match="unverified"):
            build_transforms_json({"endzone": track}, {"endzone": frames}, out)
