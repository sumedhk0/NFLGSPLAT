"""Tests for the gated body-model installer.

Built with synthetic archives, since the real ones are licence-gated and cannot
live in a test fixture. What is being guarded is the mechanical part that is
easy to get wrong and fails LATE: the two zips nest their files differently, and
SMPL ships basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl where every consumer here
expects SMPL_NEUTRAL.pkl.
"""
from __future__ import annotations

import importlib.util
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "install_body_models",
    Path(__file__).resolve().parents[1] / "scripts" / "00b_install_body_models.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _payload():
    return {"v_template": np.zeros((10, 3)), "shapedirs": np.zeros((10, 3, 2)),
            "J_regressor": np.zeros((2, 10)), "f": np.zeros((4, 3))}


def _smplx_zip(path, *, nest="models/smplx/"):
    with zipfile.ZipFile(path, "w") as zf:
        for gender in ("NEUTRAL", "MALE", "FEMALE"):
            buf = path.parent / f"tmp_{gender}.npz"
            np.savez(buf, **_payload())
            zf.write(buf, f"{nest}SMPLX_{gender}.npz")
            buf.unlink()
    return path


def _smpl_zip(path, *, nest="SMPL_python_v.1.1.0/smpl/models/"):
    names = {"neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
             "m": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
             "f": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"}
    with zipfile.ZipFile(path, "w") as zf:
        for fname in names.values():
            buf = path.parent / "tmp.pkl"
            buf.write_bytes(pickle.dumps(_payload()))
            zf.write(buf, nest + fname)
            buf.unlink()
    return path


def test_smplx_lands_in_the_canonical_layout(tmp_path):
    dest = tmp_path / "body_models"
    mod.install([_smplx_zip(tmp_path / "models_smplx_v1_1.zip")], dest)
    assert (dest / "smplx" / "SMPLX_NEUTRAL.npz").exists()
    assert (dest / "smplx" / "SMPLX_MALE.npz").exists()


def test_smpl_basicmodel_names_are_renamed(tmp_path):
    """The rename is the whole point: consumers ask for SMPL_NEUTRAL.pkl."""
    dest = tmp_path / "body_models"
    mod.install([_smpl_zip(tmp_path / "SMPL_python_v.1.1.0.zip")], dest)
    assert (dest / "smpl" / "SMPL_NEUTRAL.pkl").exists(), "neutral not renamed"
    assert (dest / "smpl" / "SMPL_MALE.pkl").exists()
    assert (dest / "smpl" / "SMPL_FEMALE.pkl").exists()


def test_nesting_depth_does_not_matter(tmp_path):
    """Matching is on member substrings, so a differently nested archive works."""
    dest = tmp_path / "body_models"
    mod.install([_smplx_zip(tmp_path / "a.zip", nest="some/deeper/path/")], dest)
    assert (dest / "smplx" / "SMPLX_NEUTRAL.npz").exists()


def test_validate_accepts_a_real_looking_model(tmp_path):
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(path, **_payload())
    assert "ok" in mod.validate(path)


def test_validate_rejects_a_file_missing_body_model_arrays(tmp_path):
    """A truncated or wrong-file download is otherwise indistinguishable from a
    good one until inference crashes much later."""
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(path, something_else=np.zeros(3))
    with pytest.raises(SystemExit, match="missing"):
        mod.validate(path)


def test_non_zip_input_is_skipped_not_fatal(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not an archive")
    notes, done = mod.install([junk], tmp_path / "body_models")
    assert done == {}
    assert any("skip" in n for n in notes)
