"""05p's base cache: an explicit --refit holding one-view records is never swapped silently (2026-09-24)."""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "refit_mono", Path(__file__).resolve().parents[1] / "scripts" / "05p_refit_mono.py")
refit_mono = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refit_mono)


def test_a_cache_without_one_view_records_is_its_own_base(tmp_path):
    given = tmp_path / "two_view.json"
    assert refit_mono.base_cache(given, None, tmp_path, has_mono=False) == given
    assert refit_mono.base_cache(None, None, tmp_path, has_mono=False) == tmp_path / "poses_refit.json"


def test_an_explicit_merged_cache_is_refused_without_a_named_base(tmp_path):
    (tmp_path / "poses_refit_fused.json").write_bytes(b"old")        # the trap: an old backup sits in the play dir
    with pytest.raises(SystemExit, match="--fused-base"):
        refit_mono.base_cache(tmp_path / "merged.json", None, tmp_path, has_mono=True)


def test_a_named_base_is_used_and_must_exist(tmp_path):
    base = tmp_path / "ft2_2v.json"
    with pytest.raises(SystemExit, match="missing"):
        refit_mono.base_cache(tmp_path / "merged.json", base, tmp_path, has_mono=True)
    base.write_bytes(b"x")
    assert refit_mono.base_cache(tmp_path / "merged.json", base, tmp_path, has_mono=True) == base


def test_the_pipeline_default_path_keeps_the_play_dir_backup(tmp_path):
    backup = tmp_path / "poses_refit_fused.json"
    with pytest.raises(SystemExit, match="missing"):
        refit_mono.base_cache(None, None, tmp_path, has_mono=True)
    backup.write_bytes(b"x")
    assert refit_mono.base_cache(None, None, tmp_path, has_mono=True) == backup
