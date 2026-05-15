"""Atomic-write JSON/NPZ readers + a content-hash manifest for stage idempotency.

Every stage writes a ``manifest.json`` next to its outputs with:

- ``inputs``:  {path: sha256} for each file it consumed
- ``outputs``: {path: sha256} for each file it produced
- ``config``:  OmegaConf resolved config snapshot
- ``stage``:   stage name (for debugging chained failures)

A stage can skip work if its manifest.inputs hashes still match the current
inputs on disk AND its outputs still exist. Run with ``--force`` to invalidate.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to a temp file in the same dir, then rename over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    payload = json.dumps(obj, indent=indent, sort_keys=False, default=_json_default).encode("utf-8")
    atomic_write_bytes(path, payload)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_npz(path: Path | str, **arrays: np.ndarray) -> None:
    # ``np.savez_compressed`` appends ``.npz`` to any string path without it.
    # Pass an opened file handle to dodge that, then atomic-rename.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON-serializable: {type(o).__name__}")


def manifest_path(stage_dir: Path | str) -> Path:
    return Path(stage_dir) / "manifest.json"


def write_manifest(
    stage_dir: Path | str,
    *,
    stage: str,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
    config_snapshot: Mapping[str, Any] | None = None,
) -> None:
    write_json(
        manifest_path(stage_dir),
        {
            "stage": stage,
            "inputs": dict(inputs),
            "outputs": dict(outputs),
            "config": dict(config_snapshot) if config_snapshot else {},
        },
    )


def manifest_matches(
    stage_dir: Path | str,
    *,
    expected_inputs: Mapping[str, str],
) -> bool:
    """Return True iff ``manifest.json`` exists and its input hashes match
    ``expected_inputs`` AND every listed output file still exists."""
    mpath = manifest_path(stage_dir)
    if not mpath.exists():
        return False
    try:
        m = read_json(mpath)
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("inputs") != dict(expected_inputs):
        return False
    for out_rel in m.get("outputs", {}):
        if not (Path(stage_dir) / out_rel).exists():
            return False
    return True
