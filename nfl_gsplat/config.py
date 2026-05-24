"""Config loading: merge ``pipeline.yaml`` + stage YAML(s) + CLI dotlist overrides.

Every stage CLI loads its effective config the same way::

    cfg = load_config(overrides=["identity.season=2024", "avatars.library.rebuild=true"])
    cfg = load_config("configs/field_recon.yaml")   # overlay a stage file

Precedence (lowest → highest): ``pipeline.yaml`` → each extra YAML in order →
dotlist overrides. OmegaConf interpolations (e.g. ``${identity.season}``) resolve
lazily on attribute access, so a ``--set identity.season=2024`` override flows
through to ``identity.roster_path`` for free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def load_config(
    *extra_yaml: Path | str,
    base: Path | str = DEFAULT_CONFIG,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Return the merged pipeline config.

    ``base`` is loaded first (the full ``pipeline.yaml``), then each path in
    ``extra_yaml`` is overlaid, then ``overrides`` (``key=value`` dotlist) win.
    """
    layers = [OmegaConf.load(str(base))]
    for path in extra_yaml:
        layers.append(OmegaConf.load(str(path)))
    merged = OmegaConf.merge(*layers)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(merged, DictConfig)
    return merged
