"""Shared CLI plumbing for stage entry points.

Each stage module exposes a ``typer`` app whose ``main`` takes the standard
``--game / --play / --config / --config-override / --set`` flags and loads its
effective config via :func:`load_cli_config`. The shell orchestration
(``scripts/04_process_play.sh``) and the SLURM arrays invoke them as
``python -m nfl_gsplat.<stage> --game … --play …``.

Stage template::

    import typer
    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(game: str = typer.Option(...), play: str = typer.Option(...),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT):
        cfg = load_cli_config(config, config_override, set_)
        ...

    if __name__ == "__main__":
        app()
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import typer
from omegaconf import DictConfig

from nfl_gsplat.config import DEFAULT_CONFIG, load_config

CONFIG_OPT = typer.Option(DEFAULT_CONFIG, "--config", help="Base pipeline config YAML.")
CONFIG_OVERRIDE_OPT = typer.Option(
    None, "--config-override", help="Stage YAML overlaid on the base config."
)
SET_OPT = typer.Option(
    None, "--set", help="Dotlist overrides, e.g. --set identity.season=2024."
)


def load_cli_config(
    config: Path = DEFAULT_CONFIG,
    config_override: Path | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Resolve a stage's effective config from the standard CLI flags."""
    extra: list[Path] = [config_override] if config_override else []
    return load_config(*extra, base=config, overrides=list(overrides) if overrides else None)
