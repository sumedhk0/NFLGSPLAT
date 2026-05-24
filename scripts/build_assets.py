"""Author the one-time generic assets into the avatar library.

Populates the reserved library slots the avatar stage depends on:

- ``__football__`` — canonical football (oriented along the Kalman velocity).
- ``__referee__`` — generic striped-shirt avatar for officials.

Idempotent: skips slots that already exist unless ``--force``. Run once per
season root before processing plays, or referee tracks raise a SetupError.

Usage::

    python scripts/build_assets.py --season 2024
"""
from __future__ import annotations

import typer

from nfl_gsplat.avatars.generic_assets import make_referee_avatar
from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.ball.ball_asset import make_football_asset
from nfl_gsplat.cli import CONFIG_OPT, SET_OPT, load_cli_config
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    season: str = typer.Option("0", help="Library season root."),
    library_root: str = typer.Option("library"),
    force: bool = typer.Option(False, help="Rebuild even if the asset exists."),
    config=CONFIG_OPT,
    set_=SET_OPT,
) -> None:
    load_cli_config(config, None, set_)        # validate config loads; reserved for future knobs
    lib = AvatarLibrary(library_root, season=season, rebuild=force)

    if force or not lib.has_football_asset():
        lib.put_football_asset(make_football_asset(), provenance={"source": "make_football_asset"})
        _LOG.info("authored football asset (__football__)")
    else:
        _LOG.info("football asset present; skipping (use --force to rebuild)")

    if force or not lib.has_referee_avatar():
        lib.put_referee_avatar(make_referee_avatar(), provenance={"source": "make_referee_avatar"})
        _LOG.info("authored generic referee avatar (__referee__)")
    else:
        _LOG.info("referee avatar present; skipping (use --force to rebuild)")


if __name__ == "__main__":
    app()
