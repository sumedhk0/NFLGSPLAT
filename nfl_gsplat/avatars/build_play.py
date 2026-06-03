"""Avatar stage CLI (single-play path): build this play's player avatars.

Reads ``entities.json``, takes the distinct player ``player_uid``s (referees and
the football are generic library assets authored once by ``scripts/build_assets.py``,
not per play), and builds each into the library via
:func:`avatars.build_one.build_one_avatar` — which itself skips uids already
cached. The per-uid reference (crop + frozen betas) was written by the pose stage.

At season scale this loop is replaced by the S3 avatar-build array (one SLURM
task per unique uid); this module is the inline path for single-play / debug runs.
"""
from __future__ import annotations

from typing import Callable, Iterable

from nfl_gsplat.avatars.build_one import build_one_avatar
from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.avatars.lhm_wrapper import LHMConfig
from nfl_gsplat.identity.registry import REFEREE_UID
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

_GENERIC_UIDS = {REFEREE_UID, "__football__"}


def player_uids(entities: Iterable[dict]) -> list[str]:
    """Distinct player ``player_uid``s to build (generic assets excluded), order-stable."""
    seen: dict[str, None] = {}
    for ent in entities:
        uid = ent.get("player_uid")
        if uid and uid not in _GENERIC_UIDS and ent.get("entity_type") == "player":
            seen.setdefault(uid, None)
    return list(seen)


def build_play_avatars(
    entities: Iterable[dict],
    season: str,
    library: AvatarLibrary,
    *,
    heroes: set[str] | None = None,
    generate_fn: Callable | None = None,
    hero_fn: Callable | None = None,
    lhm_cfg: LHMConfig | None = None,
) -> list[str]:
    """Build each play player's avatar into ``library``. Returns the uids built
    (or already cached). Cached uids are skipped by ``build_one_avatar``."""
    from nfl_gsplat.avatars.lhm_wrapper import generate_avatar

    heroes = heroes or set()
    built: list[str] = []
    for uid in player_uids(entities):
        build_one_avatar(
            season, uid, library,
            is_hero=uid in heroes,
            generate_fn=generate_fn or generate_avatar,
            hero_fn=hero_fn,
            lhm_cfg=lhm_cfg,
        )
        built.append(uid)
    _LOG.info(f"build_play: {len(built)} player avatars ensured in library")
    return built


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import play_paths
    from nfl_gsplat.utils.io import read_json

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(game: str = typer.Option(...), play: str = typer.Option(...),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pp = play_paths(cfg, game, play)
        season = str(cfg.identity.season)
        entities = read_json(pp.entities)
        heroes = {str(h) for h in (cfg.avatars.heroes or [])}
        library = AvatarLibrary(pp.game.library_root, season=season,
                                rebuild=bool(cfg.avatars.library.rebuild))
        build_play_avatars(entities, season, library, heroes=heroes)

    app()


if __name__ == "__main__":
    _main()
