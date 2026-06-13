"""Build one player's avatar into the library (the S3 array's per-task unit).

Runs once per unique ``player_uid``. The perception stage saves each player's
best season reference (crop + frozen betas) at
``data/{season}/_library/_refs/{uid}.npz``; this loads it, runs LHM++ (or
3DGS-Avatar for heroes), and stores the canonical avatar + betas under ``uid``.
Because the season driver schedules exactly one task per uid, concurrent library
writes never collide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.avatars.lhm_wrapper import LHMConfig, generate_avatar
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.io import read_npz
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def reference_path(library_root: Path | str, season: str, uid: str) -> Path:
    return Path(library_root) / season / "_refs" / f"{uid}.npz"


def load_reference(library_root: Path | str, season: str, uid: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load ``(crop, betas)`` for ``uid``. Raises if perception didn't write it."""
    path = reference_path(library_root, season, uid)
    if not path.exists():
        raise SetupError(
            f"no reference for {uid} at {path}. The perception stage writes the "
            "best season reference per player; run S2 first. See SETUP.md §9."
        )
    d = read_npz(path)
    return d["crop"], d.get("betas")


def build_one_avatar(
    season: str,
    uid: str,
    library: AvatarLibrary,
    *,
    is_hero: bool = False,
    generate_fn: Callable[[np.ndarray, LHMConfig], dict] = generate_avatar,
    hero_fn: Callable[[str], dict] | None = None,
    lhm_cfg: LHMConfig | None = None,
) -> Path:
    """Generate + store one avatar. Skips if already cached (unless rebuild)."""
    if library.has_avatar(uid):
        _LOG.info(f"{uid} already in library; skipping")
        return library._avatar_path(uid)

    crop, betas = load_reference(library.root, "", uid)
    if is_hero:
        if hero_fn is None:
            raise SetupError(f"{uid} is a hero but no 3DGS-Avatar builder provided.")
        avatar = hero_fn(uid)
    else:
        avatar = generate_fn(crop, lhm_cfg or LHMConfig())

    library.put_avatar(
        uid, avatar, betas=betas, entity_type="player",
        provenance={"source": "3dgs_avatar" if is_hero else "lhm", "season": season},
    )
    _LOG.info(f"built {'hero ' if is_hero else ''}avatar for {uid}")
    return library._avatar_path(uid)


def _main() -> None:
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, SET_OPT, load_cli_config

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(
        season: str = typer.Option(...),
        uid: str = typer.Option(...),
        data_root: Path = typer.Option(Path("data"), "--data-root"),
        config=CONFIG_OPT,
        set_=SET_OPT,
    ) -> None:
        cfg = load_cli_config(config, None, set_)
        heroes = set(str(h) for h in (cfg.avatars.heroes or []))
        library_root = Path(data_root) / str(season) / "_library"
        lib = AvatarLibrary(root=library_root, season="",
                            rebuild=bool(cfg.avatars.library.rebuild))
        build_one_avatar(season, uid, lib, is_hero=uid in heroes)

    app()


if __name__ == "__main__":
    _main()
