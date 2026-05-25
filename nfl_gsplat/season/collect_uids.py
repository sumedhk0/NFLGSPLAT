"""Collect the unique player uids needing avatars across the season.

This is the dedup that makes caching correct under cluster parallelism: the
avatar-build SLURM array runs **one task per unique uid** (never two jobs racing
to write the same ``library/{season}/{uid}``). We scan every play's
``entities.json``, take the distinct player uids, and drop those already cached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.identity.registry import EntityType
from nfl_gsplat.utils.io import read_json


def find_entities_files(outputs_root: Path | str, game_ids: Iterable[str]) -> list[Path]:
    """All ``entities.json`` under ``outputs/{game}/*/`` for the given games."""
    root = Path(outputs_root)
    out: list[Path] = []
    for game in game_ids:
        out.extend(sorted((root / game).glob("*/entities.json")))
    return out


def collect_player_uids(entities_files: Iterable[Path | str]) -> set[str]:
    """Distinct ``player_uid``s of entity_type player across the given files."""
    uids: set[str] = set()
    for path in entities_files:
        for ent in read_json(path):
            if ent.get("entity_type") == EntityType.PLAYER.value and ent.get("player_uid"):
                uids.add(ent["player_uid"])
    return uids


def uids_to_build(entities_files: Iterable[Path | str], library: AvatarLibrary) -> list[str]:
    """Sorted player uids that are not yet in the library (the S3 array work)."""
    return sorted(u for u in collect_player_uids(entities_files) if not library.has_avatar(u))


def write_worklist(path: Path | str, uids: list[str]) -> Path:
    """Write one uid per line (the avatar-build array reads line ``$SLURM_ARRAY_TASK_ID``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(uids) + ("\n" if uids else ""))
    return path


# --- CLI: emit the avatar-build worklist for the season --------------------

def _main() -> None:
    import typer

    from nfl_gsplat.config import load_config

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(
        season: str = typer.Option(...),
        games: list[str] = typer.Option(..., help="Game ids to scan."),
        outputs_root: Path = typer.Option(Path("outputs")),
        library_root: Path = typer.Option(Path("library")),
        worklist: Path = typer.Option(Path("outputs/avatar_worklist.txt")),
    ) -> None:
        load_config()  # validate config is loadable
        files = find_entities_files(outputs_root, games)
        lib = AvatarLibrary(library_root, season=season)
        uids = uids_to_build(files, lib)
        write_worklist(worklist, uids)
        print(f"{len(uids)} uids to build → {worklist}")

    app()


if __name__ == "__main__":
    _main()
