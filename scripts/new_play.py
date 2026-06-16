"""Scaffold a new play folder + meta.yaml stub.

    python scripts/new_play.py --season 2024 --week 1 --away NO --home ATL \
        --play play_001 --fps 30 [--gsis-play-id 36] [--force]

Creates data/{season}/week_NN/{away}_at_{home}/{play}/ and a meta.yaml stub;
drop sideline.mp4 + endzone.mp4 into the printed folder afterward.
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.season.scaffold import scaffold_play

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


@app.command()
def main(
    season: str = typer.Option(...),
    week: int = typer.Option(...),
    away: str = typer.Option(...),
    home: str = typer.Option(...),
    play: str = typer.Option("play_001"),
    fps: float = typer.Option(30.0),
    gsis_play_id: str = typer.Option("", "--gsis-play-id"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    # Convert empty string to None for scaffold_play
    gsis_id = gsis_play_id if gsis_play_id else None
    pd = scaffold_play(data_root, season=season, week=week, away=away, home=home,
                       play=play, fps=fps, gsis_play_id=gsis_id, force=force)
    print(f"created {pd.dir}")
    print(f"  -> drop {pd.video('sideline')} and {pd.video('endzone')}")


if __name__ == "__main__":
    app()
