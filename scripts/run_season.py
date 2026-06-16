"""Submit the staged season DAG to SLURM (PACE).

Stages (see the plan / SETUP.md §9):

    S2 perception_array   per play  (GPU)         field folded in
    tail                  collect uids + submit S3/S4 (CPU)   depends: all S2
    S3 avatar_build_array per unique uid (GPU)    depends: tail
    S4 render_array       per play  (GPU)         depends: S3

Plays are discovered by walking the per-play filesystem tree
(``data/{season}/week_NN/{matchup}/play_NNN``). The avatar array size isn't
known until perception finishes (it depends on how many distinct players
appear), so a small CPU "tail" job runs ``collect_uids`` and submits S3 + S4
with the right ``--array`` range. ``--dry-run`` prints the plan without
submitting.

Usage::

    python scripts/run_season.py --config configs/season.yaml --dry-run
    python scripts/run_season.py --config configs/season.yaml --submit
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from omegaconf import OmegaConf

from nfl_gsplat.season.dag import build_submission_plan

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/season.yaml")),
    dry_run: bool = typer.Option(True, "--dry-run/--submit"),
) -> None:
    cfg = OmegaConf.load(str(config))
    from nfl_gsplat.season.discover import discover_plays
    data_root = OmegaConf.select(cfg, "paths.data_root") or "data"
    plays = discover_plays(data_root, cfg.season)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/play_worklist.txt").write_text(
        "\n".join(str(p.dir) for p in plays) + ("\n" if plays else "")
    )
    plan = build_submission_plan(cfg, plays)
    print(f"# Season DAG plan ({len(plays)} plays):\n")
    for step in plan:
        print(step)
    if dry_run:
        print("\n# dry run — nothing submitted. Re-run with --submit.")
        return
    raise typer.Exit(
        code=subprocess.call(["bash", "-c", "echo 'submit path: wire job-id capture on PACE'"])
    )


if __name__ == "__main__":
    app()
