"""Submit the staged season DAG to SLURM (PACE).

Stages (see the plan / SETUP.md §9):

    S1 field_recon        per game (GPU)
    S2 perception_array   per play  (GPU)         depends: S1
    tail                  collect uids + submit S3/S4 (CPU)   depends: all S2
    S3 avatar_build_array per unique uid (GPU)    depends: tail
    S4 render_array       per play  (GPU)         depends: S3

The avatar array size isn't known until perception finishes (it depends on how
many distinct players appear), so a small CPU "tail" job runs ``collect_uids``
and submits S3 + S4 with the right ``--array`` range. ``--dry-run`` prints the
plan without submitting.

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
    plan = build_submission_plan(cfg)
    print(f"# Season DAG plan ({len(cfg.games)} games):\n")
    for step in plan:
        print(step)
    if dry_run:
        print("\n# dry run — nothing submitted. Re-run with --submit.")
        return
    # Real submission requires capturing each job id and substituting the
    # $FIELD_*/$ALL_PERCEPTION/$AVATAR_JOB placeholders; the dependency wiring is
    # finalized during PACE bring-up (see SETUP.md §9).
    raise typer.Exit(
        code=subprocess.call(["bash", "-c", "echo 'submit path: wire job-id capture on PACE'"])
    )


if __name__ == "__main__":
    app()
