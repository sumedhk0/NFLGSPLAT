"""Assemble the season SLURM submission plan (pure → unit-testable).

Kept out of ``scripts/run_season.py`` so the DAG structure can be tested without
a cluster. ``build_submission_plan`` returns the ordered sbatch commands for
S1 (field) → S2 (perception) → tail (collect uids + S3) → S4 (render).
"""
from __future__ import annotations

from pathlib import Path


def num_plays_range(game: str, plays_dir: Path | str = "configs/plays") -> str:
    """``1-N`` array range for a game's play list, or ``1-N`` placeholder."""
    pf = Path(plays_dir) / f"{game}.txt"
    if pf.exists():
        n = sum(1 for ln in pf.read_text().splitlines() if ln.strip())
        return f"1-{max(n, 1)}"
    return "1-N"


def qos_flags(cfg) -> list[str]:
    """``--qos``/``--requeue`` shared by GPU and CPU jobs.

    On PACE Phoenix ``embers`` is the free, preemptible backfill QOS; jobs there
    can be killed when a paid (``inferno``) job needs the node, so ``--requeue``
    lets SLURM restart them (our stages skip already-cached work, so a restart is
    cheap and correct). Omitted entirely when ``slurm.qos`` is unset.
    """
    s = cfg.slurm
    flags: list[str] = []
    qos = s.get("qos")
    if qos:
        flags.append(f"--qos={qos}")
    if s.get("requeue"):
        flags.append("--requeue")
    return flags


def slurm_flags(cfg) -> list[str]:
    s = cfg.slurm
    return [
        "-A", str(s.account),
        "--partition", str(s.partition),
        f"--gres=gpu:{s.gpu}",
        f"--cpus-per-task={s.cpus_per_task}",
        f"--mem={s.mem}",
        *qos_flags(cfg),
    ]


def build_submission_plan(cfg, plays_dir: Path | str = "configs/plays") -> list[str]:
    """Ordered list of sbatch command strings for the staged season DAG."""
    season = str(cfg.season)
    flags = " ".join(slurm_flags(cfg))
    qos = " ".join(qos_flags(cfg))            # CPU tail job: same QOS, no GPU alloc
    qos = f"{qos} " if qos else ""
    plan: list[str] = []
    for game in cfg.games:
        plan.append(
            f"sbatch --parsable {flags} --time={cfg.slurm.time_field} "
            f"scripts/slurm/field_recon.sbatch {game}   # S1 field[{game}]"
        )
        plan.append(
            f"sbatch --parsable {flags} --time={cfg.slurm.time_perception} "
            f"--dependency=afterok:$FIELD_{game} --array={num_plays_range(game, plays_dir)} "
            f"--export=ALL,NFL_GAME={game} scripts/slurm/perception_array.sbatch   # S2 perception[{game}]"
        )
    games_flags = " ".join(f"--games {g}" for g in cfg.games)
    plan.append(
        f"sbatch --parsable {qos}--dependency=afterok:$ALL_PERCEPTION "
        f'--wrap="python -m nfl_gsplat.season.collect_uids --season {season} {games_flags} && '
        f"sbatch {flags} --time={cfg.slurm.time_avatar} "
        "--array=1-$(wc -l < outputs/avatar_worklist.txt) "
        f'--export=ALL,NFL_SEASON={season} scripts/slurm/avatar_build_array.sbatch"   # tail: collect + S3'
    )
    for game in cfg.games:
        plan.append(
            f"sbatch {flags} --time={cfg.slurm.time_render} "
            f"--dependency=afterok:$AVATAR_JOB --array={num_plays_range(game, plays_dir)} "
            f"--export=ALL,NFL_GAME={game},NFL_SEASON={season} "
            f"scripts/slurm/render_array.sbatch   # S4 render[{game}]"
        )
    return plan
