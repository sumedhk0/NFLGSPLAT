"""Assemble the season SLURM submission plan (pure → unit-testable).

Kept out of ``scripts/run_season.py`` so the DAG structure can be tested without
a cluster. ``build_submission_plan`` returns the ordered sbatch commands for
S2 (perception, field folded in) → tail (collect uids + S3) → S4 (render).
"""
from __future__ import annotations


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


def build_submission_plan(cfg, plays: list) -> list[str]:
    """Ordered sbatch strings for the per-play season DAG.

    ``plays`` is the discovered ``list[PlayDir]``. Perception runs once per play
    (building that play's own field + calibration-driven reconstruction), then a
    CPU tail collects uids and submits the avatar-build array, then render runs
    per play.
    """
    season = str(cfg.season)
    flags = " ".join(slurm_flags(cfg))
    qos = " ".join(qos_flags(cfg))
    qos = f"{qos} " if qos else ""
    n = max(len(plays), 1)
    plan: list[str] = []
    plan.append(
        f"sbatch --parsable {flags} --time={cfg.slurm.time_perception} "
        f"--array=1-{n} --export=ALL,NFL_SEASON={season} "
        f"scripts/slurm/perception_array.sbatch   # S2 perception[{n} plays]"
    )
    plan.append(
        f"sbatch --parsable {qos}--dependency=afterok:$PERCEPTION "
        f'--wrap="python -m nfl_gsplat.season.collect_uids --season {season} && '
        f"sbatch {flags} --time={cfg.slurm.time_avatar} "
        "--array=1-$(wc -l < outputs/avatar_worklist.txt) "
        f'--export=ALL,NFL_SEASON={season} scripts/slurm/avatar_build_array.sbatch"   # tail: collect + S3'
    )
    plan.append(
        f"sbatch {flags} --time={cfg.slurm.time_render} "
        f"--dependency=afterok:$AVATAR_JOB --array=1-{n} "
        f"--export=ALL,NFL_SEASON={season} "
        f"scripts/slurm/render_array.sbatch   # S4 render[{n} plays]"
    )
    return plan
