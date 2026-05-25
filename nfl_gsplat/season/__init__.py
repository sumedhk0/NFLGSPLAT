"""Season-level orchestration helpers (cache-correct avatar build, betas refine).

These drive the staged SLURM DAG: after per-play perception, we collect the
unique players needing avatars (one build task per uid → no library write races)
and pick each player's best reference for shape refinement.
"""
