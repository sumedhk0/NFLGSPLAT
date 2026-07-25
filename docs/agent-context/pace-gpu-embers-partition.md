---
name: pace-gpu-embers-partition
description: All NFLGSPLAT GPU jobs on PACE Phoenix must run on the embers partition
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 958bce15-2cda-4c3c-a75c-4a5a76a63e5e
---

Any NFLGSPLAT work that runs on the GPU (model training, inference jobs, anything
CUDA) must be submitted to the **embers** partition on GT PACE Phoenix.

**Why:** user directive (2026-06-26), during the learned field-landmark detector
build. embers is Phoenix's preemptible/backfill partition (cheaper/free-tier); the
user wants GPU compute kept there.

**How to apply:** in SLURM scripts / `sbatch` flags for GPU jobs use
`--partition=embers` (with the project's charge account `paceship-pso` and a GPU
gres request). Jobs may be preempted on embers, so GPU code must checkpoint and be
resumable. CPU-only steps (labeling tool, dataset prep, tests) are unaffected.
Related: [[next-cycle-focal-pose-from-homography]].
