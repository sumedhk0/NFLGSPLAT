---
name: landmark-detector-bringup-risks
description: Known risks to watch when training/running the learned NFL field-landmark detector on real footage
metadata: 
  node_type: memory
  type: project
  originSessionId: 958bce15-2cda-4c3c-a75c-4a5a76a63e5e
---

The learned field-landmark detector (`nfl_gsplat/landmarks/`, merged 2026-06-26)
passes all CPU tests, but two risks only surface once a real model is trained on
real hand-labels — watch for them at bring-up:

1. **Absent-channel false positives (top risk).** `train._masked_mse` masks the
   loss by `vis`, so landmark channels NOT labeled-visible in a frame get **zero
   gradient** — the net is never taught to output ~0 for an absent class. At
   inference (`conf_thresh=0.5` on a sigmoid head) those unconstrained channels can
   fire **confident false detections** → bad correspondences → confidently-wrong
   PnP. The field-overlay diagnostic is the only guard. Mitigations to try: if
   hand-labels are COMPLETE (every visible landmark clicked), supervise known-absent
   channels toward zero (drop the mask, or add a `supervise_absent` flag); else
   raise `conf_thresh` and lean on the overlay. This is a label-completeness
   tradeoff — decide from the real data.

2. **yard-window consistency.** The schema is only correct if the SAME
   `--yard-min/--yard-max` window is used at label, train, AND infer time.
   `build_autocalib_npz_learned` now fails loud if `ckpt["classes"]` ≠ schema
   (commit d1635f5), but the window is still a hand-entered triple point (train CLI,
   sbatch args, `02_autocalibrate --yard-min/--yard-max`). The `02_autocalibrate`
   `TODO(bring-up)` is to move `model_ckpt` + window into `meta.yaml`.

Also deferred to v1 follow-up: **geometric augmentation** (spec wanted small affine
with consistent uv transform; only color/blur shipped) — the main overfit defense on
limited footage, add once basic training works.

GPU training runs on [[pace-gpu-embers-partition]]. Acceptance test = the
`scripts/diag_calib.py` field-overlay tracking the painted lines across the frame.
Related: [[next-cycle-focal-pose-from-homography]].
