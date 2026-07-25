---
name: next-cycle-focal-pose-from-homography
description: "RESOLVED 2026-07-07 — fixed-center joint solve shipped; first full-play cameras.npz. Remaining follow-ups listed."
metadata:
  node_type: memory
  type: project
  originSessionId: 958bce15-2cda-4c3c-a75c-4a5a76a63e5e
---

**RESOLVED (2026-07-07, merge fc85104):** the near-affine focal/pose problem
is solved by the fixed-center joint solve (`nfl_gsplat/calibration/
joint_solve.py`, pretrained mode phase 3). First full-play `cameras.npz`
produced for SEA_at_AZ play_001: 816/986 field-visible frames ≤ 6 px (median
1.39 px), camera center (−3.6, 80.5, 35.9) m (std ±0.3 m), focals
5954–8323 px tracking the zoom. Grid overlays from metric K,R,t track the
painted field at both zoom regimes.

**Load-bearing lessons (spec addendum, 2026-07-06 design doc):**
- Per-frame PnP on planar telephoto = multimodal garbage; never trust it,
  not even as an initializer. Multi-start grid + look-at/span-focal seeding.
- Left/right hash labeling is CAMERA-SIDE dependent; a mirrored labeling
  keeps homographies perfect but no rigid camera fits it. Joint solve
  auto-resolves by scoring both reflections (this play: camera on +Y side,
  fusion convention was flipped; `mirrored=True` logged).
- Broadcast graphics (watermark/score bug) poison hash-row fitting; static
  16 px-cell census across the play (>25 % occupancy = graphics) filters them.
- Solve order matters: reflection resolve → per-frame rescue refit at the
  multi-start winner → staged solve on CLEAN frames only → final rescue of
  all frames. Solving on all frames first lets the poisoned minority drag C.

**Follow-ups (not started):**
- ~170 frames/play still rejected (early wide shots) — interpolated with
  conf=0; investigate residual fusion errors there if splat quality suffers.
- Endzone camera + multi-camera coupling (same joint-solve machinery).
- Downstream: player tracking/triangulation and gsplat training now unblocked
  by valid per-frame K,R,t.
