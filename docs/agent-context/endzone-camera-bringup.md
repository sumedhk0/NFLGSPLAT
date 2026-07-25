---
name: endzone-camera-bringup
description: "Endzone camera: field-markings EXHAUSTED, geometric cross-camera FAILED on real data (87px mispairs, Y-reflection outscores truth). Jersey-IDENTITY approach BUILT + reviewed merge-ready on branch jersey-identity-endzone (unmerged, pending real acceptance). Synthetic recovery 0.062m."
metadata:
  node_type: memory
  type: project
  originSessionId: 958bce15-2cda-4c3c-a75c-4a5a76a63e5e
  modified: 2026-07-24T23:47:43.375Z
---

**CURRENT STATE (2026-07-24).** Endzone camera calibration, three approaches tried:

1. **Field markings — EXHAUSTED (~9 prototypes).** Even with rotate-90 + player
   masking (both SHIPPED, merges 922b7d8/2ddf25b), steep down-field perspective
   makes yard-line IDENTITY labels 67-210px wrong; no single camera fits. The
   yard NUMBERS are edge-on/unreadable from the endzone angle, so lines can't be
   labeled. Do NOT retry field markings for endzone.

2. **Geometric cross-camera (per-frame foot-point Hungarian) — FAILED on real
   data.** Shipped on branch `cross-camera-roll-fix` (roll-aware `_hyp_cam` +
   tight-inlier bootstrap, 3 commits, reviewed clean, synthetic recovers 0.06m)
   but real acceptance FAILED: measured on SEA@AZ play_001 — even the TRUE
   endzone camera reprojects the foot correspondences at **87px median**, only
   ~198 of ~5600 matches tight; the **Y-reflection** camera (+120,+20,roll270)
   scores MORE tight inliers (366) than truth (~200); `solve_fixed_center` keeps
   0 frames. ROOT CAUSE (fundamental): sideline's weak axis is Y (grazing rays
   at 80-100m); endzone measures Y most sensitively; so foot correspondences are
   both noisy AND ~85% mispaired without identity. Per-frame geometric matching
   is the wrong primitive. Branch `cross-camera-roll-fix` left UNMERGED
   (superseded). See [[next-cycle-focal-pose-from-homography]] for sideline.

3. **Jersey-IDENTITY cross-camera — BUILT, reviewed READY TO MERGE, branch
   `jersey-identity-endzone` (UNMERGED, pending real acceptance).** Pivot:
   correspondence by jersey identity, not geometry. Cross-camera identity is
   GEOMETRY-FREE — `identity.registry.resolve_tracks(id_col="track_id")` per
   camera (jersey OCR + team color, OcrOnlySource synthesizes
   `{season}_{team}_{jersey}` uids) then JOIN on `player_uid`; breaks the
   chicken-and-egg (`cross_cam_reid.reid_pipeline` needs both cams calibrated,
   this doesn't). Multi-play SHARED-CENTER solve (fixed tripod: one C across all
   first-half plays) via existing `solve_fixed_center` `_frame_data_override`
   path, NATIVE pixels (view_deg=0, NO rotation/roll — the 90deg rotation only
   served the dead field-marking detector). EndzonePrior (X<0 behind Seattle's
   end, |Y| small, from meta.yaml `endzone_prior:` block) bounds C to exclude the
   Y-reflection. New modules: `endzone_identity.py` (join),
   `endzone_multiplay.py` (EndzonePrior + solve_endzone_identity),
   `identity_precompute.py` (global team split + player_uid),
   `scripts/03c_identity_tracks.py` (PACE precompute), `--mode identity-endzone`
   in 02_autocalibrate. `solve_fixed_center` got default-preserving
   `center_bounds` + `audit_drop_px` params. Synthetic multi-play recovers C to
   **0.062m**, 75/75 frames; reflection-exclusion test load-bearing. 346 tests
   pass (+2 pre-existing OpenCV field_detect failures, unrelated). Reuses the
   existing `nfl_gsplat/identity/` subsystem (built for pose/avatars, never
   wired to calibration).

**REMAINING = real acceptance (the actual gate, NOT yet done).** User chose
"validate on real data before merge" (2026-07-24). Needs: several first-half
SEA@AZ plays downloaded (clean sideline.mp4/endzone.mp4), PACE precompute per
play (`scripts/03c_identity_tracks.py`, nfl_smplx env, PaddleOCR+GPU on embers)
to fill player_uid, `endzone_prior:` block in each meta.yaml, then
`02_autocalibrate.py --mode identity-endzone --play-dirs <...>`. Two review
findings deferred to acceptance (data-dependent): global team-color split could
partition by CAMERA not team (fails loud via min_frames if so); real jersey-OCR
yield must clear min_frames floor. easyocr proxy (CPU, local) measured 6
cross-camera jersey anchors from 12 sampled frames — floor, PaddleOCR+per-track
voting will do better.

**KEY DATA FACTS.** `tracks.parquet` for play_001 is DETECT-ONLY (track_id=-1,
jersey_number_ocr=-1 all rows) — real acceptance must re-run detect_and_track
(track IDs) + jersey_ocr. Sideline camera SOLVED & valid. Local env: torch
2.4.1+cpu, ultralytics 8.4.96, easyocr installed (CPU OCR proxy); PaddleOCR is
PACE-only (nfl_smplx). data/2025/week_04/SEA_at_AZ/play_001 has
sideline.mp4/endzone.mp4/cameras.npz/tracks.parquet.
