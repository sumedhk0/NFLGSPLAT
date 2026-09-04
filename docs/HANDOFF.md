# NFLGSPLAT — Project Handoff / Agent Context

**Last updated:** 2026-09-04 (branch `pipeline/avatar-twin`; `main` at `93e4cf5`).
Purpose: give any agent full working context WITHOUT the chat history. Read
this, then the memory index the harness loads, then `docs/agent-context/`.

## What this project is

NFL **All-22** broadcast footage (synced sideline + endzone clips per play,
from pro.nfl.com Film Room) → camera calibration from paint and players →
detection, ground-plane linking, SMPL-X pose in both views, jersey identity →
an **avatar twin** render: real bodies (roster height/weight) at the real
positions with colours read off the footage, drawn as Gaussians from any
viewpoint. Everything runs on the local machine.

## Hard rules (do not violate)

- **Never commit real NFL video/frames.** `data/` and `kp_eval/` are
  gitignored. Diagnostics go to `C:\Users\sumedh\diag\` or the scratchpad.
- **Local only, never PACE** (user, 2026-09-02): the RTX 4080 runs everything,
  so results can be looked at here. **GPU embargo until 2026-09-03 12:00**
  (user's other project); CPU work is fine meanwhile.
- **The machine gets switched off at will.** Long work must be resumable:
  `scripts/pipeline_play.sh` leaves `.done_<stage>` markers per play-dir;
  re-run the same command to resume. Keep stages under ~10 min.
- **Fail loud** (`SetupError`/`CalibrationError` + pointer). No silent fallback.
- **Corrections must beat what they correct, measured**, and a calibration
  correction needs a SECOND ruler before it is applied (five cases where a
  sensible prior made things worse are in memory).
- Commit and push freely, small and often, `git commit -F <file>`; messages
  say what was tried and rejected. End with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the
  `Claude-Session:` line the harness gives.
- Substantial work gets an independent review before a `--no-ff` merge.
- Secrets: Roboflow key at `C:/Users/sumedh/.roboflow_key` (never echo, never
  quote a Roboflow error URL); NFL bearer is borrowed from the logged-in
  browser and never written down; Kaggle needs `~/.kaggle/access_token`.

## The pipeline (All-22, two views) — stage order matters

```
paint     scripts/08_reconstruct_all22.py     sideline cameras per frame from the yard-line grid,
                                              lens/distance from player box heights. Candidates are
                                              then judged by THREE independent tests before any
                                              endzone solve (each has caught a camera the other two
                                              passed): paint rulers (hash rows 3.124 m, numeral rows
                                              12.50 m; agree within 10%, scale 0.80-1.25), player
                                              height (1.55-2.15 m; catches the wrong lens branch),
                                              grid-on-paint (calibration.grid_fit: projected 5-yd
                                              lines within 25 px of the detected ones; a skewed
                                              camera reads 149 px, a right one 10). Endzone
                                              reconciliation cannot veto a wrong sideline. Player
                                              cost is a loose 0.75 pre-filter.
refine    scripts/08e_refine_cameras.py       every frame's camera onto the painted yard lines
                                              (calibration.refine_paint: rotation + focal per frame,
                                              centre fixed, deltas smoothed along the track); play 1
                                              grid 9.5 -> 4.0 px, jitter p95 0.10 deg
shift     scripts/08d_field_offset.py --no-rows --apply
                                              reads the painted NUMERALS through the camera and
                                              votes the 5-yard shift -> cameras in the rule-book
                                              frame (both views). Validated by the play description
                                              in the clip filename (--los-yards). Also prints two
                                              cross-field rulers (hash rows, numeral rows).
endzone   scripts/08 --sideline-from <play>   endzone camera from the players (from_players):
                                              mount grid (now absolute), lens from boxes, rotation
                                              per frame; the mirrored mount is solved too and the
                                              NUMERALS decide which end the camera is behind.
export    scripts/08b_export_play_dir.py      cameras.npz for every frame + YOLO on every frame,
                                              feet fused across views, linked on the turf
                                              (tracking.link3d) -> tracks.parquet, track_id == player
check     scripts/08d --los-yards N           rulers + line of scrimmage, nothing applied. The
                                              pipeline FAILS the play here on DISAGREE (rulers) or
                                              MISMATCH (formation vs the description's yard line).
pose      scripts/05c_pose_play.py x2         SMPLest-X per view (endzone with --match-frames)
identity  scripts/08c_identity_all22.py       OCR votes per player (both views), roster (nflverse)
fuse      scripts/05e_fuse_views.py           joints both views agree on, placed at the compromise
refit     scripts/05f_refit_fused.py          SMPL-X params refit to the fused joints (FK forward)
fit       scripts/05i_fit_appearance.py       per-body Gaussian colour/scale/opacity fitted to the
                                              footage (compositing.splat_torch, sparse differentiable
                                              splatter; fit_appearance), held-out L1 vs the median
                                              texture printed per body -> <play>/appearance/
field     scripts/05l_field_from_footage.py   every 4th calibrated frame of each camera warped onto
                                              the ground plane, person boxes masked, per-texel median
                                              (players move out), procedural field colour-matched
                                              where the cameras never looked -> <play>/field_texture.npz
                                              and a PNG in diag: LOOK -- the footage's goal line,
                                              numerals and end zone must sit on the drawn ones (this
                                              caught play 2's 80-yard frame error)
hifi      scripts/05k_render_hifi.py          THE DELIVERABLE: render.timeline gives every detected
                                              player a body every frame (fused / single-view / median
                                              stance, SLERP between posed frames, tilt clamped 35 deg,
                                              view-aware de-duplication: a one-view id within 4 m
                                              along its camera's depth axis of a two-view id is the
                                              same man); --stitch joins fragments (tracking.stitch)
                                              so a player keeps one id and texture -- OFF until the
                                              harness says the joins are right; sparse GPU splatter
                                              at 1920x1080, --resume, mp4 out. Pipeline flags:
                                              --field-texture (the footage field), --helmets (team
                                              shell on the head), --follow --eye-offset 2 -26 10
                                              --fov 50 (dolly on the smoothed centroid, ~16 s/frame;
                                              the 34 m camera is 10 s, a 20 m one 50 s); --pads
                                              (shoulders out) built, unjudged
render    scripts/05d_render_play.py          preview, world mode: bodies at the fused placement with the
                                              fitted appearance (--fitted-appearance); CPU preview
                                              splatter; --ply-dir writes scene PLYs for 05h (gsplat)
```
`bash scripts/pipeline_play.sh <play-dir> <side.mp4> <end.mp4> <los-yards> [--fresh] [--from-paint]`
runs all of it, resumable. Two environments: `C:\venvs\nflgsplat` (py3.14,
numpy 2; calibration, tracking, identity) and `C:\venvs\smplx312` (py3.12,
numpy 1; pose, fuse, refit, render). **Pickles written under numpy 2 do not load
under numpy 1** — the pose caches (`poses_*.json` are pickles) must be written
in smplx312. `08c --cpu` still opens a CUDA context: treat 08, 08c, 08d and
05c as GPU stages.

## Current state (2026-09-04)

Two plays of BAL@KC 2024 wk1 run end to end unattended through
`scripts/pipeline_play.sh`: `data/all22/bal_at_kc_2024_wk1/play_001` =
`001_Sideline_KC_2-20_BLT_24` + `002_Endzone_KC_2-20_BLT_24` (KC 2nd-and-20
at the BAL 24); `play_002` = `004_Sideline_KC_3-9_BLT_13` +
`003_Endzone_KC_3-9_BLT_13`. The clip numbers pair by PLAY DESCRIPTION, not
by number. Play 3 (`010_Sideline_BLT_1-10_BLT_30`) fails the sideline paint
gates (players 2.8 m) and is parked.

Calibration, measured per play: sideline from paint with the rulebook
constants (hash tick centre 3.1242 m, numeral centre 12.50 m) reads both
rulers at 1.00; three judges pick the sideline candidate (rulers agree,
players 1.55–2.15 m, grid-on-paint ≤ 25 px); per-frame refinement to paint
(08e) takes play 1's grid from 9.5 to 5.4 px and play 2's from 12.7 to 9.0;
5-yard shift lands the formation at 23.3 / 14.1 yd from the goal line
against the descriptions' 24 / 13 (LOS check from the filename, with a LATE
guard when tracks start after frame 120). Endzone side of play 1 had been
mirrored; fixed.

Fidelity rework (user: "players falling over, not every player visible,
not smooth"): `render/timeline.py` draws every detected player every
frame (26–27 bodies/frame; 22 play, two deep officials, ~2 cross-camera
ghosts), ground from both views' feet, poses fused → single-view → median
stance facing velocity, per-joint SLERP, tilt clamp 35° (median tilt had
been 39°), view-aware dedupe. `05k_render_hifi.py` renders 1920×1080 on the
GPU at the source rate over the stride, ~6 s/frame, resumable.

Appearance: `05i` fits per-vertex colours to the footage (sparse torch
splatter, coverage-weighted loss, colour prior). v2 textures (60/106 and
53/85 ids) still rendered khaki: at 140 px a limb is 3–5 px wide and its
vertices sample turf-mixed pixels. v3 (commit ebd0355) drops samples within
0.12 of the frame's turf colour before the median (turf-likeness 0.72–0.84
→ 0.01 on the two bodies measured; held-out crop L1 gets slightly WORSE,
the instrument rewarded the bleed). The v3 re-fit of both plays and the
`render_hifi_v3` renders take ~5–6 h locally (2.5 min/body) and resume.

`--helmets` (render.helmet) dresses the head vertices in the team's shell;
on in the pipeline's hifi stage (a red or black shell reads as a football
player where a bare head reads as a mannequin).

**Field frame (2026-09-04).** Play 2's cameras were 80 YARDS off along
the field and every gate passed: paint is periodic and end-symmetric, a
"10" numeral backs both ends, the LOS check measures to the nearest goal
line. The footage warped onto the ground plane (`05l_field_from_footage.py`,
pipeline stage `field`, PNG in diag) is the check that caught it: the
footage's goal line, numerals and end zone must sit on the drawn ones.
Play 1 did; play 2 did not until a −73.15 m shift was applied (cameras,
poses_refit, poses_fused; `*_before_80yd.*` kept). The shift solver had
three faults (commit 8877612): strips only inside the goal lines, a
margin on shared support, and readings vetoing candidates. Play 2 now
solves −80 yd at net 11.6 vs −1.3; LOS 13.1 yd vs 13.

The same texture is the hi-fi render's field (`05k --field-texture`,
pipeline stage `hifi`): real paint, end zones and night lighting, the
procedural field colour-matched where the cameras never looked. Against
real turf the v2 khaki bodies vanish; the v3 textures are required.

Still open, in value order: identity/texture continuity across track
breaks (stitching measured no better than none, see below); the two deep
officials render as team players (stripe score and torso colour both
measured non-discriminative; position at the snap is the remaining
instrument, ~10 m behind the offence, and play 2's clip starts mid-play
so it only helps play 1); cross-camera ghosts (measured 2026-09-04: a
sideline-only and an endzone-only id that project into each other's
boxes average 0.8/frame on play 1, 0.6 on play 2, before the timeline's
dedupe -- scratchpad `ghost_crosscam.py`; a reprojection merge was built
(de1e2c8) and measured: every pair it finds already falls to the
depth/across rule, bodies per frame and duplicates dropped unchanged on
both plays, so it was reverted); 25–34 default-posed states per play. `05k --follow` dollies the virtual camera with the play's smoothed
centroid (render.camera_path); judge on a clip before it goes in the
pipeline.

## What has been measured and rejected (do not re-propose without new evidence)

- Officials by shirt stripes (horizontal-gradient energy of the torso band): a
  continuum on real crops, players on top, at 140 and 200-260 px bodies.
- Per-frame camera refinement to the yard lines alone: the pencil of near-parallel
  lines trades rotation about their direction against focal length; it needs
  priors, a 3 deg / 20% reject and along-track smoothing, or it wanders 2 deg.

Memory `measured-dead-ends.md` has the numbers. Headlines: endzone from paint
(dead); per-detection team labels and looser gates in the linker (worse);
generic and football-trained re-ID embeddings (neutral / worse than ImageNet
at the linker's question); a whole-field coverage gate (rejected the right
camera); numerals as the only ruler (constant was wrong — the hashes caught it).

## Open items, in value order

1. Verify the constant fix, rebuild both plays (~2 h each, unattended).
2. Identity: ~30% of ids named; levers not yet measured: facing-gated OCR
   from the poses, whole-play voting, roster priors.
3. Tracking continuity: players still come in pieces (fused ids); motion +
   pose consistency is the unmeasured lever; appearance is dead.
4. Real Gaussian appearance (optimise against both views) and a local
   `gsplat` render (needs the CUDA toolkit for the JIT).
5. Generalise across the ~160 downloaded play pairs (paint-solve dead zone
   35–55 deg handled by `--vertical-deg 45`; expect new failure classes).
6. The 180-degree field turn is unobservable from paint; take it from play
   metadata when it matters.

## Environment

RTX 4080 Laptop, 12 GB, driver 595.71 / CUDA 13.2. `C:\venvs\nflgsplat`
(Python 3.14.7; torch 2.11 cu128, ultralytics 8.4, numpy 2.4.4, opencv 5,
easyocr) and `C:\venvs\smplx312` (Python 3.12; torch cu128, smplx, numpy
1.26, pyarrow, imageio, av). Venvs live at short paths on purpose
(`LongPathsEnabled=0`). Do NOT `pip install -e .` (pyproject pins numpy<2 for
paddle/mmcv); run with `PYTHONPATH` set to the repo. ffmpeg 9 via winget.
SMPL-X models under `data/body_models/smplx/` (license-gated). Rosters:
`scripts/fetch_nflverse_rosters.py` → `data/rosters/2024/`.
