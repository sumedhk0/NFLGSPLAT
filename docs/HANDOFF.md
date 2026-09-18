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

## The loop (standing process from 2026-09-16)

hypotheses -> tests for them -> code -> run -> a new footage version -> REVIEW THE FOOTAGE -> affirm or
deny -> loop. The rendered clip is the output; the rulers are instruments, not the verdict. Reviewing
the footage means frame SEQUENCES, not one still: `05q_overlay_footage.py --player N --start F --count 8
--step 2` (the drawn body over the real player, both cameras) for every id a ruler flags, plus tiled
consecutive rendered frames (ffmpeg select+tile) of the phases that matter. A ruler that says "fixed"
while the strips show flailing is a ruler to be replaced (v44: joint jitter p90 0.058 and the user
still sees flailing -- the jitter ruler cannot see a smooth, wrong sweep).

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
refit_mono scripts/05p_refit_mono.py         one-view bodies (no fused record) refit to the sideline's
                                              2-D keypoints, feet on the turf (pose.fit_mono2d); merged
                                              into poses_refit.json, the 05f cache kept as _fused.json
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
                                              the 34 m camera is 10 s, a 20 m one 50 s); --uniforms
                                              (synthetic kit by region, THE look); --pads (shoulders
                                              out) judged on a frame: negligible at 150 px, off
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

**Joints decision (2026-09-04 night, a17c5af..459bc45).** The 3-D joints
now come from GEOMETRY: 2-D keypoints per view (05m, YOLOv8-pose,
matched to the tracked boxes) triangulated with both calibrated cameras
(05n, through pose.triangulate) and refit to SMPL-X unchanged (05f on
`poses_tri.json`). The monocular regressor per view fused across views
(05e) is opt-in (FUSE=1). The camera ruler (05o: both refits projected
into both views against the detector keypoints) on play 1: monocular
22.7 / 40.7 px (sideline / endzone), triangulated 8.8 / 12.1 px, every
joint group, posed every frame instead of every sixth; play 2: 28 / 55
→ 9 / 11 px. Ground ruler: planted feet reach the turf under play 1's
cameras (raw triangulated lower ankle p10 +0.02 m), so no endzone pitch
correction there; play 2's fitted bodies float (p10 +0.22 m): its
endzone pitch is off by a few tenths of a degree and only the ankle
ruler sees it (the box-bottom gap is minimal at the current pitch by
construction). The second ruler now exists (efaed88: grid_fit scores the
endzone view's across-the-image yard lines, any orientation gated
against the projected lines) and settles it differently: on play 2 the
planted feet sit UNDER the turf at the current pitch (raw triangulated
lower ankle p10 −0.22 m) while the median floats (+0.20), and no pitch
moves both to the turf -- the float is dispersion in a sparse
triangulation, not a camera bias; the endzone paint reads 60–77 px there,
flat. On play 1 the feet are right at 0° (p10 −0.01) while the paint
prefers +0.3..0.5° (41 → 27 px): a 27 px residual is an error in another
parameter (roll, yaw, focal) a pitch cannot absorb. No pitch correction
on either play. The full endzone refinement against its paint was then
built (623eab0: `08e --cam endzone --orient-tol 25 --out`) and REFUSED
by the ankle ruler on play 1: paint 39.3 → 22.6 px, but the triangulated
lower ankle p10 −0.01 → −0.45 m, ray agreement 7.3 → 10.2 px, joints
passing the gate 47699 → 9291. The paint pulls the endzone away from
where the sideline's rays cross; the endzone's residual is not in
rotation+focal and its paint at 23–27 px is not a ruler the geometry
trusts. cameras.npz untouched (`cameras_endzone_refined.npz` kept for
the record). Do not pursue the endzone paint further.

**Play 2's right-edge column (2026-09-05).** Not one lineage at several
depths: 2–3 endzone-ONLY bodies per frame, each its own endzone track,
all placed at x ≈ −24 m because their endzone boxes are clipped by the
TOP of the frame (y1 = 0, 55–80 px tall, foot at the frame's top edge,
which maps to one ground line); 45 % of endzone boxes in frames 250–450
touch the top or bottom edge. x = −24 is 8–9 m behind the offence: the
officials, seen at the far top of the endzone view, drawn as default-
posed KC players. Candidate rules, not applied: a box touching the
BOTTOM edge has no foot point (drop from placement); a box touching the
TOP edge keeps its foot but has no height (no height gate; the geometric
signature for the officials question). Measured as an exclusion (one-view
id, never two-view, every box within 8 px of an edge): play 1 drops 23
ids (4.4 % of states, median 0/frame; 3 on the sideline at y ≈ +25 m, 20
late in the BAL end zone), play 2 drops 19 (10 % of states, median
3/frame, all 2–10 m behind the offence); no sure identity hit on either
play; every dropped id is TOP-clipped. Play 1's late end-zone group is
people beyond the end line (a seated row of photographers and an official
with a flag in the endzone frame's top 160 px), not players the sideline
lost: the sideline keeps 19–22 boxes through its zoom-out, 13 of the 19
project outside its frame and the 6 inside project onto empty end zone;
the one real end-zone player (BAL 32, a sure two-view id) is not dropped.
No time guard needed. The coordinator is adding the exclusion and fixing
build_timeline's views default (a pid with no views record was treated
as two-view at interpolated frames).
Two-view bodies now stand at their refit's translation (median shift
0.58 m from the box-bottom point); the tilt clamp is 60° for two-view
poses (measured p90 38°, none past 60), 35° for single-view. Play 2's
triangulation covers fewer players (23 % of joints pass); with `05f
--min-valid-joints 6 --min-frame-frac 0.5` all 30 refit, reprojection
13.4 / 12.2 px against 28 / 55 monocular, the head weakest (27 px
sideline: sparse joints), feet mixed (p10 −0.20, median +0.24 m) -- the
renderer drops bodies to the turf regardless. Renders `render_hifi_v5`:
play 1 from `poses_refit_tri.json`, play 2 from `poses_refit_tri6.json`
with `--team-by-colour`.

**Appearance decision (2026-09-04 evening, e37176c).** The hi-fi render
wears SYNTHETIC UNIFORMS by body region from identity's team
(render.uniform: helmet, jersey with sleeves to the elbow, forearm skin,
gloves, pants, socks, shoes; kits for KC red over white and BAL white
over black). Textures fitted from the footage cannot give a jersey at
140 px (one to three clean samples per vertex, turf bleed, speckle; after
de-mixing and smoothing the fit's gain over the plain median is ~0) and
flat team colours with a helmet read cleaner than any of them. The fit
stage (05i) is opt-in in the pipeline (FIT=1); the fitted textures stay
on disk for side-by-sides. Jersey numbers (c586cc2) go on ids whose
identity is sure -- a roster name and a (team, number) no other id of the
play claims (18/106 in play 1, 12/85 in play 2) -- as a surface decal:
ink texels of a 64-px raster pinned to jersey faces by barycentric
weights, 4 mm Gaussians lifted 8 mm off the posed surface (vertex
colours smeared: the torso lattice is 2.5 cm). The footage keeps the
field. Pipeline hifi flags: `--uniforms --numbers --helmets --follow
--eye-offset 2 -26 10 --fov 50 --field-texture`.

**Joints from geometry (2026-09-04 evening, 0036731).** `05m_triangulate_compare.py`
triangulates the regressor's 2-D joints (per-view caches; SMPLest-X's own
137 layout, 17 SMPL-X body joints mapped by NAME, and mirrored about the
box centre because the caches predate the runner's Y-up flip) through the
per-frame cameras and scores against the fused monocular lift on rulers
neither optimises. Play 1, 50 two-view players: the fused lift is 20–25 %
SHORT of anatomy on every bone (thigh 0.39 m, shank 0.41, ear-to-ankle
1.20 for a 1.85 roster median); triangulation has the metric scale (0.45,
0.50, 1.48) and is noisier per frame (bone variation 0.14 vs 0.10, ankles
0.32 vs 0.16 m off the turf). Next source for the refit: triangulated
joints with bone lengths held constant. `render.roster_shape` (dbcf29e)
already sets every avatar's stature to the roster height (the regressor
betas gave 1.71 m; roster 1.85). `08f_team_by_colour.py` (team from torso
saturation where bimodal: play 2 47/35 against identity's 62/23; play 1
refuses) feeds `05k --team-by-colour`.

**Timeline exclusions and seeds (2026-09-05).** `render.edge_rule`
(c47e073): a one-view id whose every box touches its frame's top or bottom
edge is left out (officials at the top of the endzone view, sideline
personnel, people beyond the end line; measured on both plays, no sure
identity lost). Interpolated frames now inherit an id's nearest recorded
views (a one-view id had passed as two-view there). Play 3's sideline
paint solve refuses: its candidates fit the paint (17 px) with a
25-degree lens 74 m out and players 2.78 m; a focal sweep along the paint
was built and FALSIFIED (player height is invariant along it, and the
paint costs 24 px per 25 % of focal). Remedy: `08 --seed-from <play-dir>`
(`SEED_FROM=` in the pipeline) seeds the joint solve with the sideline
mount of a solved play of the same game (plays 1 and 2 agree on (-4,
-100, 42) m, 12-15 degrees).

**Play 3 diagnosis (2026-09-05).** Holding the centre at the mount
(5e2a833) did not rescue it: the paint asks for a 21–53° lens from that
mount while the players' boxes (107 px, against 116 and 122 on plays 1
and 2) say 12°. The paint READER mislabels this clip; the gates refuse
correctly and nothing is exported. The correspondence diagnostic (drawn on
frames 394 and 622, `diag/play3_corr_*.png`) found the row-labelling trap
listed under "measured and rejected"; fixed in dd1b95c. Play 4 (`025_Sideline_KC_1-10_KC_49` +
`026_Endzone`, midfield) runs the whole chain with the mount held.

**Gates that lied by omission (2026-09-05).** `08` judged candidates only
when there were more than one, so a single held-mount candidate skipped
the three judges (play 3 went through at 2.81 m); fixed 55764d0. The
paint reader's row labelling in two-row frames called the far sideline
plus the far hash row the two sidelines (2.3x stretch, 41° lens) and grid
consistency cannot see a cross-field stretch; `assignment_is_possible`
now takes a lens band from the seed play (dd1b95c, `08 --seed-from` /
`--band-from`). The hash ruler is fooled by the midfield logo's white
paint (play 4 at the KC 49: hashes 1.82, numerals 0.99, LOS 48 vs 49,
field texture exact), so the pipeline's check passes a disagreement on
two of three witnesses, loudly (09757f0). Play 4 is the third play
through the chain, delivered 2026-09-05 06:30 (`diag/play_004_hifi_720.mp4`):
identity 44 of 82 named, 43 players triangulated at 7.7 px, refit 19 of
43 at the old 70 % frame threshold (the pipeline now uses 6 joints and
50 %), teams by colour refused (not bimodal), so identity's split stands
and leans KC. Any further play runs with `scratchpad/run_play.sh <dir>
<sideline.mp4> <endzone.mp4> <los-yards> --fresh --from-paint`, which
sets RED/WHITE and SEED_FROM=play_001. Play 3 under the band with the centre free gives the
right lenses (16–18°) and still fails every judge (rulers 0.78 vs 2.11 on
one candidate, players 2.55 m on the other, grids 61–95 px): its row
labelling is wrong beyond the two-row trap. Parked, refused honestly;
the next instrument there is the reader's per-frame labelling drawn on
the frames (`scratchpad/play3_correspondences.py`).

**Renderer (2026-09-04, 7688363).** Every body and the field rendered as
per-pixel salt-and-pepper (a flat red body: std 34/255, 1 % near-black
pixels), unchanged by splat size or opacity. The sparse splatter's
running sum of log(1 − α) spans every pair of the frame and reached
tens of millions, where float32 resolves to ~4; the per-pixel exclusive
difference was noise. Summed in float64 now; a dense uniform plane must
render flat (test). Every hi-fi render before this carried it. The
textures are also mesh-Laplacian smoothed (aca87c2: speckle from one to
three valid samples per vertex; roughness halves) and the fit's held-out
measure skips turf pixels like the loss (59d31e3).

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
both plays, so it was reverted); 25–34 default-posed states per play; identity's TEAM labels: 08c splits each
track's torso colour two ways (team_color.split_two_teams_balanced) and
votes the clusters onto the real teams by roster overlap, and on play 2
that disagreed with the per-id saturation gap on 22 of 52 ids (47/35 by
colour against identity's 62/23); the render takes teams from 08f where
the saturation is bimodal, but identity itself should adopt the gap split
where it passes -- a re-run of 08c re-keys the fused/refit caches, so it
waits for a pose-chain rebuild. `05k --follow` dollies the virtual camera with the play's smoothed
centroid (render.camera_path); judge on a clip before it goes in the
pipeline.

### Kits: the team from saturation votes, and what it exposed (2026-09-05)

The team label every play carried came from ONE crop per track and a global
two-means on mean HSV (`calibration/identity_precompute`). Against
crop-verified kits (`diag/sure_id_corrections.json`, sideline crops) it was
near random: a crosstab against the new rule is 25/31 vs 24/26 on play 1.
The "sure" identities (roster name from a number unique to one roster)
inherited it, so 10 of 18, 4 of 12 and 2 of 15 on plays 1, 2, 4 wore the
other team's kit and had that team's number painted on them.

**Rule D** (`identity/team_color.split_by_saturation_votes`, side agent's
measured winner): per CAMERA a 1-D two-means on every detection's torso
SATURATION; each detection votes; a track's label is the majority. The
higher-saturation cluster is the coloured kit in both cameras (centres
44/141 sideline, 55/150 endzone on every play so far), so one label means
one kit in both views without a global fit. Per detection, the label is
2.3 % wrong at margin 0.4 (`|S - mid| / (hi - lo)`) over 83 % of the boxes
(sideline 2.1 %, endzone 4.6 %).

**The kit decides the roster** (`08c --saturated KC`, pipeline `RED`): the
number is looked up on the kit's roster; a number unique to the OTHER roster
no longer names the player (the OCR is 75 % per track and a wrong name
paints a wrong number). From cache on plays 1, 2, 4: 31 / 27 / 38 named
of 106 / 85 / 82, the kit overruling the number on 4 / 2 / 7 ids, mostly
one-digit reads (#0, #1, #3). The roster vote (numbers unique to a roster)
"disagrees" on the white cluster on every play -- it rests on 0-2 reads, and
is printed as a check only.

**What it exposed.** Of the ids seen by both cameras, 24 %, 53 % and 30 %
(plays 1, 2, 4) wear DIFFERENT kits in the two views: the per-frame pairing
in `08b` joined different players (play 2 id 7: BAL #44 white on the
sideline, a KC red jersey in the endzone, `diag/kit_play_002_id7.png`), and
some sideline tracks switch players mid-track (play 1 id 5,
`diag/kit_play_001_id5.png`). Every later stage inherited it: triangulation
across two people (why only 23-28 % of joints pass the gates on plays 2 and
4), names on the wrong body, 5-7 pieces per player.

**The fix in 08b** (`tracking/kits.py`, commit d74f3b7): each box carries a
signed saturation margin; `fuse_frame` refuses a cross-kit pair; the fused
points carry the kit into `link3d` as labels (which it already gates on: a
track votes its label, a detection never joins a known mismatch);
`tracks.parquet` gains `kit_margin` and `kit`, and identity's `team_votes`
reads them. `--no-kit-link` keeps the pairing gate only. NOT yet run end to
end (the GPU was on play 5): the first run is play 2 against its current
`tracks.parquet` -- ids disagreeing across cameras, track count,
triangulated joints passing, names. The earlier "labels in the linker are
worse" dead end used the balanced HSV split that carried no information;
`07j` has a fourth variant (`kits`, margin 0.4) re-measuring it on the
helmet set.

**Play 5** (047/048, BAL 30): the paint judge's grid limit was missed by 2
px (27 vs 25) with both rulers agreeing at 0.95 and players 1.72 m, so
`08 --max-grid-px` (pipeline `GRID_PX`) widens it for one play, printed
with the verdict; the check stage then passed on two of three (numerals
0.983, LOS ok, hashes 1.043) and the footage field landed on the drawn one
(logo on the 50, faint yard lines on the drawn lines).

### Play 1 v8 (2026-09-05 19:32): kits from the saturation vote

`diag/play_001_hifi_720.mp4`, 255 frames. Identity from cache with the kit
deciding (31 of 106 named, the OCR overruled on 4), keypoints and
triangulation again (47 players of 83 with keypoints, 42 % valid joints at
8.3 px), refit 47 players over 443 frames at 0.091 m rms, 08f refused
(gap 10), so identity's teams stand. Frame `diag/play_001_v8_a.jpg`: the
red cluster sits on the ball, white spread; numbers on the sure ids;
officials still white; one-view duplicates remain (the endzone track).

### One roster player, one avatar at a time (2026-09-08)

The identity cache on v16 carried four names on two overlapping ids each:
"Harrison Butker" (#7 -- the kicker, on a 2nd-and-20) on ids 16 and 17
for 616 frames, Isiah Pacheco (#10) on 5 and 37 for 278, Marquise Brown
(#5) on 53 and 89, Nelson Agholor on 79 and 82. 08c names every id on its
own from its jersey vote, so two tracks that both read "7" are both the
kicker, and the render draws him twice. The vote counts behind a winner
were only logged (`vote_jersey_numbers`), so nothing could rank the two
claims. Now: `jersey_votes_win` / `jersey_votes_total` per row (future
runs), `scripts/08j_jersey_votes.py` for a play whose cache predates them
(the OCR re-voted on the current global ids -- no re-pairing, the pose
caches keyed by id stay valid -- into `jersey_votes.json`), and
`identity.exclusive.exclusive_names` in 08c after `merged` is built: among
ids claiming the same (team, jersey) with overlapping spans the strongest
vote keeps the name, the others lose it (jersey 0, `P<id>`, kit kept); a
tie within 1.5x names neither, since a wrong name is the worse defect.
Play 1 (08j on the CPU, 5 min: 32 (cam, id) keys read a number, the
re-vote agrees with the cached winner on 30): four ids lost a shared
name -- 37 (Pacheco, kept 5), 17 (Butker, kept 16), 89 (Brown, kept 53),
79 (Agholor, kept 82); 22 named. The kept "Butker" is still the kicker on
a 2nd-and-20: `specialist_veto` (K, P, LS unnamed unless `--kicking-play`)
takes that one too.

### The ruler was blind, and the gate preferred it (2026-09-12)

Every placement ruler used so far scores a body against the keypoints of the camera it was fitted
to. That is blind by construction: depth along the camera ray and rotation about it are exactly what
one view cannot constrain, so the number cannot rise however wrong the 3-D is. `05t`
(`scripts/05t_cross_view_error.py`) scores the same cached bodies in the OTHER camera and is now the
standing ruler for placement.

    v34 cache, the same 1812 body-frames both cameras see     sideline (fitted)   endzone
    rms p50                                                          7.6 px        8.7 px
    rms p90                                                         14.3          25.8
    rms p99                                                          --          341.5
    over  20 px                                                       6 (0.3 %)   197 (10.9 %)
    over 100 px                                                       0           129 ( 7.1 %)

Across all 5099 sideline body-frames the cache reads p50 3.4 px, p90 11.0, 0.3 % over 20 px. By the
fitted camera's ruler play 1 is finished; by the other camera 7 % of it is over 100 px wrong. **The
sideline footage overlay (05q) shares this blindness** -- red skeletons sit on green keypoints while
the body stands a metre downrange -- so it cannot show what a free-viewpoint render will.

**The worst player, and a second ruler in metres.** id 19: 4.0 px sideline, 142.2 px endzone (p90
339.5) over 107 paired frames, his hips 149 px sideways from his own endzone box and 16 px down.
Sideways in the endzone is depth along the sideline ray, so the defect was measured again in metres
against where the two cameras' ankle rays actually cross:

    id (ankle-ray miss p50)   gap from the triangulated feet   along the ray   across it
    id 30 (0.03 m)                      0.01-0.03 m               -0.01 m       0.02 m
    id  5 (0.17-0.22 m)                 0.01-0.03 m               +0.02 m       0.00 m
    id 19 (0.33-0.38 m)                 0.89-1.10 m               +0.89..+1.10  0.02 m

Nine sampled frames of id 19, every one a metre out along the ray and 2 cm across it. A pure depth
slide is the signature of a body placed by one camera.

**That decomposition is also an identity test, and it clears the pairing.** Splitting every paired
body's gap from the triangulated crossing into along-ray and across-ray over the whole cache -- 23
players with 15+ paired frames -- gives |along| p50 **0.01 m** and |across| p50 **0.02 m**. Not one
player exceeds 0.25 m ACROSS the ray (the worst is id 6 at 0.16 m), which is what a wrong-man pairing
would look like, so no mis-pairing survives among the paired ids. Exactly one player exceeds 0.5 m
along it: id 19 at 1.04 m, the next worst being id 17 at 0.25 m. The entire metric placement defect
of the paired population is one man -- which is why the endzone p99 of 341 px reads like carnage: it
is his 107 frames.

**And the blind spot this ruler cannot reach.** Only 1832 of the cache's 5103 body-frames have a
second view on that frame. 3271 (64 %) are sideline-only, and 1165 of those belong to ids the endzone
never sees at all. Their depth is not unconstrained -- an ankle ray meets the turf at exactly one
point, so one camera plus the ground plane does fix a position -- it carries the error of the
feet-on-ground assumption, and that error is **smaller than it looks**: measured against the
triangulated feet on the 1719 frames where truth exists, the ankle-turf depth is out by **p50 0.19 m,
p90 0.56 m**. (An earlier line here said ~0.4 m; that was id 5's pre-snap figure, one player, and the
population number is half it.) The depth snap reached 610 of the 3271 sideline-only body-frames
(moving them p50 0.25 m, p90 0.95, max 2.46 against its 2.5 m cap), leaving 2661 (52 % of the cache)
on the turf estimate alone -- which is decent, not dire.

**A ruler for those frames was designed, calibrated and thrown away the same hour.** A player's
standing height is fixed by his build (the 51 ids span 1.75-1.96 m, p50 1.87), so his box height in
pixels measures his range independently of the ankle ray, and two disagreeing estimates would localise
a bad placement with no second camera at all. It does not work, and the numbers say why:

    1719 paired frames, error against the triangulated feet   p50      p90
    ankle ray meets the turf (what the fit uses)              0.19 m   0.56 m
    box height against known standing height                  6.73 m  59.44 m
    the same, upright boxes only (857 frames)                  3.27 m  10.08 m

and the disagreement between the two carries no information: bucketing by it, from under 0.5 m to
over 5 m, |ankle - truth| stays flat at 0.13-0.21 m p50. The cause is geometry, not implementation:
at the median range of **112 m**, depth from apparent size inherits the box's RELATIVE error, so ten
pixels of slop on a 140 px box is eight metres of depth. Intersecting a ray with the turf is far
better conditioned. Do not re-propose depth-from-apparent-size at All-22 ranges; it is unusable
whatever the pose gating.

**So why does the depth snap reach so few of the blind bodies?** Its own docstring measures it firing
on 69 % of frames, but that was the PAIRED population; on the bodies that actually need it, it fires
on 19 %. Counting the refusals by reason:

    reason                                          paired (1907)   blind (3196)
    SNAPPED                                          1260 (66 %)     614 (19 %)
    no endzone bodies that frame at all                  0          1278 (40 %)
    no endzone body within LATERAL_M 0.6 m of the ray   112           492 (15 %)
    a body is on the ray but beyond MAX_MOVE_M 2.5 m     86           433 (14 %)
    two candidates within MARGIN_M: refused            449 (24 %)    379 (12 %)
    the team gate left no candidate                      0             0

A suspicion of mine died here: `snap_ground` filters candidates to `teams.get(q) == side`, which
drops a candidate whose team is UNKNOWN whenever the sideline body's team is known (the docstring
only promises leniency for an unknown sideline id). It costs nothing on play 1 -- all 90 endzone ids
carry a team label, 0 candidate-slots were dropped for an unknown team, and that bucket is empty in
both populations.

The largest bucket is not a gate. **1278 of the 3196 blind body-frames (40 %) have no endzone ground
points at that instant**, because the endzone camera is solved on 510 frames (sideline 140-649) while
the play is rendered over roughly 14-660: before frame ~140 there is no second view by construction.
That is calibration coverage, not a threshold, and it is the ceiling on everything two-view --
including 05t's own reach. Nor is loosening the gates attractive: the blind bodies' nearest candidate
sits p50 0.28 m and p90 1.44 m off the ray against 0.16 m and 0.51 m for paired ones, so the endzone
track frequently does not hold that man at all (the same fragmentation as the census problem) and a
wider `LATERAL_M` would mostly buy wrong matches. With the turf estimate already at 0.19 m p50 the
snap's remaining upside is small. The two levers that are left are extending the endzone camera solve
and fixing fragmentation -- not tuning this rule.

**And "fragmentation" turns out to be the wrong word for the census problem.** The v34 entry below
reads the 32 Kansas City and 51 Baltimore sideline ids as a player being lost and re-acquired under a
new id, which would be repaired by stitching the pieces back together. It is not that. Pairing every
id with its most plausible continuation (a later id starting within 90 frames of its end, close enough
that the join needs no more than 9 m/s) and chaining greedily leaves **KC 32 -> 28 chains and BAL
51 -> 40**: only 4 and 11 ids respectively have any successor at all. The ids are not sequential
pieces of one man. Many of them start at frame 14 and run for hundreds of frames side by side
(3[14-636], 5[14-628], 11[14-660], 12[14-638]), and where they coexist they stand in **different
places**: 11.2 concurrent KC ids occupy 10.4 distinct positions at 0.5 m single linkage, 14.2 BAL ids
occupy 13.8. They are separate people, so neither stitching nor the twin merge (08o, which only folds
ids sharing one body) has much left to take.

Two more things the count exposes. `entity_type` cannot be used to exclude non-players: it types only
**6 of 83** sideline ids as `player` and the other 77 as `other`. And the raw tracked population is
wildly phase-dependent -- at the snap 21 ids in 20 places, every one on the field (|y| <= 9.5 m),
against 36-37 ids in 30-36 places by frames 580-620 with **12-14 of them beyond the sideline**
(|y| > 24.4 m) and x spreading from -43.8 to -10.8. That late crowd is staff and players arriving after
the play is over. The drawing rules filter most of it (the census counts timeline states, not raw
tracks), but it means any census averaged from the snap to the end of the clip is scoring the aftermath
as much as the play, and the phase split already in the v34 entry (a deficit at the snap, a surplus
during and after the run) is the real signal.

The snap itself is the sharpest number: **21 bodies for 22 players, split KC 9 / BAL 12** by the kit
vote, in 20 distinct places. So two Kansas City players are never tracked at all at the snap, and one
Baltimore pair is a duplicate -- 9 + 11 real men + 1 duplicate = 21.

**The kit vote is not the problem, and I nearly recorded that it was.** Wanting a team ruler
independent of colour, I checked every body clear of the line against which side of the line of
scrimmage it stands on, and got **13 disagreements out of 13** -- which I first read as a labelling
catastrophe. Thirteen of thirteen is not a defect, it is an inverted convention: I had assumed the
offence stands at `sign * (x - LOS) < 0`, and play 1 has every KC body at +1.6 to +7.1 m and every BAL
body at -2.1 to -10.4 m. Read the right way round the two rulers agree on **13 of 13 bodies with no
exceptions**, deep safeties (BAL ids 2 and 30, ten metres off the line) and the KC back seven metres
into his own backfield included. The kit vote from torso saturation is sound at the snap; so is
`line_of_scrimmage`. When a check fails on literally every sample, suspect the check.

**Then the footage settled what the geometry could not.** I had a promising theory that the ball-following
All-22 sideline crop simply does not frame the whole field, which would make "eleven a side" unreachable
and part of the census error not a defect at all. The visible ground footprint is real enough -- at the
line of scrimmage the sideline frames y = -12 to +24 m of a +-24.4 m field and the endzone only -6 to +7,
about 25 % and 21 % of the ground around the ball -- but the theory is **wrong for the snap**. Looking at
sideline frame 300 itself: nothing is clipped at either edge (the leftmost body sits at x 250 px, the
Kansas City receiver split widest at x 1765), and a human count gives 11 Baltimore in white against 9-10
Kansas City in red. The 21 tracked boxes match what is actually visible. Kansas City's missing men are not
out of frame and are not missed by the detector -- **the sideline views the line of scrimmage nearly
end-on, so the offensive line overlaps into one mass**. Endzone frame 285 shows exactly those men
separately (76, 62, 65, 74, 83, with the passer 15 and the back 10 behind them).

**So the snap deficit is a cross-camera bookkeeping failure.** At frame 300 the sideline holds 21 ids and
the endzone 24, and only **12 of them share an id**: 12 endzone-only, 9 sideline-only, a union of KC 14 and
BAL 19 against a truth of 11 and 11, while the v34 cache draws 19 bodies (KC 8, BAL 11). The five
endzone-only Kansas City ids (22, 37, 38, 74, 86) stand at x -22.7 to -24.2, on the line: the occluded
linemen, tracked perfectly well by the camera that can see them and never joined to the sideline. Admitting
them wholesale would overshoot badly, because some endzone-only ids are the same men as sideline ids that
were never paired -- which of the two they are is the number that picks the fix.

**Measured, by asking each endzone-only id how far the nearest sideline body is** (frames 300-400):

    of the 12-14 endzone-only ids per frame
      5-7  stand 0.14-0.70 m from a sideline body: the SAME MAN under two ids -- a pairing miss
      2-4  have no sideline body within 1.5 m: genuinely unseen there (the occluded linemen, and officials)
      rest ambiguous at 0.7-1.5 m
    both cameras pooled, 45 ids fall in 26-28 distinct places: KC 11-12, BAL 15-18 (truth 11 and 11)

Kansas City's pooled count is **right**. The two cameras together do see all eleven; the snap deficit is
pure bookkeeping, and the repair is to join the ids rather than to admit more bodies. Ground-point distance
alone cannot do that joining -- 0.7 m folded two real men once before (the passer and the lineman beside him
at 0.29 m) -- and the obvious alternative, the cameras' ankle RAYS, **does not work per frame either**.
Tested over 137 (frame, endzone id) pairs at frames 300-420: only **10** gave a confident join (best miss
<= 0.30 m and at least 0.30 m clear of the runner-up), **113 were close but ambiguous** and 14 had no
sideline body agreeing at all. The runner-up typically sits 0.03-0.05 m behind the best, because at ~100 m
the rays to a neighbour standing a metre away miss by almost as little as the rays to the right man. The
"0.20 m against 1.11 m" separation quoted for twins was a grossly mis-paired case, not adjacent players,
and it does not generalise to picking one man out of a formation.

The signal is real but thin: where a pair is already correct (ids 2, 3, 5, 9, 12, 15, 30) the true partner
does come out best. So the form that can work is not a per-frame threshold but a **one-to-one assignment
per frame accumulated over the whole play** -- the ambiguity is mostly two endzone ids competing for one
sideline id, which a global assignment resolves, and per-frame noise averages out over hundreds of frames.

Tried, as a Hungarian assignment on the ray-miss matrix over 124 frames. The first run gave 9 of 31 ids a
partner winning 60 % of their frames and several plainly wrong winners, including cross-team ones -- but
two of those failures were the measurement's, not the method's: nothing forbade a cross-team assignment,
and the "share" divided by every frame the endzone id appears in, including frames where its true partner
was not detected at all, which penalises exactly the intermittent tracks this is meant to repair. With a
team gate and the share taken over co-present frames, **14 of 28** ids have a partner winning at least
60 %, and seven are joins the pipeline does not currently make:

    89 -> 7 (82 %, 89 frames)   5 -> 19 (74 %, 50)   38 -> 17 (71 %, 34)   45 -> 4 (71 %, 31)
    101 -> 5 (71 %, 34)         22 -> 80 (69 %, 29)  17 -> 25 (61 %, 33)

all same-team, and two of them (45->4 and 89->7) were independently flagged as the same man at 0.33 m by
the ground-point test above -- two instruments agreeing. **Not adopted, and checking each one against the pairing it
would replace is why:**

    endzone -> sideline   ray miss proposed/current   ground gap proposed/current   verdict
      5 -> 19                 0.33 / 0.19 m               6.44 / 0.48 m             WRONG, six metres apart
     38 -> 17                 0.30 / 0.10                 1.27 / 0.54               wrong
     17 -> 25                 0.14 / 0.18                 0.96 / 0.71               ambiguous, rulers disagree
     22 -> 80                 0.17 / 0.22                 0.71 / 0.45               ambiguous, rulers disagree
     45 ->  4                 0.11 / 0.90 (1 frame)       0.59 / 1.17               good
     89 ->  7                 0.08 (179 frames) / --      0.36 / --                 good, no current pairing
    101 ->  5                 0.25 (71) / --              0.89 / --                 plausible, no current pairing

The suspicion about `5 -> 19` was right and worse than expected: those two bodies stand **6.44 m apart**,
so a 74 %-confidence vote would have welded together two men at opposite ends of the formation. The
Hungarian one-to-one competition manufactures that -- when the true partner is undetected on a frame, the
assignment must still give the id to somebody. Two of seven candidates are flatly wrong and two more have
the two rulers disagreeing, which is far too high a false-positive rate to apply automatically; a wrong
merge costs more than a merge not made, as the 0.35 m twin threshold proved when it folded a passer into
the lineman beside him. Three survive an independent ruler (45->4, 89->7 at 0.08 m over 179 frames, and
101->5), and 89->7 in particular is a man the two cameras have never had under one id. The route forward
is the vote as a CANDIDATE GENERATOR with the ground-gap check as the gate, not the vote as a decision.

`scripts/08r_pair_by_rays.py` is that, and it proposes only -- on play 1, 254 frames, 13 candidates clear
the vote and the gate throws out seven, including `85 -> 84` at **14.24 m** (the referee, at 91 % vote
confidence) and `5 -> 19` at 6.44 m. But the six survivors are not all applyable, because a union may not
put two tracks of one camera under one id at the same time:

    97 -> 11   CLEAN addition (endzone 11 has no track at all)      102 -> 0   CLEAN addition
    89 ->  7   81 endzone frames collide                             45 -> 4   53 endzone + 14 sideline
    101 -> 5   17 endzone frames collide                             27 -> 14  41 sideline frames collide

**Four of six are re-pairings, not additions.** Sideline 7 already carries an endzone track under id 7 on
81 of the same frames, so joining endzone 89 to it means unpairing endzone 7 first -- a different and
riskier operation. The evidence does favour 89 (the existing (7,7) pairing has *zero* frames with
confident ankles in both cameras, while (89,7) has 184 at 0.08 m), but that is a decision to state
explicitly rather than smuggle inside a merge, so 08r proposes clean additions by default and needs
`--allow-repairing` for the rest. Applying any of them must route through
`pair_by_appearance.global_ids_checked`, which refuses the union outright.

**And then the two clean additions turned out not to be worth applying, which is the most useful thing
this thread produced.** Checking what `97 -> 11` would buy: over its 30 frames the one-view cache stands
p50 0.34 m but p90 8.38 m and max 10.27 m along the ray from the two-camera point, with 0.01 m across it.
That looks like a body wandering ten metres in depth. It is not. The 12 frames with over a metre of error
have a ray miss of **p50 1.65 m**, against **0.06 m** for the other 18 -- they are the frames where endzone
97 is not sideline 11's man at all. At frames 360-380 the cached body stands still at (-22.9, -4.5) while
the "two-camera point" walks from (-24.4, +2.5) to (-25.0, +5.5): two different people. Where the rays do
agree (468-510, 628-638, miss 0.02-0.14 m) the one-view placement is **already** 0.04-0.36 m accurate.

So the join buys almost nothing, and the 67 % vote share means something worse than "probably right": the
pairing is **time-localised**, valid over some stretches of the track and wrong over others. A whole-id
merge is therefore the wrong granularity -- it would weld the correct frames together and actively corrupt
the rest by fitting one man's body against another man's keypoints, which is exactly the failure the
two-view work spent this session repairing. Any future version of this has to join per interval, with the
ray miss deciding frame by frame where the join holds, and 08r's `share` must not be read as confidence in
a whole-track identity. The honest state: the cross-camera bookkeeping is real (12 shared ids of 21 and 24
at the snap) but neither ankle rays nor an accumulated assignment can fix it at track granularity, and the
placement upside for the bodies it would join is a fifth of a metre.

**Then the generator got better, and two of its own rules were wrong.** 08r now builds candidates from ONE
global assignment on per-pair MEDIANS -- the median ray miss over every frame two ids share -- instead of
accumulating per-frame votes. A single frame cannot separate neighbours, and votes fail in a way a summary
hides: when the true partner is undetected the assignment must still give the id to somebody, which is how
a referee got proposed onto a player 14.24 m away at 91 % confidence. The median version is validated by
what it leaves alone -- of 25 pairs it keeps **16** exactly as the pipeline has them, including every
pairing independently confirmed good -- and it corroborates six candidates that two other methods found
separately. It still cannot be trusted as a decision, because a complete matching displaces a chain into
worse slots (22 -> 38 at 0.45 m against its current 0.22), so the gate remains what decides.

The second wrong rule was mine: **a relabel is not a union.** A global id spans both cameras, so joining
endzone 19 to sideline 11 can either relabel the endzone track onto 11 -- leaving sideline 19's man, a
different person, alone -- or union ids 11 and 19, which sweeps him in. Only a collision in the OTHER
camera means a track must be given up; an own-camera collision merely dictates that the join is a relabel.
Classifying on the own-camera overlap had rejected `sideline 11 <- endzone 19` (0.14 m over 150 frames) for
"370 sideline frames collide" when its endzone collision was zero. Four candidates now survive on play 1:

    0 <- 102  (0.13 m rays, 0.44 m turf,  35 frames, union)    11 <- 19  (0.14, 0.38, 150, relabel)
   28 <-  21  (0.08 m rays, 0.44 m turf, 138 frames, relabel)  80 <- 22  (0.17, 0.64,  59, union)

held back as re-pairings: `4 <- 45`, `7 <- 89`, `33 <- 37`, `37 <- 139`. The one that hurts is **`7 <- 89`,
0.08 m over 184 frames** -- the strongest pairing evidence in the play -- because taking it means giving up
endzone 7, whose pairing with sideline 7 has *zero* frames with confident ankles in both cameras. That is
very likely right and still needs a deliberate decision. Nothing is merged: applying is a separate step and
its ruler is 05t plus the along-ray gap, never the sideline's own pixels.

### v36: the four pairings applied, and they pay in metres (2026-09-12)

08s relabels an endzone track onto its real sideline owner -- it never unions the ids, because an id
spans both cameras and unioning 11 with 19 would sweep in sideline 19, a different man present on 370 of
the same frames. Applied to play 1's four accepted proposals: **1453 track rows and 22457 keypoint rows**
moved, three ids that had no second view now have one, and 05p refitted (two-view coverage 24 -> 25
players, 1793 -> 2047 frames, 1851 anchored on the triangulated ankles).

    05u, the same 1900 body-frames on 24 players     v35        v36
    id 80                                          0.26 m     0.02 m
    id  0                                          0.23       0.02
    id 11                                          0.12       0.01
    id 28                                          0.07       0.01
    every other player                             identical to the centimetre
    pooled |along-ray| p90 / p99                0.21 / 0.65   0.05 / 0.41
    players worse by more than 0.10 m                         0 of 24

    05t, the same 2061 paired body-frames            v35        v36
    endzone p50 / p90 / p99                    9.7/45.8/163.5  8.7/15.5/80.5
    endzone frames over 20 px                   478 (23.2 %)   78 ( 3.8 %)
    endzone frames over 30 px                   371 (18.0 %)   41 ( 2.0 %)
    sideline p50 / p90                            3.4 / 12.7   4.0 / 12.5

**Read the v35 column carefully.** It is v35's BODIES scored against the CORRECTED pairing, which is why
its endzone numbers are worse here than when v35 was scored against its own wrong pairing (p90 16.7
then, 45.8 now). Both columns share one truth, which is the only fair comparison, but it means the
headline 45.8 -> 15.5 partly measures the pairing fix rather than the fit. The metres are the cleaner
statement: four players improve by 0.06-0.24 m and nobody moves the wrong way.

**The census did not move: 3.02 bodies a frame, 3 frames of 361 exactly eleven and eleven.** Placement
and population are independent axes and this was a placement change. Also worth noting what the ruler
cannot show: id 19, v35's worst body, had a wrong endzone partner (the two cameras' own ground points
for "him" 1.31 m apart); relabelling that track onto its real owner leaves him one-view and therefore
absent from 05u's table rather than visibly better. Removing a wrong pairing is right, but count the
players in each column before reading the percentiles.

Still held back: **`7 <- 89` at 0.08 m over 184 frames**, the strongest pairing evidence in the play,
because taking it means giving up endzone 7 -- whose pairing with sideline 7 has *zero* frames with
confident ankles in both cameras. It needs `--allow-repairing` and a deliberate decision.

### ground_positions keyed by the TRACKER's id, not the player's (2026-09-12)

`ankle_ground` keys `(cam, frame, global_player_id)`. `ground_positions` looked those ankles up by
`r.track_id` and emitted its own output under `r.track_id` too, while **every caller asks by player id**
-- 05p tests `pid not in ground.get(f, {})` with `pid` from the keypoints, the depth snap matches endzone
candidates the same way, the census counts players, and the docstring says `{pid: xy}`. It never showed
because this pipeline had never assigned a global id different from a track id.

08s is the first thing that does, and all four relabelled ids reported **zero endzone ground frames**
with their rows and keypoints perfectly intact (id 11: 423 endzone rows, 405 keypoint boxes, 0 ground
frames, against control id 12 at 381/377/381). Their points were landing under the OLD track ids and the
ankle lookup missed every time, falling back silently to the box bottom.

What it did and did not break, measured rather than assumed. The **fits are unaffected**: 05p uses the
sideline ground dict and the relabel only touched endzone rows, so the ids still coincide there, which is
why 05u showed v36 as a clean win. What was blind is the depth snap's endzone candidates for those ids
and **08r's turf ruler** -- the missing turf statistic is why the "beat the incumbent on both rulers"
check silently skipped and proposed `11 <- 97` (0.19 m over 30 frames) against an incumbent measuring
0.14 m over 150. After the fix that candidate disappears from the assignment entirely (23 pairs, 18
unchanged, against 20 and 14 before).

Fixed to key by `global_player_id`, and a table without that column is now refused with a `SetupError`
naming it rather than falling back to `track_id` -- silence is how this hid. Confirmed on the play: the
four ids now report 75, 423, 464 and 491 endzone ground frames, the stale keys 19/21/22/102 are empty,
and the controls are untouched. Two existing tests failed on the change and **both were fixture
artifacts, not a contradicting contract** -- `test_ankle_ground` and `test_timeline` built rows carrying
only `track_id`, so they hit an AttributeError; neither asserted the old key, and both assertions stand
once the fixture carries the column the real table always has. Suite 1112 passed.

### v37: two re-pairings, and the census finally moves (2026-09-12)

`08s --give-up-incumbent` applied the two the gate passed, evicting each sitting endzone track to a
**fresh unused id rather than deleting it** (endzone 7 -> id 145, endzone 25 -> id 146; 648 track rows and
10676 keypoint rows moved). sideline 7 went from 95 to 467 endzone frames, sideline 25 from 172 to 181.

    05u, 2093 identical body-frames, 25 players    v36        v37
    id 25                                        0.69 m     0.02 m
    id  7                                        0.23       0.00
    every other player                           identical to the centimetre
    pooled |along-ray| p90                       0.22       0.05
    players worse by more than 0.10 m                       0 of 25

    05t per player (the same keypoints)      sideline p50/p90     endzone p50/p90
    id 7  v36                                    1.8 /  2.3        27.1 /  38.6
    id 7  v37                                    5.0 /  7.3         6.3 /   7.9
    id 25 v36                                   17.0 / 32.3        98.4 / 386.2
    id 25 v37                                    8.9 / 13.2        15.5 / 483.4

id 7 is the textbook case: near-perfect in the camera it was fitted to and 27 px in the other -- the
blind-camera signature -- replaced by 5-6 px in **both**, which is what a correct pairing looks like.

**The census moved for the first time this session: 3.02 -> 2.94 bodies a frame, and frames exactly
eleven-and-eleven 3 -> 9 of 361.** The mechanism is in the log rather than inferred: "frames beyond an
id's sideline span left out" fell **2110 -> 1412**, because a relabelled endzone track now sits under an
id with a long sideline span instead of being cut as out-of-span. Correct pairing buys population, not
just placement.

**The residual, named.** id 25 keeps the tail: its endzone p50 improves six-fold but p90 goes 386 -> 483
px, and it owns the pooled p99 rise (313.7 -> 388.5) while the metres stay flat (p99 1.01 -> 1.03, max
4.49 unchanged). That is time-localisation inside the new pairing -- right for most frames, badly wrong
on a minority -- and endzone 33 spans 378-562 against sideline 25's 381-634. The join should be per
interval, exactly as the earlier finding said.

**And time-localisation turns out NOT to be general, which makes that job small.** Scanning every
pairing's ray miss per 60-frame block (16 ids with 40+ paired ankle frames):

    id  frames  overall p50   best block   worst block   spread   over 0.6 m
    25      46        0.17         0.11          1.90     1.79       37 %     <- re-paired in v37
     7     184        0.08         0.03          1.54     1.51        7 %     <- re-paired in v37
    the other 14      0.04-0.19       --            --    0.02-0.10   0-2 %

Only **2 of 16** vary by more than 0.25 m between blocks, and they are precisely the two tracks v37
re-paired by eviction. Every long-standing pairing is flat. So the fix is not general interval support
in 08r/08s: it is trimming a newly applied relabel to the span where its rays actually agree, which is
a much smaller change than the earlier finding implied. The mechanism is plain enough -- an evicted
incumbent's replacement is chosen on the frames where both ids have confident ankles, and applied to the
whole span, so the edges of that span are where it goes wrong.

### v38: every join trimmed to the frames its rays agree, and it wins on all three rulers (2026-09-12)

08r now emits `frame_from`/`frame_to` -- the longest unbroken run whose ray miss is within
`--max-ray-miss` -- and 08s converts that by the clip offset and relabels only inside it, refusing a
proposal that carries no interval. v38 replays every join from the pre-pair backups rather than trimming
in place: six applied (0<-102, 7<-89, 11<-19, 25<-33, 28<-21, 80<-22), one eviction confined to its
interval (endzone 25 over 385-471 only, 51 track rows). Two-view coverage 25 -> **27 players**.

    05u, 2155 identical body-frames, 27 players    v37        v38
    id 25                                        0.23 m     0.04 m
    id 22                                        0.20       0.01
    every other player                           identical to the centimetre
    pooled |along-ray| p90 / p99              0.06 / 0.83   0.04 / 0.39
    players worse by more than 0.10 m                       0 of 27

    05t, 2299 paired body-frames                   v37        v38
    endzone p50 / p90 / p99               8.5/16.5/135.5   8.4/15.3/60.1
    endzone frames over  20 px              134 ( 5.8 %)    71 ( 3.1 %)
    endzone frames over  30 px               98 ( 4.3 %)    33 ( 1.4 %)
    endzone frames over 100 px               36 ( 1.6 %)    13 ( 0.6 %)
    sideline p50 / p90                        4.2 / 12.1    4.3 / 12.4

    census                                        2.94        2.84   (exactly 11 and 11: 9 -> 10)

The trim hits exactly what it was aimed at -- the tail. id 25's endzone p90 of 483 px is gone, and the
pooled p99 halves on both rulers while the sideline stays flat and no player regresses.

**And one result nobody proposed: id 22 improved from 0.20 m to 0.01 m without being a candidate.**
Endzone track 22 was relabelled onto sideline 80 only over frames 363-511; outside that window it keeps
its own id, and sideline 22 holds it there at 0.01 m. That single endzone track was covering **two
different men at different times**, and splitting it by interval gave each his own -- a failure whole-track
joining cannot even express, let alone repair.

**Scanned for more of them, and there are none.** Asking which sideline id wins each 60-frame block of
every endzone track flagged **19 of 25** -- which is not a finding, it is the same mistake as the
per-frame ray test, made again: no margin requirement, on a measurement where a neighbour at ~100 m
misses almost as little as the right man (10 confident joins out of 137 per-frame tests). Requiring the
block winner to beat its runner-up by 0.15 m and each claimant to hold two or more blocks leaves **1 of
19**, and that one does not survive inspection either: endzone 74 is claimed by sideline 74 over frames
440-559 at 0.08 m rays / 0.67 m turf, which is simply its own correct pairing, and by sideline 3 over
140-379 at 0.17 m rays but **1.42 m turf**, failing the 1.0 m cap outright. So the id 22 case was the
only real one, the interval trim already repaired it, and no further machinery is warranted.

**id 17, the worst remaining player (0.24 m along, 0.13 m across), is not a pairing problem either.**
Searched against every endzone track: the one in use is the best on both rulers (0.18 m rays, 0.71 m
turf), and every alternative is worse (0.25/0.70, 0.30/1.27, 0.80/1.93, 0.83/2.80, 1.19/6.11). A clean
negative -- stop looking for a partner there.

**So placement is done to diminishing returns and the census is the binding problem.** At pooled
|along-ray| p90 0.04 m and endzone p99 60 px, the remaining error is the body count at 2.84 a frame. The
measured cause at the snap is 21 ids for 22 players, KC 9 and BAL 12, because the sideline views the line
end-on and merges the offensive line into one mass -- while the endzone sees those men individually and
`endzone_only_ids` discards them as ghosts. The targeted fix looked like keeping an endzone-only id when NO
sideline body stands within ~1.5 m of it (2-4 such ids a frame, measured), which is the same "is the
sideline drawing that man" test `beyond_sideline_span` already carries.

**Measured before building it, and it is dead.** Counting endzone bodies with no sideline body within
1.5 m, by phase:

    snap (300-340)       3.5 a frame     KC 0.2     BAL 3.4
    run  (340-460)       3.0             KC 0.3     BAL 2.8
    after the whistle    5.0             KC 2.1     BAL 3.0
    late (560-660)      16.3             KC 9.9     BAL 6.4

At the snap they are almost entirely Baltimore and essentially **zero Kansas City** -- and KC is the team
short by two, while BAL is already over at twelve. The rule would add bodies to the wrong team. The ids
most often "alone" are 85 and 87 at 510 frames each, which are the officials the behind-the-offence rule
already excludes, plus 88 and 106 from the same list.

The reason is structural: **in a pile every occluded man has a sideline body within 1.5 m by
construction -- his neighbour.** So "no sideline body nearby" cannot tell a man the sideline merged into
his neighbour from a man it already draws, which was the entire point. Recovering the line needs the
endzone bodies admitted with a correct identity, and that is the pairing problem again -- which needs
both cameras to see the ankles, exactly what a pile denies. Do not re-propose a proximity test here.

**But the deficit IS recoverable, and I nearly concluded otherwise on a bad number.** In the line box
(|x - LOS| <= 3 m, |y| <= 8 m) the endzone resolves **9-11 Kansas City ground points** on every sampled
frame from 290 to 360, against the sideline's 7-8 and the 6-7 the render actually draws. Pooling the two
cameras' points at 0.6 m then gave KC **12-13 places**, which is more men than can be in that box, and I
read that as the endzone over-segmenting the pile -- which would have killed the whole line of attack.
It is a clustering artifact: a man whose sideline and endzone points sit more than 0.6 m apart is
counted twice by the pooling, so the pooled figure inflates and means nothing here.

The film settles it. Marking the endzone's line-box ground points on frame 285 (= sideline 300) gives
**10 markers and exactly 10 distinct red-jerseyed men** -- 87, 76, 62, 15, 65, 7, 83, 12, 10 and one
half-hidden behind 87 -- each marker at a different man's feet, none stacked on one body. The endzone
genuinely resolves the men the sideline merges into a mass.

So the route is admission by **region and count, not identity**: inside the line box, where the endzone
resolves more distinct places than the sideline, the surplus are real men. That needs no pairing, which
is what the pile denies. The risk is drawing a body at the wrong depth, so it must be scored on 05t and
05u as well as the census. That was the next build, and it is dead too -- measured before writing it.

The candidates do not separate from the duplicates by distance. Splitting every Kansas City endzone body
in the line box by whether the sideline tracks that id **anywhere** (so the sideline can see him: a
re-count) or **never** (so it cannot: plausibly a man it merged), and measuring how far each sits from
the nearest sideline KC body:

    the sideline DOES track this id elsewhere (a re-count)   1150 body-frames   p50 1.16 m
    the sideline NEVER tracks this id (plausibly unseen)      579 body-frames   p50 1.17 m

    threshold   admits unseen   admits re-counts   purity
      0.6 m          436              873           33 %
      1.0            301              654           32 %
      1.2            287              531           35 %
      1.5            216              149           59 %
      2.0             82              101           45 %

Identical medians, and no threshold reaches even 60 % purity; the 1.5 m bump sits between 35 % and 45 %,
so it is a wiggle in a noisy curve, not a boundary. Tuning to it would be fitting noise.

The film had already shown why, on the snap frame: of the four candidates there, id 37 sits at lineman
76's feet with the sideline's own point 0.68 m away and id 22 sits on the back 10 with sideline points
beneath it -- both re-counts -- while id 38 and id 74 stand on men carrying no sideline point at all. Two
real and two duplicates, at 1.38/1.49 m and 0.68/1.06 m, ranges that overlap the moment cross-camera
disagreement is included. **A man the sideline merged into his neighbour and a man it already draws sit
the same distance from that neighbour.** Proximity cannot tell them apart, in the line box or anywhere
else.

So the census route through geometry is closed: admitting the line needs identity for those bodies, and
the pile is exactly what denies identity (both cameras must see the ankles). The remaining honest
options are a detector that separates overlapping bodies in the sideline view, or accepting that the
snap deficit is not recoverable from this footage.

**Jersey identity cannot admit them either.** The obvious next door: the endzone plainly shows the
linemen's numbers, the OCR reads them confidently (id 86 reads 62 on 250 frames, id 97 reads 82 on 215,
id 38 reads 55 on 326), so a body with a number is a man whether or not the ankle rays can pair him.
Dead, because `player_uid` is **not unique per man**: seven uids are claimed by two or three different ids
(`2024_T1_7` by ids 22, 76 and 80; `2024_T1_62` by 33, 139 and 140; `2024_T1_52` by 17 and 104), three
ids carry more than one uid (id 25 reads 76 on 112 frames, 62 on 87, 5 on 9), and the totals are 16 and
16 against a truth of 11 and 11. Admitting a body because it carries a number would draw the same man two
or three times -- the ghost problem in a new costume.

**Two corrections to myself, both from reading rather than assuming.** I reported the line-box candidates
as having "no identity" because `PlayerIdentity.name` was None: that attribute does not exist on the
object at all (`cameras, corroborated, height_m, jersey, player, team, total_votes, tracks, votes,
weight_lb`), and the identities are populated -- the sample resolves to jersey 21, BAL, "Brandon
Stephens". And I reported that `pipeline_play.sh` would reproduce roughly v34 because it runs 05p without
`--two-view`: it does not, line 59 sets `--two-view --endzone-weight $EZW` unless `ONE_VIEW=1`, and
`TWO_VIEW_PX_MAX = 40` is already the committed default inside 05p. The real gap was narrower -- 08r/08s
were not pipeline stages, so a fresh play got v38's fitting but not its pairing. Now wired in as `pair`,
between `twins` and `roles`, because 08s rewrites the tables that roles/tri/refit all read.

**The single worst body-frame in the play is id 17 at frames 458-460**, 4.49 m along the ray -- and the
two cameras' ankle rays there miss by **2.34 m**, so the triangulated "truth" is itself wrong. It is a
mis-paired window, not a misplaced body, and it accounts for id 17's 0.13 m across-ray component, the
last named placement defect. A candidate for the same interval trim.

**So 05u was scoring bodies against a truth it had never checked, and the fix moves the headline.** The
ruler now drops any frame whose two cameras' ankle rays miss by more than 0.6 m -- the same bound the
two-view fit uses to call a frame mis-paired -- because a crossing point the cameras disagree about is
not a truth to score against. On play 1 that is **15 body-frames of 2161**, and the effect is entirely
in the tail:

    pooled |along-ray|        v37 before -> after     v38 before -> after
    p99                        0.83 -> 0.70 m          0.39 -> 0.36 m
    max                        4.49 -> 1.50            4.49 -> 0.62

Every player's median is unchanged and it is still 0 of 27 worse, which is the signature of removing
noise rather than data. **The worst body-frame in play 1 is 0.62 m, not 4.49 m** -- the figure quoted in
the v36, v37 and v38 entries above was the mis-paired id 17 window being scored as placement error.

The general lesson, which cost three entries' worth of inflated maxima: a ruler that scores against a
triangulated point must validate the triangulation first. The same discipline as the cross-view ruler
itself -- measure with an instrument whose own error you have bounded.

### The census metric was scoring the aftermath (2026-09-12)

`|KC-11| + |BAL-11|` has been averaged from the snap to the END OF CLIP, and roughly half of it is the
post-play crowd rather than a defect:

    window                frames   KC mean   BAL mean   error   exactly 11/11
    current metric           360      11.5       12.0    2.82      10 of 360
    the play only (300-460)  160      11.1       11.7    1.98       9 of 160
      the snap  (280-340)     60       8.9       11.7    2.85       0 of  60
      the run   (340-460)    120      11.8       11.8    1.79       9 of 120
    after the whistle        100      12.8       12.6    3.46       1 of 100
    late (560-660)           100      11.0       11.7    3.52       0 of 100

So play 1's real body-count error is **1.98 a frame, not 2.84**, and every figure quoted above (2.95,
2.94, 2.84) includes 200 frames of players and staff walking onto the field after the whistle. Score the
play, not the aftermath. **The snap remains the one phase this does not explain away** -- KC 8.9 against
a truth of 11, 0 of 60 frames correct -- and that is the pile, whose only surviving route was a detector
that separates overlapping bodies in the sideline view.

**That route is closed too, and the film closed it.** Projecting the six Kansas City men the endzone
resolves in the line box but the sideline does not, onto the sideline plate at frame 300, and drawing
every box the detector actually found: **not one cross lands on bare grass.** id 74's falls on a lineman
already boxed twice; 38 and 22 land in the overlapping tangle of boxes 17, 25 and 82; 86 and 37 land on
bodies inside boxes 11 and 17; id 9's lands at the feet of a man the detector did find. The sideline
detector is not MISSING these men -- it is **merging** them, because from the sideline the line of
scrimmage is viewed edge-on and several players occupy one box. The endzone's extra resolution comes
from seeing that same pile face-on, not from a better detector.

So no detector threshold recovers them: separating bodies that overlap this completely needs instance
segmentation or a pose-driven split of a shared box, which is a different project, not a tuning change.

**The pose model was the cheap version of that split, and it does not reach them either.** 05m already
runs `yolov8x-pose.pt`, and a pose model returns one skeleton per person it finds, so a skeleton inside a
merged box would be the split for free. Tested per man at the snap: of the six Kansas City men the
endzone resolves and the sideline lacks, exactly **one** (id 38) has a pose skeleton nearer than any
existing box. No `-seg` weights are on disk, so true instance segmentation is a download plus rework of
every box-keyed stage downstream.

**A correction to my own first reading of that test.** The single-frame version appeared to show three of
the six "already inside a tracked box", which would have meant the men were detected and merely
unmatched -- a pairing problem, not a detection one. That was a pixel-versus-field artifact: it measured
to a box's bottom-centre in pixels, while a merged box yields **one** id and therefore **one** ground
point, so a cross can land inside a box with no separate body there at all. Repeating it on the field,
across frames 280-345: **69 of 72** missing-KC instances have no sideline body within 0.6 m under any id
or team, and only 3 do. The men really are absent from the sideline's body set, exactly as merging
predicts, and the closure stands.

**And the ghost rule is not what deletes them either -- measured and rejected 2026-09-13.**
`endzone_only_ids` drops an endzone-only id for its WHOLE life on a >= 50 % frustum vote, with no test of
whether the sideline actually has a body there, while its sibling `beyond_sideline_span` carries exactly
that test (`SAME_BODY_M = 1.2`). The asymmetry looked like the snap deficit's cause. It is not. Giving
`endzone_only_ids` the same-body test keeps 13 of 35 ghost ids back and the census gets WORSE on every
window: snap 2.82 -> 3.18, play 1.98 -> 2.23, run 1.79 -> 1.94, aftermath 3.44 -> 4.00, whole clip
2.89 -> 3.21. Kansas City at the snap does not move at all (8.89 either way), because none of the 13 kept
ids live at the snap: 11 of 13 are Baltimore, and the two Kansas City ones begin at frames 457 and 538,
after the play. They add surplus to a team already over eleven (BAL 11.70 -> 12.07). Eighth entry in
corrections-must-beat-what-they-correct.

The negative is the useful part: the six missing Kansas City linemen are NOT excluded as endzone-only
ghosts. Their endzone detections must reach the timeline under ids that also carry sideline rows, so no
endzone-only rule ever classifies them. The per-frame mechanisms that can still delete a body at the snap
are `beyond_sideline_span` (1680 body-frames left out on this play) and the timeline's own `dedupe_frames`:
a state the SIDELINE detected that frame is protected (the 2026-09-08 fix), but a state the ENDZONE alone
sees dedupes against every kept state inside an axis-aligned `ONE_VIEW_DEPTH_M` x `ONE_VIEW_ACROSS_M` =
4.0 x 1.5 m box in FIELD axes. On the line of scrimmage every lineman shares x and they stand about a
metre apart in y, which is inside that box -- the 7th correction-must-beat case recurring in the other
camera, since that fix protected only sideline-detected states. This also fits the 0.6 m number above
rather than contradicting it: a man 1.2 m from the merged body is genuinely "no sideline body within
0.6 m" and still inside the dedupe box.

**That dedupe hypothesis is measured and rejected too (same day).** The one-view box kills ZERO Kansas City
states in the snap window. Of 500 dedupe drops in the clip, 42 fall in 280-345 and every one is the
`interp` branch at `INTERP_DUP_M` = 0.4 m (35 Baltimore, 7 Kansas City) -- a second fragment id standing on
a body the sideline detected, which at 0.4 m is the same man. Sweeping the across-radius 1.5 -> 1.0 -> 0.8
-> 0.6 m changes the drop count not at all (500 every time) and every census window to the digit, so that
branch binds nowhere on this play. The reason is `_nearest_views`: an id with any sideline record within
`MAX_GAP_FRAMES` inherits "sideline" among its views and is then either protected as detected or deduped as
interpolated, so almost nothing reaches the one-view box. The instrument passed its own validity gate --
the instrumented copy of `dedupe_frames` reproduced the real run exactly (0 frames differing, identical id
sets, 500 drops both).

**Both exclusion routes are now closed, and what is left is bookkeeping.** The six men are in the endzone's
detections, are not endzone-only, are not excluded by any rule, and are not deduped -- so their endzone
rows hang off global ids whose SIDELINE rows belong to other men. Because `side_ground` overwrites `ground`
(play_timeline.py:327-329), such a body is drawn where the sideline puts its own man and the endzone's extra
detection adds no body at all: ten endzone men plus eight sideline bodies still render eight. That is the
bookkeeping figure already recorded -- 21 sideline ids and 24 endzone ids at frame 300 sharing only 12 --
and it is the pairing being locally wrong across a window where the track as a whole is right, which
`mispaired_ids` cannot see because it gates on the MEDIAN distance over a whole track and a 60-frame error
does not move a median. Next test: per frame at the snap, the gap between a global id's sideline and
endzone ground points; where it is large the endzone row is a different man and deserves an id of its own.

**THE SNAP DEFICIT IS RECOVERABLE, and two claims above are now wrong (2026-09-13).** Two rules were each
masking the other, which is why every single-rule measurement read as a no-op:

  - `beyond_sideline_span` drops ids 74/38/22 at the snap, so they never reach dedupe. Measuring dedupe
    alone therefore showed 0 Kansas City states killed by the one-view box and no sensitivity to its
    radius -- true of the shipped configuration, FALSE of the mechanism. The claim above that the box
    "binds nowhere on this play" holds only while the span rule deletes its inputs first.
  - Give the span rule its same-body test and those 751 body-frames reach dedupe, where the one-view box
    kills every one (id 74 by id 1 on 65 of 65 frames, id 38 by id 4, id 22 by id 13; `n_duplicates`
    500 -> 1585). That is why the span fix alone measured +0.00.

Corrected together, the census moves for the first time. Kansas City at the snap 8.9 -> 10.8, frames
exactly eleven-a-side at the snap 0 -> 23, and Baltimore never moves (11.7), so the change is targeted:

    configuration          snap   play    run  after   full
    baseline (shipped)     2.82   1.98   1.79   3.44   2.89
    span fix only          2.82   2.03   1.86   3.53   2.90
    span fix + across 1.0  1.20   1.73   2.00   3.57   2.52
    span fix + across 0.8  0.93   1.67   2.01   3.66   2.43

**The footage and a control group say ids 74 and 38 are real men, and 22 and 86 are not.** Paired Kansas
City ids agree between cameras at p50 **0.42 m** (n = 380 rows in 280-345). Ids 74 and 38 stand **1.44** and
**1.24 m** from the nearest sideline body, and the gap is mostly ACROSS the field (|dy| p50 1.19 m for
both), which is the direction a formation's linemen separate -- not along the endzone's blind depth axis.
Ids 22 and 86 come in at 0.65 m, inside the control distribution, and contributed only 1 and 10 frames.
Zoomed crops agree: box 12 plainly spans two to three Kansas City players with a single ground point inside
it, the merged box this document describes.

**NOT SHIPPED, because the fix as swept is a blanket threshold and it over-admits.** The run window
regresses 1.79 -> 2.01 with Kansas City at 12.1, over eleven: a smaller dedupe box recovers real men at the
line and admits copies once the players spread out. The principled form is the same-body test INSIDE the
one-view box -- an endzone-only state is a copy only where the sideline actually has a body -- rather than
a narrower box, which is the same lesson as `beyond_sideline_span` carrying `side_ground`. Measure that
before changing any constant.

**And that principled gate is measured and REJECTED -- it is worse than the blanket threshold (2026-09-13).**
Requiring the sideline to have a body within `same_body` before calling an endzone-only state a copy
regresses the run to 2.24-2.50 against a baseline of 1.79 (the blanket radius gave 2.00), and it breaks what
the blanket never touched: Baltimore goes 11.7 -> 12.7 at EVERY window, a whole extra body a frame, with
frames exactly eleven-a-side collapsing to 2-3 where the blanket at 0.8 gave 41. Whole clip 2.98-3.21
against 2.89 baseline, a net loss.

The reason, plain in hindsight: the gate replaced "near a kept state" with "the sideline has a body nearby",
so a state FAR from any sideline body is kept -- which is exactly the play-2 endzone ghosts strung along x
that `ONE_VIEW_DEPTH_M` was built to kill. The rule I called principled readmitted the precise failure the
box exists for. Ninth entry in corrections-must-beat-what-they-correct. A second tell, worth remembering:
`box AND sideline` and `sideline only` print identical numbers at every threshold, so once the same-body
test is added the 4.0 x 1.5 m box never binds -- the two conditions are not independent and pairing them
buys nothing.

So the only configuration that helps Kansas City without disturbing Baltimore is still the blanket
`span fix + across ~1.0`, and its run regression (1.79 -> 2.00) is the open question. Diagnose which bodies
that window gains before shipping anything: if they are copies, a discriminator exists; if the control test
calls them real men while truth is still eleven, something else is wrong.

**Measured: the run regression is NOT caused by the change, and the census is the broken instrument there
(2026-09-13).** The bodies that window gains under `span fix + across 1.0` come to 41 body-frames over 121
frames -- **0.34 a frame** -- and by the control test most are real men: id 74 for 33 frames at |gap| 1.40 m
with |dy| 1.30 (the same man the snap analysis found), plus ids 38, 54 and 19 for one or two frames each.
Only ids 11 and 25 are copies, 4 frames between them at 0.26 and 0.19 m, inside the control distribution.

The arithmetic that matters: baseline Kansas City in the run window is already **11.79** against a truth of
eleven, so roughly **0.8 phantom Kansas City bodies a frame exist before any change**. Adding 0.34
mostly-real men lifts |KC - 11| from 0.79 to 1.13, which reads as a regression while the render is getting
MORE correct. A census cannot referee a change that adds true bodies to a count that is already too high.

So the run is not a reason to withhold the dedupe fix -- but it is not yet a reason to ship it either,
because the pre-existing surplus is unexplained and could share a cause with the bodies being added. Find
the phantom first: which drawn Kansas City bodies in 340-460 stand within the control's 0.95 m of another
drawn Kansas City body, and whether any id labelled KC is misassigned. Note the surplus is phase-specific --
8.89 at the snap (a deficit), 11.07 over the play, 11.79 in the run -- so it grows as the players spread
out, which is the opposite of what a merged-box fault would do.

**First attempt at that measurement was invalid, in two ways worth recording.** It counted PAIRS of drawn
Kansas City bodies closer than 0.95 m and got 2.31 a frame, which cannot be compared with a 0.79 surplus:
pairs overlap -- (17, 25), (25, 33) and (19, 25) all share id 25 -- and the phantom count is bodies minus
CLUSTERS, not pairs. Second, 0.95 m was the p90 of the CROSS-CAMERA control (the same id seen by both
cameras), which is not a "two different teammates" radius at all; teammates legitimately stand about a metre
apart, so that test declares real players duplicates. It is the 0.6 m clustering artifact from earlier in
this document wearing a different hat. Any cluster radius here has to be one below which two real players
cannot both stand (~0.4-0.5 m), and even then same-team clustering during a run is a weak instrument.

**What does survive: fourteen ids are drawn as Kansas City in the run window.** Nine run near-continuously
(111-121 of 121 frames: ids 3, 5, 9, 12, 37, 11, 25, 38, 17) and five are partial -- 80 (84 frames), 33
(83), 19 (75), 39 (69), 74 (52). With truth at eleven, the surplus rides on the partial ids, which is the
signature of fragments rather than of a threshold.

Two leads, neither yet confirmed:
  - id 39 carries jersey 6 and a roster name that is a DEFENSIVE back. Kansas City has the ball on this
    play, so a safety cannot be on the field; if the roster confirms the position, that is 69 frames of a
    body that should not exist. Verify against the pipeline's roster rather than from memory.
  - id 74's "defence side of the line" flag is an artifact: its mean x is 0.69 m past the line of scrimmage,
    which is where a tight end on the line stands. Not evidence, and not a reason to doubt the footage and
    control-group verdict that id 74 is a real man.

**The run surplus is localised and it is an IDENTITY fault, not a geometry one (2026-09-13).** Counting
clusters rather than pairs: bodies 11.79 a frame, clusters 11.56 at 0.40 m and 11.29 at 0.50 m, so only
~0.2-0.5 of the surplus is one man counted twice and the rest stand genuinely apart. (At 0.95 m clusters
read **9.69**, below truth -- independent proof that radius merges real men and that the earlier pair-based
reading was invalid.) The surplus is not spread across the window either: frames 340-380 read 10-11 bodies,
correct, and it climbs to 13-14 across **390-430**. Two ids carry it: id 39 adds +1.41 bodies a frame when
present and id 80 adds +1.52, against 74 (+0.75), 19 (+0.50) and 33 (+0.26).

Checked against the 2024 roster, those ids' jerseys do not belong on the field:

    id 39  jersey  6  Bryan Cook          DB   -- defence, and Kansas City has the ball
    id 38  jersey 55  Uche / Bozeman      LB   -- defence (Bozeman is practice squad, status DEV)
    id 19  jersey 67  no Kansas City player wore 67 in 2024
    id 80  jersey  0  P80  -- NO jersey was read for this id at all (corrected below)
    id 74  jersey 83  Noah Gray           TE   -- correct
    id 25  jersey 76  Kingsley Suamataia  OL   -- correct
    id  5  jersey 10  Isiah Pacheco       RB   -- correct

**A wrong jersey does not make a body false, and the two failures need separating.** id 38 was validated as
a real man by the control group and the footage; what is wrong there is its identity, not its existence. But
a real body with a wrong TEAM label corrupts the census directly, because the census counts by team -- so
team labelling is now a third suspect instrument, alongside the pair-versus-cluster error and the borrowed
radius. Before trusting any census number by team again, establish how often the team label is wrong.

**Cross-team double-drawing is NOT the explanation, and the census-by-team survives (2026-09-13).** At frame
420 the id-39 cross sat beside Baltimore body id 4, which would mean one man drawn under two team labels --
inflating BOTH counts, matching KC 11.79 and BAL 11.84 both sitting over eleven, and invisible to a dedupe
that never looks at team. Measured over 380-440 it is false: id 39's nearest opposite-team body is **p50
1.89 m** (id 4 is its commonest neighbour but on 23 of 54 frames at ~1.9 m), and id 80's is **p50 3.16 m,
min 3.06 m** -- nowhere near a Baltimore body. The picture was single-frame proximity, not co-location.

Across the whole clip, cross-team pairs closer than 0.30 m are **1.0 %** of 14 397 drawn body-frames and
closer than 0.40 m **1.4 %**. That is not systemic, so every census-by-team figure in this document stands,
**including the snap 8.9 -> 10.8 result**. It also cannot explain a 0.79-a-frame surplus, which would need
~6.7 % of the Kansas City bodies.

What the same probe did find: **id 19's nearest SAME-team body is p50 0.58 m, min 0.17 m** -- inside the
control distribution, so that one is a genuine double. Accounting for the 0.79 surplus so far: same-team
near-duplicates account for ~0.2-0.5 of it (clusters 11.56 at 0.40 m, 11.29 at 0.50 m against 11.79 bodies),
leaving ~0.3-0.5 as bodies standing genuinely apart. The remaining candidate is officials or staff drawn as
players -- a failure mode already recorded on this play, where three officials and a field marker were drawn
as Baltimore players and one got a roster name.

**Where this leaves the dedupe fix.** Its two supporting rulers -- the cross-camera control group (paired ids
agree at p50 0.42 m, so ids 74 and 38 at 1.24-1.44 m with |dy| ~1.19 m are real men) and the footage -- are
both independent of team labels, so neither depends on the census being sound. The run regression is now
explained as a pre-existing census error the fix does not cause. `ONE_VIEW_ACROSS_M` also has a data-driven
justification rather than a census-fitted one: the same man reads <= 0.95 m across cameras, real separate men
read >= 1.19 m, so a radius of 1.0 m sits between the two populations.

**SHIPPED 2026-09-13.** `play_timeline` now passes `side_ground` to `beyond_sideline_span`, and
`ONE_VIEW_ACROSS_M` is 1.0 (was 1.5), set between the two measured populations -- same man across cameras
<= 0.95 m at the p90, separate men 1.19-1.30 m. Play 1: snap 2.82 -> 1.20, play 1.98 -> 1.73, whole clip
2.89 -> 2.52, frames exactly eleven-a-side 10 -> 33, Kansas City at the snap 8.9 -> 10.8 with Baltimore
unchanged. A test covers the dedupe half (an endzone-only state 1.2 m across is kept; it fails at 1.5).

Verified on the SHIPPED path rather than the prototype: every figure above came from probes that
monkeypatched `beyond_sideline_span` and rebuilt `side_ground`, while production passes the real local that
has been through `snap_ground`. An unpatched `load_play_timeline` reproduces every window exactly. The
discrimination is also sharper than the census alone shows -- body-frames drawn over 280-345 are
**{74: 66, 38: 39, 22: 0}**: the two ids the control group and footage called real men are drawn, and id 22,
which the same control called a copy at 0.65 m (inside the 0.95 m p90), is still dropped. The change admits
exactly what the two non-census rulers vouched for and nothing they rejected.

**The `ONE_VIEW_ACROSS_M` half was REVERTED within the hour: play 2 regresses.** Measured on play 2, of the
123 states 1.0 keeps that 1.5 drops, |dx| to the body they duplicate is **p50 2.87 m** (p90 3.52) -- id 2 on
53 frames at 3.46 m, id 34 at 2.90, id 1 at 2.87 -- the endzone's copy strung along its own depth axis,
exactly the failure that radius exists to prevent. Breaking play 2 to fix play 1 is the thing
corrections-must-beat-what-they-correct forbids, so the constant is back at 1.5 and the test now guards the
play 2 case instead. `side_ground` stays: it is the half that restores the bodies, and it harms nothing on
its own (play 1 +0.00, play 2 unaffected) -- it simply cannot pay off until dedupe stops eating what it
restores.

**But the same measurement hands over the discriminator.** In the marginal band (|dy| 1.0-1.5) the two
populations separate cleanly on |dx|, the axis nobody was looking at:

    play 2 ghosts          |dx| 2.4-3.5 m    the endzone's copy, displaced in DEPTH
    play 1 merged linemen  |dx| 0.38-0.79 m  a different man beside him, at the SAME depth

A copy is displaced along x; a separate man is displaced across y at the same depth. So the fix looked like a
depth-aware exception -- an endzone-only state beside a kept state (|dx| small) but clear of it across
(|dy| >= ~1.0) is a different player -- rather than a wider or narrower box.

**Measured on both plays, and REJECTED (2026-09-13), because the premise was my own measurement error.**
The exception does nothing on play 1: the snap census stays 2.82 at near_depth 0.8, 1.0 and 1.5, with id 74
drawn on 3 frames against 66 under the blanket radius; only 2.0 moves it at all (2.64, 14 frames). On play 2
it newly keeps 54 states at every setting (|dx| p50 0.29) -- the ghosts stay dead, but it is pure cost with
no benefit.

The premise was wrong: **|dx| 0.38-0.79 m was id 74's distance to the nearest SIDELINE BODY** (from the
offset-control probe), not its separation from the kept state that dedupe actually compares against, which
is id 1. Two different relationships; I carried the statistic from one into the other and built a rule on
it. Sixth instance this session of the same error family -- see the standing note about stating the
population before reading a number.

**Measured, and the line is CLOSED (2026-09-13).** Separation from each dropped state to the kept state that
actually kills it:

                     n    |dx| p50   p90     |dy| p50   p90
    play 1 (KC snap)   163     2.07    2.48       1.11    1.39
    play 2 (all)      1267     2.38    3.75       0.70    1.28

    |dx|      p10    p25    p50    p75    p90
    play 1   0.94   1.98   2.07   2.29   2.48
    play 2   0.09   0.77   2.38   3.22   3.75

Play 2's ghosts span 0.09-3.75 m and straddle play 1's entire 0.94-2.48 range, so **no threshold in
(|dx|, |dy|) keeps the linemen and drops the ghosts**. Narrowing the DEPTH reach instead fails for the same
reason: a quarter of play 2's ghosts sit inside 0.77 m.

This also corrects the framing above. id 74's killer is id 1 at **|dx| 2.03**, and id 38's is id 4 at 2.41 --
not a box-mate 0.38-0.79 m away, which was the distance to the nearest sideline BODY. The linemen are
deleted because a body sits ~2.4 m off in depth, so it is the box's 4.0 m depth reach that swallows them,
not the across radius I spent the evening adjusting.

**What would be needed is a different feature, not a better threshold** -- the pair's geometry does not carry
the answer. Candidates never attempted: appearance (do the two states look like the same player?), or
tracking which camera's detection is genuinely unpaired rather than inferring it from distance. Four
replacements for this rule are now measured and rejected. `side_ground` stays on main because it is
measured-neutral and a prerequisite for any future version; the snap gain does not ship.

Two caveats from the original entry, one now discharged. `ONE_VIEW_ACROSS_M` exists for PLAY 2's endzone
ghosts strung along x and only play 1 was measured -- if those ghosts return, this belongs behind a per-play setting rather than in the
constant. And no unit test guards the call-site wiring: a future edit could drop `side_ground` at
play_timeline.py:337 and only the constant's test would still pass.

**The run surplus is an identity fault, and it splits into two faults needing opposite fixes (2026-09-13).**
Of 17 jersey-carrying drawn ids on play 1, 11 are consistent with the play and 6 are not, covering **1798 of
14445 body-frames (12.4 %)**. Kansas City has the ball, so KC ids should carry offensive numbers:

    id 39  KC  jersey  6 = DB (Bryan Cook)      233 frames   nearest body p50 0.60 m, 70 % inside 0.95 -> DUPLICATE
    id 42  BAL jersey 76 = OL (Dalcourt)        196 frames   nearest body p50 0.60 m, 84 % inside      -> DUPLICATE
    id  3  KC  jersey 18 = not on the roster    623 frames   p50 2.72 m,  0 % inside                   -> distinct
    id 38  KC  jersey 55 = LB (Uche/Bozeman)    172 frames   p50 1.47 m, 15 % inside                   -> distinct
    id 78  BAL jersey  4 = WR (Zay Flowers)     167 frames   p50 4.17 m,  0 % inside                   -> distinct
    id 19  KC  jersey 67 = not on the roster    407 frames   p50 1.21 m, 41 % inside                   -> BIMODAL

A wrong jersey does not make a body false: the distinct ones are real men whose LABEL is wrong, and deleting
them would delete players. The two duplicates are extra bodies. Note `dedupe_frames` cannot remove them --
a state the sideline detected that frame is never a duplicate (the 2026-09-08 rule) and these are
sideline-detected -- so whatever fixes this is a different mechanism, not a radius.

**Two corrections to claims I already committed.** id 39 was called "a distinct real body, not a duplicate"
on the strength of the cross-team probe, which measured **frames 380-440 only** (nearest same-team 1.20 m);
over the whole clip it sits 0.60 m from something 70 % of the time. Same id, different window, opposite
answer. And id 19's "distinct" verdict is an artifact of reading a median: p50 1.21 m with 41 % of frames
inside 0.95 m is bimodal, not distinct. Seventh and eighth instances tonight of a number read without its
population.

**None of them is a duplicate: the "count fault" half is wrong and withdrawn (2026-09-13).** Asking WHICH
body each flagged id sits on, rather than how often something is near it:

    id 39  162 close frames  partners id 25 x74 (46 %), 80 x22, 42 x21, 84 x13   -> rotating
    id 42  164 close frames  partners id 25 x43 (26 %), 15 x37, 39 x35, 37 x20   -> rotating
    id 19  166 close frames  partners id 25 x79 (48 %), 33 x39, 20 x24           -> rotating
    id  3    0 close frames  p50 2.72 m from anything                            -> distinct
    id 78    0 close frames  p50 4.17 m from anything                            -> distinct

No twin tracks. The "70 % within 0.95 m" figure measured CONGESTION, not duplication -- id 25 (Suamataia, an
interior lineman) is the commonest neighbour of three different ids because everyone near the pile is near
him. The earlier "DUPLICATE (count fault)" verdict on ids 39 and 42 rested on a proximity share without ever
asking whether the neighbour was the same body, and it is withdrawn. Ninth instance tonight of a number read
without its population.

So all six flagged ids are **real bodies carrying wrong labels**, removing any of them deletes a man, and the
census cannot be improved by removal at all.

**What that sharpens.** id 3 is drawn on **623 of 647 frames** (KC, jersey 18, not on the 2024 roster)
standing p50 2.72 m clear of everyone, and id 19 on **407 frames** (KC, jersey 67, also not on the roster).
A near-permanent body that no roster number matches is the shape of an official or a staff member that
`sideline_dwellers` / `striped_ids` / `behind_the_offence` did not catch -- which would explain the run
surplus better than anything tested tonight, and would be a bounded fix rather than a threshold. The footage
decides it; proximity statistics demonstrably do not.

**The footage kills both remaining explanations: they are all real players (2026-09-13).** id 3 is a Kansas
City man in full uniform in a three-point stance at the line, on two separate frames -- so the "officials the
off-field rules missed" idea is dead, and the biggest flagged id (623 frames) is correctly counted with only
its jersey misread. Detection status settles the rest:

    id 19  drawn 407  detected 380  interpolated 27 (7 %)   frame 500 interpolated
    id 78  drawn 167  detected 167 (100 %)                  frame 500 DETECTED (sideline)
    id  3  drawn 623  detected 622                          frames 400 and 500 detected

**And my "phantom on empty grass" reading was a sampling error.** I cropped id 19 at frame 500 -- one of its
27 interpolated frames, 7 % of its life -- and read it as evidence about the id as a whole. id 78 is 100 %
detected including that frame, so its cross belongs to a real player in the crop (most likely the Baltimore
man above right) with the ground point placed low, not to a body on bare turf. Tenth instance tonight of
sampling the rare case and generalising from it.

So there are no phantoms and nothing to remove: **all six flagged ids are real detected bodies with wrong
labels**, and the count-fault half of the identity thread is dead alongside the officials idea.

**What is left is the one assumption never tested: that truth is eleven in that window.** If the whistle
falls before frame 460, the tail of the "run" window is people walking on -- exactly the error already
corrected once in this document, when the census was scoring 200 frames of aftermath. The end of the play
has never actually been located; it was assumed from the window bounds.

**The end of the play is still unlocated, and the probe built to find it was broken (2026-09-13).** Two
proxies, both useless, recorded so nobody rebuilds them:

  - **Runner speed.** Taking the FASTEST Kansas City body per frame reports 48.0 m/s at frame 430, 35.6 at
    500 and 24.6 at 465. A world-class sprinter peaks near 12. A max over all bodies is dominated by
    whichever id is glitching, so "fastest KC >= 4 m/s across 300-560" says the metric is noise, not that
    the play is live. Any speed signal here needs ONE identified carrier and a median filter.
  - **Bodies beyond the sideline.** Zero anywhere in 300-560 -- but the crowd-crossing evidence recorded
    earlier in this document is at frames **580-620**, outside the window I scanned. The signal was real and
    I looked in the wrong place.

Eleventh instrument error of the session. What the run does show is genuine: drawn bodies climb from 20-21
at frames 300-335 (Kansas City 9, the snap deficit), to 23 at 345 as the pile separates, to **25-26 with
Kansas City at 13-14 across 390-430**. The surplus is concentrated exactly where the play is most congested,
which fits neither "aftermath crowd" nor any removal route -- all six flagged ids being real detected
players. The end of the play needs settling on the FOOTAGE, since eyes have been reliable tonight and
derived proxies have not.

**The play is LIVE through frame 470, so the window was right and the census thread closes here
(2026-09-13).** At frame 425 (24 bodies, KC 13, BAL 11) players are running and engaged in blocks with a
carrier in the scrum; at frame 470 (25 bodies, KC 12, BAL 13) a tackle is still forming, and the people at
the top of the plate are off-field on the sideline. No whistle, no converging officials, nobody standing
around. The aftermath explanation for the run surplus is therefore dead too, and 340-460 is all live play.

**Where that leaves play 1's body count.** Every removal route is closed -- the ghost rule, dedupe, cross-team
double-drawing, duplicates, officials, phantoms, and now the window itself. All six odd ids are real detected
players carrying wrong labels. The surplus is genuine over-counting during the congested phase, of which the
cluster test accounts for only ~0.2-0.5 bodies a frame (clusters 11.56 at 0.40 m and 11.29 at 0.50 m against
11.79 bodies); the remainder is unexplained. **Census stands at 1.98 a frame over the play, where the session
found it.**

One discrepancy recorded rather than acted on: at frame 425 the census reads KC 13 while a count of red
shirts on the plate gives roughly 10-11, with a dense knot of Kansas City dots over some 4-5 players in the
pile. That looks like more duplication than the cluster test measured. Eyeballing a pile is precisely what
produced several of this session's errors, so the instrument stands until a better one is built -- but the
disagreement is real and is the first thing to re-measure, with a method that can count bodies in a pile
without a radius.

**And the obvious candidate for that method does not work.** Pose skeletons look like a radius-free counter
-- one skeleton per person by construction -- but `keypoints_2d.parquet` is
`[frame, cam, global_player_id, joint, x, y, conf]` with exactly 17 joints per id and **no per-detection
key**. Each skeleton IS an id, so counting skeletons recounts the very ids the census counts and arbitrates
nothing. (Consistent at frame 425: 20 sideline skeletons against 24 drawn bodies, the difference being
endzone-only and interpolated states.) A genuinely independent count needs a FRESH pose inference over the
crop, ungated by tracking -- a GPU build rather than a query. Recorded so nobody spends an hour rediscovering
that the keypoints are id-keyed.

**Tiled inference does NOT recover the merged players -- rejected on the footage, twice (2026-09-14).**
The full-frame pose pass finds 19-20 people where 22 players plus officials stand, so it under-detects, and
the obvious remedy was tiling: 3x2 overlapping tiles each resized to 1920, merged by NMS. The gate passed
perfectly (every full-frame person reproduced, none lost) and it "recovered" 7/6/9/26 extra people. All of
it collapsed under inspection:

  - **off-field contamination.** At f470, 14 of 26 recoveries are bench and staff at |y| ~26.8 m, which a
    2x resolution now resolves. Recovered is not recovered PLAYER.
  - **truncated boxes.** Recoveries at 0.03-0.14 m from an existing body are the same man re-detected: a
    partial box's bottom edge is at mid-torso, so back-projecting it to the ground lands metres beyond the
    player. The "0.6-1.6 m separation" I read as shoulder-to-shoulder lineman spacing was that artifact.
  - **the height test was necessary but not sufficient.** Back-projecting each box to an implied stature
    (gate: real full-frame boxes read p50 1.78-1.80 m, matching the project's 1.85 m) dropped the slivers at
    0.50-1.38 m and kept 2 per frame at 1.43-2.11 m. The footage then showed all four survivors are narrow
    FULL-HEIGHT strips down men who already have a box. Testing vertical extent never tested width.

So higher resolution yields duplicate boxes of the same men, not the merged ones. The earlier closure --
those bodies overlap genuinely in the image and need instance segmentation rather than more pixels -- now
rests on much stronger evidence, and a future session should not spend a GPU build rediscovering it.

**This also walks back the "pile over-count" reading, in the conservative direction.** The census drawing 24
bodies where the sideline detector sees 20 is NOT evidence of over-counting: the sideline merges 2-4 men and
the endzone legitimately supplies them. The "duplicates" found by matching drawn bodies to sideline skeletons
are that merging, which was already known, not extra bodies.

**The untried feature is cross-camera RAY MISS.** The one-view dedupe box cannot separate play 1's linemen
from play 2's ghosts in (|dx|, |dy|) -- but if an endzone-only state is truly the endzone's copy of a kept
sideline player, their two rays should nearly intersect, while two different men's rays miss by a metre or
more. 05u already computes that quantity and the dedupe rule has never used it.

**Cross-camera ray miss fails too, and backwards (2026-09-14).** The prediction was that an endzone GHOST is
the endzone's copy of a man the sideline also sees, so the two rays nearly intersect, while a merged LINEMAN
is a different man whose ray misses by a metre. Measured over every one-view-box kill on both plays:

    play 1 (merged linemen)  p10 0.28  p25 0.68  p50 0.75  p75 0.81  p90 0.84   under 1.0 m: 100 %
    play 2 (endzone ghosts)  p10 0.28  p25 0.75  p50 0.99  p75 1.31  p90 1.74   under 1.0 m:  51 %

Play 2 sits HIGHER than play 1, the p10s are identical and the sub-0.5 m shares are 14 % against 18 %. No
separation, and the sign is inverted. Two reasons it could never have worked, worth keeping: the
cross-camera control already showed the SAME man disagrees by 0.42 m, so a 0.4-0.5 m noise floor swamps a
signal of this size; and only 514 of 1089 kills on play 1 and 290 of 1267 on play 2 have ankle rays in both
cameras, so it would gate under half the cases even if it separated.

**That is the fifth rejected candidate and the (dx, dy)-style dedupe line is closed permanently** -- ghost
rule, blanket radius, depth-aware exception, any box at all, ray miss. No sixth variant of the same shape.

**What the failures point at is a COUNT, not a distance.** Every rule so far asks "is this endzone state near
a kept sideline state?". The mechanism is the other question: "does the endzone resolve TWO men where the
sideline sees ONE?" A merged lineman should show two endzone detections against one sideline body; a play 2
ghost should show one endzone detection duplicating one sideline man. That uses the endzone's own separation
as evidence, which no rule has done, and it is the last untried shape.

**The wide-killer-box rule is the first one that MECHANICALLY works, and still is not shippable
(2026-09-14).** Rule: in the one-view branch, a killing sideline box that is anomalously wide for its height
holds two men, so the state it would kill is the second man and survives. It measures merging rather than
inferring it from position, which the six earlier candidates all did.

    play 1  threshold   snap    play    run    full   11/11   watch 74/38/22
            off         2.82    2.03   1.86    2.90    10     0/1/0
            0.79        2.49    2.09   2.09    2.97     7     61/26/0
            0.90+       2.82    2.03   1.86    2.90    10     0/1/0     (exact no-op)

    play 2  off 35.10 | 0.79 35.15 | 0.90 35.14 | 0.96 35.14 | 1.05 35.13   bodies a frame

Two genuine firsts. **Play 2 is untouched at every threshold** (+0.03 to +0.05 bodies a frame) -- the first
candidate all session that does not pay for play 1 with play 2's ghosts. And at 0.79 it draws **ids 74 and
38** (61 and 26 frames at the snap), exactly the men the cross-camera control and the footage independently
validated as real.

Rejected as shippable on two pre-committed criteria. There is **no plateau**: only 0.79 acts, and 0.90 upward
are exact no-ops -- mechanically sensible, since skipping one wide killer merely lets a narrower kept body
kill the victim instead, so enough killers must be skipped, which makes the effect fragile. And it is **net
worse**: the snap improves 2.82 -> 2.49 while play, run, aftermath and the whole clip all regress
(2.90 -> 2.97), with exact eleven-a-side frames 10 -> 7.

**The reason is diagnostic, not fatal.** Kansas City rises 8.9 -> 10.2, toward truth. Baltimore rises
11.7 -> 12.7, away from it. The test is not team-aware and Baltimore's defensive linemen are merged in the
same pile, so it recovers real men on BOTH sides -- and Baltimore was already over eleven before any change
of mine. **Baltimore's baseline surplus is what stands between play 1 and a correct census**, and nothing
tonight has examined it: at the snap BAL reads 11.70 against a truth of 11. The relabel experiment cannot
explain it either, since all four flipped ids contribute zero body-frames at the snap.

**THE TELEPORTS -- what the user sees, and the census cannot (2026-09-15).** v38 shows players swapping,
teleporting and jittering; a Baltimore man standing in the Chiefs' O-line then snapping back to linebacker.
A plausibility ruler (per-id contiguous step; nobody exceeds 0.20 m/frame = 12 m/s) finds 487 of 14 384
body-frames over the sprint bound and a top speed of 73 m/s. I had this signal a day earlier -- a probe
reported the fastest body at 48 m/s -- and dismissed it as a noisy metric. It was the defect.

Three distinct mechanisms, each found by tracing raw per-camera positions through the pipeline stages:

  1. **A welded tail.** id 19 = a KC sideline track (14-430) with a Baltimore ENDZONE track (483-639)
     stitched onto the same global id. Its kit vote says BAL while the head says KC; three independent
     signals (73 m/s step, live-play colour flip 0.90 -> 0.20, tail's own kit majority) agree. Cut at 430
     -- the sideline span's end, NOT the teleport frame 493, which was interpolation racing toward the tail.
     **Verified: id 19's worst step 1.23 m -> 0.32 m** (74 -> 19 m/s). The tail (id 146, BAL) is
     endzone-only and the ghost rule drops it, correctly -- that linebacker has his own sideline id.
  2. **Interval mispairs.** A pairing right for most of a track and wrong for a stretch: id 17 carries four
     sideline frames (457-460) of a man 6.2 m from its endzone track; id 82's endzone rows from 315 begin
     5.9 m from where its sideline ended; id 5's endzone sits 6.2 m from context for 389-403. mispaired_ids
     misses all of it because it gates on the whole-track MEDIAN. Fix: per-frame cross-camera disagreement
     over the control's 0.95 m for >= 4 frames marks a bad run; the intruder camera is whichever is
     discontinuous with the id's own positions outside the run; its rows move to a fresh id with a team
     from its own kit colour. A margin (intruder >= 1.5 m and >= 2x the other camera) is essential -- without
     it 50 intervals fire, most coin-flips at 0.6 m vs 0.6 m. With it, 13 intervals, 187 rows, every one
     unambiguous. APPLIED; timeline verification running.
  3. **`smooth_xy` averaged ACROSS GAPS -- fixed.** id 21 moved 0.88 m/frame at 240-242 with NO raw
     detection in either camera from 244 to 320 and raw positions at 236/242 only 0.1 m apart. My first
     hypothesis was place_from_refit; the stage trace showed its shift was 0.00 on every detected frame. The
     jump appeared only between build_timeline's input and its states, and only in a segment's last FOUR
     frames -- the smoother's pad width. smooth_xy compacted every finite row into one array before
     convolving, so frame 243 sat beside frame 320 and the 9-frame mean blended one segment's tail with the
     next segment's head, 77 frames and metres away. Now smoothed within each contiguous run only.

  4. **Same-camera track switches** (scripts/08t_cut_track_switches.py). One camera's own track jumps to a
     different man across a short gap: id 82's sideline sat at x = -23.1 to frame 307 and resumed at 315 at
     x = -28.9 -- 5.9 m in 8 frames, which fill_gaps then drew as a body crossing the field. Both cameras
     agree on the NEW man afterwards, so no cross-camera test (08u) can see it. Two things made the detector
     usable: PERSISTENCE (the medians either side of the gap must stay apart -- the endzone's depth noise
     blips ~1 m every frame and fired 90+ times without it) and scoping to the LIVE play by --max-frame,
     because after the whistle tracks legitimately hop between milling bodies. Applied to play 1 with
     END_LIVE=470: 16 cuts, 3524 rows, frames 41-465.

**Progress on the timeline ruler** (contiguous steps over 0.6 m/frame, whole play): v38 **26** -> id 19 cut
22 -> 08u 14 -> smooth_xy fix 3 -> 08t (16 cuts) **1** (id 76 at 637, after the whistle). At the tighter
0.25 m/frame bound (15 m/s), which the corrected smoother makes necessary because it spreads a 4-frame ramp
over 9, 218 steps remain across 30 ids -- mostly 0.3-0.5 m and mostly after the whistle, plus a handover
artifact on fragment 161 at 47-50 where its endzone rows precede its sideline rows by three frames. Census
unchanged by all of it (play 2.01, full 2.90): the cuts moved men, they removed none.

**And 08v** (scripts/08v_remap_poses_after_relabel.py): the pose caches are keyed by global id, so every
fragment 08t/08u/08s creates rendered DEFAULT-POSED until now -- 29 fragments, 900 posed frames on play 1.
The mapping is exact by (cam, frame, track_id) between a snapshot and the result; a pose follows the
SIDELINE row's relabel, because that is where the timeline draws the body. Runs under PYS (numpy-1 pickles). All of it is now a pipeline stage (`switches`, after `pair` so 08c cannot erase the fragment
identities), with 08u and 08t carrying pure, tested planning functions.

Failed calibrations, so nobody rebuilds them: a ratio-to-own-median teleport test flags the STILLEST ids
(median 0.006 m/frame makes any motion a 40x outlier); a colour-flip test over the whole clip flags 21 ids
because the post-whistle crowd occludes torsos -- restricted to the live play it flags 2.

**Team colour is the identity to keep.** kit_margin in tracks.parquet is finite on 96 % of rows and
separates cleanly (BAL 0.09 / KC 0.88 positive share over 68 and 58 ids). Jersey OCR gave 3 of 13 Baltimore
ids a number and the roster path produced defenders drawn on offence. Every fragment created above takes its
team from its own kit majority, never inherited.

**THE JITTER IS IN THE LIMBS, NOT THE ROOT (2026-09-15).** Two hypotheses for the "super jittery" motion
were A/B'd on the drawn timeline (scratch probe_jitter_ab: root-xy second difference, m/frame^2, all
drawn body-frames, plus the census as the guard):

    config              | p50    p90    p99   | census play / full
    default             | 0.0028 0.0294 0.178 | 2.00 / 2.90
    no depth snap       | 0.0028 0.0258 0.157 | 1.98 / 2.94   (id 162 worse: 0.16 -> 0.44)
    no refit placement  | 0.0050 0.0540 0.249 | 2.14 / 2.97   (jitter DOUBLES: the refit pelvis is smoother than the box point)
    neither             | 0.0048 0.0379 0.171 | 2.22 / 3.02

Both rejected. The root's p50 is a real player's value (0.003); the p90 tail is five post-whistle ids
(169, 170, 76, 79 from 483+) and fragment 162. So the root is not what the user sees shaking.

The limbs are (scratch probe_pose_jitter: the renderer's own forward pass, pelvis-relative joints, live
play 300-460, 31 ids): joint jitter p90 **median 0.15 m/frame^2 across ids**, ten times a visible
twitch, with ids 162 / 165 / 9 / 4 at 0.7-0.9 and hands moving 0.4-0.6 m/frame (25-37 m/s) at the p90.
Every one of them is fused-posed, keyframes at **stride 2**, so the timeline is nearly the raw
per-frame fit; smooth_axis_angles (median 7, range-gated at 0.5 rad) keeps any component that turns
more than 0.5 rad in 7 frames RAW -- which is exactly what fit noise on a 5-pixel hand does. The
gate built to save a runner's arm swing also saves the noise. Next: per-joint breakdown (which limbs),
raw keyframes vs drawn (fit or smoother), and a smoother A/B on TWO rulers -- joint jitter and limb
reprojection onto the footage keypoints in both cameras (a frozen mannequin has zero jitter, so jitter
alone would crown the heaviest smoother).

The rulers are now code: `nfl_gsplat/render/motion_rulers.py` (contiguous step, root jitter, census,
joint jitter/speed, handover steps -- pure functions, tested) and `scripts/07l_measure_plausibility.py`
(builds the timeline as 05k renders it, prints one table, writes DIAG/<play>_<tag>_plausibility.json),
so v38 / v39 / v40 are one diff.

**The fix: a Gaussian on the limbs (2026-09-15, commit 5b46e9d).** Per joint the jitter sits at the
extremities (wrists 0.12, feet 0.11, ankles 0.09, elbows 0.07 m/frame^2 at the p90), and the raw stride-2
keyframes carry the same numbers as the drawn frames: the fit itself is noisy on every frame, and the
7-frame median (gated or not) cannot remove noise that is on every frame. Smoother A/B on TWO rulers,
live play, 30 ids -- joint jitter p50/p90/p99 and limb reprojection onto the footage keypoints (px,
sideline p50/p90, endzone p50/p90):

    raw               0.035/0.261/1.38   8.6/20.2   5.9/12.3
    median 7 gated    0.040/0.231/1.38   8.9/20.2   6.1/12.9    (the old default: nothing)
    median 15         0.035/0.159/0.89   9.7/21.0   7.2/15.5
    gauss sigma 2     0.019/0.109/0.69   9.0/20.9   6.4/14.2    <- adopted (POSE_SMOOTH_SIGMA)
    gauss sigma 4     0.013/0.078/0.53   9.9/21.7   7.3/16.7    smear: +1.3 px p50, +4.4 px endzone p90

Verified on the real timeline by 07l (tag v41 vs v40): joints p50 0.037 -> 0.018, p90 0.229 -> 0.108,
p99 1.35 -> 0.69; steps, root jitter and census identical. The reprojection ruler is biased toward the
raw fit (the fit followed the detector's per-frame noise, so any smoother scores slightly worse against
the same noisy keypoints) -- it still separates repair (sigma 2) from smear (sigma 4). The sigma sweep
and the per-joint split (arms heavier than legs) were then measured on both rulers, jitter p90 /
endzone reprojection p90 px: s1.5 0.138 / 13.4, s2 0.109 / 14.2, s2.5 0.098 / 14.7, s3 0.090 / 15.5,
s2 with arms s3 0.099 / 14.7, s2 with arms s4 0.093 / 15.5. One trade curve: each 0.01 of jitter costs
~0.3 px of endzone p90 wherever the sigma goes, so the split buys nothing and sigma 2 stays. CLOSED.

**Renders (`diag/play_001_vNN_hifi_720.mp4`) and the 07l report each carries.** v39 = cut 19 + 08u +
smooth_xy fix. v40 = + 08t (live scope) + 08v + the Gaussian limb smoother = report v41. v41 = + hinge
clamp + ankle anchor + snap veto = report v44. v42 = + 08t to a fixpoint over the whole clip = report
v45. v43 (after the whole-play refit with hard hinges) = report v46, if v46 wins. The report numbers
are the truth about each render; the mp4 is what the user watches.

**"Joints in weird positions" is measurable too (2026-09-15).** On the drawn live play (3676 body-frames,
33 ids, after the Gaussian), the hinges do the impossible: R_knee hyperextended (< -15 deg) on 3.0 % of
frames with a minimum of -104 deg, L_elbow on 5.9 % (min -136), and bent SIDEWAYS (> 35 deg off the hinge
axis) L_knee 11.9 %, R_elbow 14.2 %, L_elbow 7.3 %. Collars turn 146-176 deg at the p99 (a clavicle does
30); spine segments 116-126. It is concentrated on ids 4, 9, 165, 17, 11, 38 -- the same men whose limbs
jitter most, i.e. stretches where the fit is garbage rather than noisy. Hinge axes used: knee flexion
about +x, elbow about y (R +, L -) in SMPL-X's rest pose.

The clamp, measured before the Gaussian on both rulers (jitter p50/p90/p99; limb reprojection px
sideline p50/p90, endzone p50/p90; violation shares after):

    gauss s2 (no clamp)             0.019/0.114/0.73   9.0/20.9   6.4/14.2   2.8 % / 10.4 %
    hinges + collars, off 20        0.018/0.098/0.66   9.7/23.7   7.1/19.4   0 / 0
    hinges + collars, off 30        0.018/0.100/0.66   9.7/23.7   7.0/19.0   0 / 0
    hinges only, off 20             0.019/0.107/0.74   9.3/21.1   6.5/15.9   0 / 0     <- adopted (off 25)

The COLLAR clamp is the smear (+2.8 / +5.2 px at the p90): the regressor places the whole arm through
the clavicle, so a limit there moves the arm off the keypoints. Hinges alone cost +0.2 / +1.7 px at
the p90 with the p50 unchanged and take every backwards or sideways knee and elbow to zero -- a limb
that could not exist is worth two pixels at the tail. `timeline.clamp_hinges` runs before the
Gaussian (which rounds its kinks); `motion_rulers.hinge_violations` reports the shares in 07l. The
collars and spine stay as the fit made them: their repair belongs in the fit (a pose prior), not here.

**The 21 live-play hops: three mechanisms, traced by stage (2026-09-15).** None is a camera handover.
Stage trace on ids 1, 40, 161 (raw sideline, raw endzone, sideline after the depth snap, drawn):

  1. **id 1 at 415-420**: the raw sideline point sits still at y +5.9; at 420 the DEPTH SNAP slides it
     2.0 m along its ray to y +3.0 with no endzone row of its own -- a same-team body happened to lie
     on the ray -- and the smoother spreads that into a 12-frame ramp at 0.2-0.35 m/frame. The snap is
     decided per frame; the correction a body needs along its ray varies slowly.
  2. **id 40 at 384-396**: endzone-only frames before its sideline span; the endzone's ground point
     walks 2.6 m along x (its depth axis) in 12 frames, 13 m/s, then the sideline picks him up 0.7 m
     away. The endzone's depth noise, drawn as motion.
  3. **id 161 at 368-369**: the raw SIDELINE point hops 1.2 m when the ankle keypoints stop being
     confident and the box bottom takes over; at 375 an outlier box adds 1.6 m more.

Fixes measured on four rulers (live / whole-clip steps > 0.25 m/frame, root jitter, census, and the
drawn lower joints reprojected into the ENDZONE against its keypoints -- the sideline's depth axis is
that camera's lateral axis, so a slide along the ray shows there and nowhere else):

    baseline                       live 21   full 214   root p90 0.0294   census 2.00   endzone 8.8 / 15.2 px
    box anchored to ankles (3)     live 19   full 196   root p90 0.0286   census 1.94   endzone 8.8 / 15.2   ADOPTED
    temporal snap, median +-6 (1)  live 28   full 217   root p90 0.0289   census 1.96   endzone 8.9 / 16.3   REJECTED

The anchor (`play_timeline.anchor_boxes_to_ankles`): a box point takes the id's own median (ankle -
box) offset over its ankle frames within 15, with 3+ of them; 2462 box frames moved by p50 0.31 m
(exactly the box's known bias), p90 0.73. Better on every ruler, worse on none. The temporal-median
snap replaced every per-frame correction with a windowed median and filled unsnapped frames: the
piecewise medians step where the window's membership changes, and it moved bodies the per-frame
gates had rightly left alone. The veto-only variant -- keep every per-frame snap, drop the one more
than 1.0 m from the median of the same body's other snaps within +-6 frames (2+ of them), or alone
in its window and over 1.0 m -- was then measured on top of the anchor:

    anchor alone                   live 19   full 196 (1 over 0.6)   root p90 0.0286   census 1.94   endzone 8.9 / 15.1
    anchor + snap veto (1)         live 15   full 186 (0 over 0.6)   root p90 0.0278   census 1.94   endzone 8.9 / 15.1   ADOPTED

110 of 4635 snaps vetoed (2.4 %, the documented wrong-man share); the last step over 0.6 m in the
whole clip (id 76 at 637) goes with them. `depth_snap.veto_outlier_snaps`, VETO_WINDOW 6 / VETO_M 1.0.
One id pays: fragment 162 loses its lone snaps and its root jitter p90 goes 0.18 -> 0.44 (it was the
id the no-snap A/B had shown depends on the snap); the aggregates all improve, so it shipped.

The anchor's two knobs swept on the same four rulers (id 161's hop at 370 survived on one ankle
frame): w15 s2 = shipped; w15 s1 live 14 / full 188 / census 1.95; w30 s3 live 14 / full 182 / census
1.95 / endzone p90 15.3; w30 s1 live 13 / full 172 / root p90 0.0269 / census 1.96 / endzone p90 15.3.
Steps improve, both guards move the wrong way by a hair (+0.02 census, +0.2 px). Not adopted;
w15 s3 stays. CLOSED.

Mechanism 2, REJECTED: a Gaussian on x inside each id's endzone-only runs (sigma 4 and 8, 5869
frames touched) -- live 19 -> 19, full 196 -> 195, sigma 8 census 1.94 -> 1.97 and endzone p90 +0.3
px. The drift is a monotone 2.6 m ramp over 12 frames, which a smoother preserves by design; only a
velocity bound anchored at the sideline join could remove it, and it is two live steps on one id.

**Where the live play stands after all of it (07l, tag v44):** steps > 0.25 m/frame 15 (from 21;
none over 0.6 anywhere in the clip), root jitter p90 0.015, joint jitter p90 0.102 (from 0.229),
hinges 0 / 0, census 1.94. The largest remaining term is joint jitter on the ids whose fits are
garbage in stretches (162, 9, 4, 13, 17, 165 at 0.26-0.70) -- the case docs/PLAUSIBILITY.md reserves
for the local repair.

**08t to a fixpoint over the WHOLE clip (2026-09-15, 07l v44 -> v45).** Two things were wrong with
the first 08t pass: it reports one cut per id (the earliest), so the fragments it creates carry
switches of their own -- a dry run over the full clip found 25 more cuts, six of them INSIDE the live
play on first-pass fragments (165 @131, 167 @420, 169 @229, 173 @334, 161 @51, 175 @273) -- and it
was scoped to the live play on the argument that post-whistle hops are not worth fragmenting, while
four of the post-whistle switches were CROSS-TEAM tails (a KC id turning into a BAL man at 609, 615,
623, 566) drawn in the head's colour. Applied to a fixpoint (25 + 8 + 4 + 0 cuts, 3701 rows, 37 new
ids; 08v carried 378 fused + 203 sideline posed frames):

    v44 (live scope, one pass)   live 15   full 186   census live 1.94   root p90 0.0278
    v45 (whole clip, fixpoint)   live  9   full 150   census live 1.66   root p90 0.0265   joints unchanged

The pipeline's `switches` stage now loops 08t until a pass cuts nothing, over the whole clip
(CUT_TO_FRAME narrows it); END_LIVE is only 07l's window now. 08t and 08v back up with a counter
(`tracking.relabel.backup_path`) -- the second apply had overwritten the first pass's `.pre08t`, the
only copy before any cut, and a manual `.pre08t_full` snapshot saved it.

**Hard hinge bounds INSIDE the fit beat the post-hoc clamp (2026-09-15, commit after 94b0be9).** The
soft range prior had lost to the reprojection twice; a box bound on the optimiser's parameters
(scipy trf `bounds`: knees/elbows flexion [-5, 150] deg, off-axis +-25) cannot. A/B on play 1's six
worst ids (9, 4, 165, 17, 13, 162; 05p --one-view-only, same keypoints, 993 frames each), scored on the
392 live keyframes -- violations on the RAW fits, jitter and reprojection after the shipped clamp +
Gaussian (px, sideline p50/p90; the endzone column is the one-view fit's own placement, ~60 px for
both, and only its direction counts):

    A shipped fit         hyperext 3.8 %  off-axis 27.7 %   jit 0.033/0.221/0.77   spd p90 0.27   side 13.4/23.9   end 64.9/95.2
    B hard hinges         hyperext 0.0 %  off-axis  3.7 %   jit 0.031/0.187/0.72   spd p90 0.24   side  9.4/18.8   end 53.5/94.1

Better on every ruler, INCLUDING the sideline reprojection by 4-5 px: a legal pose fits the keypoints
better than an illegal one clamped afterwards, because the other joints compensate while the fit
runs. `Mono2DConfig.hard_hinges` defaults to True (05p --no-hard-hinges to disable); the render-side
clamp stays as a belt for caches fitted before it. The whole play is being refitted with it
(poses_refit.json.pre_hard is the cache before).

**The whole-play hard-hinge refit: limbs win, root loses (07l v45 -> v46, 2026-09-15 20:43).** 5182
frames on 69 players in 26 min, reprojection median 16.4 -> 2.7 px. On the timeline: joint jitter p90
0.102 -> 0.090, p99 0.69 -> 0.55, speed p90 0.152 -> 0.135, hinges 0 / 0, census 1.66 -> 1.65 -- and
live steps > 0.25 m/frame 9 -> 19, root jitter live p50 0.0021 -> 0.0038, p90 0.0137 -> 0.0210 (id 11
walks 1.6 m over 403-407). Mechanism: place_from_refit moves each body to the refit's per-frame
pelvis, and the bounded fit's pelvis is noisier from frame to frame -- transl is solved per frame with
no memory, while a root moves smoothly. Not shipped as is: the pose gain must not ride on a root loss.
Being measured: per-id Gaussian on the refit's transl along its keyframe runs (s2, s4), an outlier
veto (> 0.5 m from the +-6-frame median), and placement off, on the four placement rulers.

**Scored where the render places it, the one-view hard-hinge cache LOSES (2026-09-15 21:05).** The
refit A/B had placed bodies at the fit's own transl (~60 px off in the endzone) and could not see this.
Same eight worst ids, live window, bodies at the TIMELINE's xy, both cameras (scratch
probe_cache_in_timeline):

    shipped (pre_hard)   hinges 6.4 % / 23.0 %   joint jit p90 0.34   root p90 0.022   endzone lower 16.4 / 32.1 px   sideline limbs 16.7 / 33.2
    hard1 one-view       hinges 0.0 % /  3.7 %   joint jit p90 0.27   root p90 0.033   endzone lower 52.5 / 70.1 px   sideline limbs 10.2 / 18.9

A bounded knee cannot fake the foreshortening a leg pointed at the sideline camera produces, so the
fit lays the leg along the ray instead -- invisible to the sideline, three times worse in the endzone.
The pelvis itself moved only 0.06 m (p50) between caches; it is the legs. "Place off" leaves the
endzone at 22 / 53 px, so no placement smoothing fixes it. poses_refit.json is back to the v45 cache
(the hard-hinge one kept as poses_refit.json.hard1). The bound stays on by default in the FIT because
the same experiment with the endzone in the objective (05p --two-view --endzone-weight 0.3, refit C)
is the one that can resolve the along-ray ambiguity; it is being scored the same way. If C loses too,
hard_hinges goes back to opt-in.

**Hypothesis under test: a stronger per-camera pose model (NLF, NeurIPS'24) instead of the regressor
we refit from (2026-09-15, evening).** Of the multi-camera toolkits the user listed (Pose2Sim, Anipose,
MVPose, VoxelPose, EasyMocap, OpenCap, MeTRAbs...), only two touch what is still wrong on play 1: a
learned pose prior in the fit (EasyMocap's VPoser route; the weights are licence-gated and not on
disk) and a better per-camera 3D initialiser (MeTRAbs/NLF). The rest re-do triangulation and
association, whose ceilings are measured (geometry 0.15 m with identity given, the endzone's 4-6 deg
lens, a 1 m pairing ambiguity in a formation), or are trained on people 2-5 m from the camera.

NLF-L (`data/models/nlf/nlf_l_multi_0.3.2.patch4.torchscript`, 521 MB, noncommercial research
licence, loads under smplx312's torch 2.11 after `import torchvision`) takes OUR boxes and OUR camera
(`estimate_smpl_batched(images, [boxes xywh], intrinsic_matrix, extrinsic_matrix in mm,
world_up_vector, model_name="smplx")`) and returns SMPL-X pose (165), betas, trans, joints3d in world
millimetres, joints2d and per-joint uncertainties. First frame ran: 15 s per box on CPU. One box per
call -- a batch with one degenerate crop went non-finite inside its fitter. Probe
(scratch probe_nlf.py) scores hinge violations, limb reprojection onto the YOLO keypoints and
stride-4 joint jitter on the worst ids against the shipped refit (3.8 % / 27.7 % violations,
13.4 / 23.9 px) and the hard-hinge refit (0 / 3.7 %, 9.4 / 18.8 px). Runs on the GPU after v42.

Session note: the previous session died with three jobs running (v42 at frame 61, the whole-play
hard-hinge refit, the suite); v42 resumed, the refit restarted from scratch (its
`PermissionError: [WinError 5]` was the pool losing its parent, not a bug), the suite is queued.

**NLF, scored the same way: REJECTED as a pose source (2026-09-15 21:05).** On the GPU it is fast
(188 boxes in 69 s after warm-up) and good on the visible men in its own camera (id 4 6.0 px, 9 9.0, 13
8.8 at the p50 against the YOLO keypoints), garbage exactly where ours is (id 17 in the pile: 57 px,
its own uncertainty 1.5 m), and 4.3 % hyperextended knees. At the timeline's placement its legs are
WORSE in the endzone than the shipped fit (lower joints 29.6 / 69.5 px vs 14.5 / 34.0) and no better
in the sideline (17.9 vs 16.3). Same lesson as the hard-hinge cache: a monocular pose at 140 px lays
the legs along the camera ray, and only the other camera can say where they are. Kept as a possible
INIT for the fit (not measured); not a replacement for it.

**Pairing by formation position (the user's question, 2026-09-15 21:15).** It is what 08r does. At the
snap (frames 296-304, median ground points, same team) 13 of 22 are already paired at 0.43 m p50 and
every one of them is CLEAR by position alone (best <= 1 m, runner-up >= 2x further); position finds ONE
more, BAL sideline 4 <- endzone 198 (0.42 m, clear both ways), which 08r rejects as a re-pairing
(endzone 4 already holds a track on 29 of those frames -- --allow-repairing --give-up-incumbent would
apply it). Everything else is the KC interior (9, 19, 37, 38, 74, 86, 97, 164, 166, 204 at x -22..-23,
y -5..5): ratios 1.0-1.3 at 0.5-1.5 m -- the line stacked along the sideline's depth axis, where each
camera's blind-axis error (0.3-0.5 m sideline, ~1 m endzone) exceeds the spacing. No label changes
that. Scratch probe_snap_pairing.py.

**Refit C, two-view + hard hinges, WINS at the timeline's placement (2026-09-15 21:18, last result
before shutdown).** Eight worst ids, live window, bodies at the timeline's xy, both cameras:

    shipped (pre_hard)   hinges 6.4 % / 23.0 %   joint jit p90 0.341   root p90 0.0219   endzone lower 16.4 / 32.1   sideline limbs 16.7 / 33.2
    hard1 one-view       hinges 0.0 % /  3.7 %   joint jit p90 0.271   root p90 0.0334   endzone lower 52.5 / 70.1   sideline limbs 10.2 / 18.9
    C two-view ez0.3     hinges 0.0 % /  2.0 %   joint jit p90 0.204   root p90 0.0174   endzone lower 20.0 / 28.7   sideline limbs  9.5 / 20.7

With the endzone in the objective the bound stops laying legs along the sideline ray: joint jitter
-40 %, root jitter -20 %, sideline limbs -7 px, endzone p90 -3.4 px, for +3.6 px on the endzone p50
(inside the endzone keypoint noise on 60-px bodies). Ship it. hard_hinges stays default on.

**v47 = the whole play refitted two-view (endzone 0.3) with hard hinges (2026-09-16 01:50; 96 min,
2244 two-view frames on 29 players at sideline 2.5 / endzone 12.1 px, 2865 one-view records on 65).**
Against v45 on the timeline: joints p90 0.102 -> 0.090, p99 0.69 -> 0.44, speed p90 0.152 -> 0.127;
root jitter live p90 0.0137 -> 0.0128, p99 0.097 -> 0.081; census 1.66 -> 1.65; hinges 0 / 0; live
steps 9 -> 11 and whole clip 150 -> 154, every new one on fragment 162 (root jitter p90 0.30 -- the
26-frame tail of track 19 that sits 0.07-0.6 m from id 165/196/204 and the ankle-ray twin test cannot
see). Ships if the cross-view scorer agrees (running); 162 is the next thing to remove, not a reason
to hold the cache.

A "short fragment riding another body" rule was measured for 162 and found nothing to drop: at
<= 40 drawn frames, nearest same-team body under 0.6 m on >= 70 % of frames, no id qualifies (162: 53 %,
median 0.57 m to id 11; 39, 153, 167: 0-38 %). Loosening it to catch 162 alone is a rule for one id.
Not built. (scratch probe_short_fragments.py)

**v47 SHIPS (2026-09-16 02:25).** The full two-view cache reproduces C's cross-view numbers on the
worst ids exactly (hinges 0 / 2.0 %, joint jit p90 0.204, root 0.0174, endzone lower 20.0 / 28.7,
sideline limbs 9.5 / 20.7). poses_refit.json = the two-view hard-hinge cache (v45's kept as
.pre_twoview); the pipeline's ENDZONE_WEIGHT default is 0.3; render v43 = report v47.

**Handover level-match REJECTED (2026-09-16 02:30).** Shifting each endzone-only run by the
(sideline - endzone) offset at its join (62 runs, 415 frames; blended when both ends touch): whole-clip
steps 154 -> 154 (handovers 11 -> 7), live 11 -> 12, root p90 0.0257 -> 0.0267, census 1.65, and the
shifted bodies' endzone lower-joint reprojection p90 60 -> 92 px -- the run leaves the endzone's own
view of the man to meet the sideline. The handover jump is the two cameras' honest disagreement; a
constant shift trades a step for a wrong placement. (scratch probe_handover_level.py)

**Bodies per sideline box -- the pile counted without a radius (2026-09-16 02:32).** Every drawn
body's pelvis projected into the sideline image and assigned to the smallest detection box holding it:
same-team bodies per box {1: 2890, 2: 251, 3: 40, 4: 9} over the live play, 148 of 161 frames with a
doubled box, 390 surplus body-frames, the surplus ids the KC interior (166, 19, 37, 38, 11, 204).
Removing the surplus takes KC 11.06 -> 8.89 (census 1.65 -> 2.47): the doubled boxes hold two REAL
men, a lineman behind another in the sideline's merged box. So the census's open lead ("KC 13 at f425
while ~10-11 on the plate") is not duplication the box test can find; the drawn count is right within
the box merging. CLOSED. (scratch probe_bodies_per_box.py)

**The orientation gets its Gaussian too (2026-09-16 02:50, 07l v47 -> v48).** global_orient had kept
the 7-frame median after body_pose got sigma 2 -- never measured. A/B on the drawn live play, all ids
(joint jitter max over joints; yaw second difference; limbs reprojected in both cameras):

    median 7 (shipped)     joint jit 0.019 / 0.090 / 0.44   yaw jit p90 2.73 deg   side limbs 6.8 / 15.0   endzone lower 11.3 / 18.8
    gauss sigma 2          0.013 / 0.062 / 0.41   1.32   6.8 / 15.6   11.3 / 19.6
    gauss sigma 4          0.013 / 0.058 / 0.27   0.53   7.1 / 16.0   11.4 / 19.9     <- ORIENT_SMOOTH_SIGMA
    median 7 + gauss 2     0.013 / 0.063 / 0.41   1.15   6.8 / 15.7   11.3 / 19.4

Same trade curve as the limbs (~0.3 px of p90 per 0.01 of jitter), and the p99 -- where the twitching
lives -- drops 38 % at sigma 4 for ~1 px on either camera's p90 with the p50s untouched. Whole
timeline (v48): joints p50 0.019 -> 0.013, p90 0.090 -> 0.058, p99 0.44 -> 0.27; steps, root jitter,
census and hinges identical. Render v43 (in flight) carries v47; the next render carries this.

**The post-whistle steps have no single mechanism (2026-09-16 02:55).** 141 steps over 0.25 m/frame
after frame 470; the top 30 traced by stage: gap fill / endzone-only 9, raw sideline hop 9 (the box
point itself moving 0.4-1.1 m/frame in the crowd -- the detector's box sliding between milling men),
placement / smoothing 7, handover 4, snap toggle 1 -- in five clusters (170 @515-521, 37 @569-577,
185 @651-656, 79 @542-544, 212 @602). Nothing generic to fix; every one is the crowd after the play.
The clean product decision is to END THE RENDER at the tackle (~frame 500 on play 1: the play is
over, the crowd's artefacts are the worst in the clip) -- not taken unilaterally; the whole clip is
still rendered. (scratch probe_postwhistle_steps.py)

**Fit sweeps at the timeline's placement, eight worst ids, live window (2026-09-16 03:00).** Endzone
weight (timeline with the median orientation): 0.3 jitter p90 0.204 / endzone 20.0-28.7 / sideline
9.5-20.7; 0.5 0.260 / 16.2-25.9 / 10.3-22.0; 1.0 0.246 / 13.5-23.9 / 11.4-25.6. More endzone buys
endzone reprojection by selling the sideline and adding jitter (its keypoints are noisier); the render
camera sits on the sideline's side, so the sideline is the better proxy for what is seen. 0.3 stays.
CLOSED. Temporal weight (timeline with the orientation Gaussian): 0.3 jitter p90 0.144 / endzone
19.8-28.0 / sideline 9.8-22.2; 1.0 0.095 / 19.9-30.4 / 10.0-22.8; 3.0 0.079, off-axis 2.0 -> 0.5 % /
19.9-29.7 / 10.1-24.0. Minus 45 % of jitter for ~1.8 px at either p90 with the p50s unchanged: the
pull toward the previous frame is the term that holds a limb where the keypoints are noisy.
Candidate default 3.0, to be confirmed on the whole play before it ships.

Session note (2026-09-16 09:30): the machine went quiet ~03:30 with the v44 render at frame 101, the
whole-play temporal-weight-3.0 refit unfinished (poses_refit.json untouched = the v47 cache) and the knob
scorer short of its three new rows. All three relaunched 09:30 (05k resumes from its frames on disk).

**Three fit knobs at the timeline's placement, same recipe (two-view 0.3, bounds, temporal 0.3), eight
worst ids (2026-09-16 09:45).** C baseline: jitter p90 0.144 / endzone lower 19.8-28.0 / sideline limbs
9.8-22.2. `--unseen-temporal-mult 5` (a limb no camera sees holds its pose): 0.114 / 19.8-27.9 / 9.7-21.6
-- 21 % less jitter at no cost on either camera, ADOPT. `--lr-symmetric`: 0.148, off-axis 2.0 -> 5.6 %,
sideline 15.9-35.4, endzone p90 33.3 -- with the hinges boxed the mirrored assignment is not realisable
and the residual picks a bad mix; REJECTED. `--joint-reject-px 15`: 0.140 / 22.7-35.7 / 10.0-22.8 --
drops keypoints the endzone needed; REJECTED. Next: temporal 3.0 + unseen 5 together on the worst ids,
then the whole play with the winner.

**Footage review of v44 (2026-09-16 16:10-16:50, the loop's first pass).** Strips (05q, drawn body over
the real player with the detector's keypoints): id 9 (the runner, 340-354) shows the arms held out and
straight; frame by frame the drawn arms sit within 2-15 px of the detector's keypoints at every
KEYFRAME -- the arms are out in the footage -- and the excursions are the in-between frames (337:
30-55 px, a SLERP swing between two keyframes whose arm rotations differ wildly) and the keyframes
whose arm keypoints the temporal outlier filter removed (354: 29-51 px, nothing held the arm). Both are
what temporal weight 3.0 tightens. id 162's strip: its detections flip between a Chiefs lineman and
the Baltimore man beside him -- a duplicate under 08t's floor; `timeline.rider_ids` drops it (a short
fragment within 0.6 m of a teammate on half its frames; one instance on play 1, thresholds to be
re-measured on play 2). id 13 and id 4 (bent linemen) track their keypoints well; their spine past
90 deg is a real crouch, so the torso bound leaves 90 deg total. The Chiefs lineman drawn as
Baltimore is id 82 (kit vote 0.44, 1 m inside the KC line pre-snap): 08w re-teams him by the
formation. Two new rulers from this pass: limb speed on slow bodies (flailing: 5.5 % of standing
body-frames) and spine/collar angle shares (37 % / 36 %), scratch probe_flail.py.

**Torso bounds on top of temporal 3.0, at the timeline's placement, eight worst ids (2026-09-16 17:05):**
tw3 alone jitter p90 0.079 / endzone 19.9-29.7 / sideline 9.9-24.0; tw3 + hard torso 0.055 / 19.8-30.8 /
10.0-23.2, root unchanged, collars past 40 deg 12.4 % -> 0.1 % on the raw records. FINAL RECIPE:
`--two-view --endzone-weight 0.3 --temporal-weight 3.0 --hard-torso`; the whole play is being refitted
with it (poses_refit.json.pre_final = the v47 cache before). Render v45 (in flight) = the v47 cache +
id 82 re-teamed + the rider rule (162 gone) + the orientation Gaussian; v46 will carry the refit.
Review both with strips before any claim. Strip of id 82 after 08w (frames 220-276): drawn red over
the Chiefs lineman he is, keypoints matching where the detector sees him, the stance held through the
pile -- affirmed on the footage.

**Keyframe turns -- the ruler for the in-between-frame swing (2026-09-16 17:20).** Per joint, the
rotation between ADJACENT keyframes (stride 2) on the live play (scratch probe_keyframe_swings.py):
shipped cache p50 1.3 deg / p90 8.8 / p99 27.4, over 45 deg 0.25 % (id 9 carries 37 of them: elbows,
shoulders); the final-recipe partial cache (tw 3.0 + hard torso, eight worst ids) p50 0.4 / p90 2.5 /
p99 10.5, over 45 deg 0.03 %, over 90 none. A real joint turns at most ~20 deg in two frames, so the
shipped tail was fits disagreeing frame to frame, not motion; the recipe removes the cause of the
SLERP swing rather than guarding the interpolation. Whole play with the recipe (16:47, 2244
two-view frames at sideline 2.9 / endzone 12.4 px, joint speed p90 3.05 -> 2.04 m/s): p50 0.7 /
p90 4.9 / p99 14.4, over 45 deg 0.03 % -- the partial result holds across all 35 ids.

**v49 = the final recipe on the whole play + 08w + the rider rule (2026-09-16 17:00).** Against v48:

    live steps > 0.25    11 -> 7      whole clip 154 -> 141, none over 0.6
    root jitter live     p90 0.0128 -> 0.0119, p99 0.081 -> 0.062
    joints (live)        p50 0.013 -> 0.011, p90 0.058 -> 0.035, p99 0.27 -> 0.11; speed p90 0.121 -> 0.091
    hinges               0 / 0
    census               live 1.65 -> 1.50, whole clip 2.88 -> 2.51

The rider rule took ten short fragments across the clip (24, 29, 32, 59, 60, 72, 158, 162, 163, 194; 177
body-frames) and the census improved at both windows with them gone: they were copies. Render v46 =
this cache (chained behind v45). Cross-view scorer and the flail / torso rulers on the same cache are
the last gate before Mono2DConfig's defaults move to temporal 3.0 + hard torso. Flail / torso rulers on
it: flailing 5.5 -> 3.0 % of standing body-frames, a collar past 40 deg 1302 -> 0 frames, the spine
past 90 deg 1349 -> 1062 (the +-30 deg box lets a segment reach 52; a 20 deg box is being measured on
the worst ids, with the bent linemen's strips as the check that real crouches survive).

**Strips under the recipe (2026-09-16 17:25; cache = temporal 3.0 + torso, spine box 20 on the worst ids).**
Runner id 9 (340-354): the drawn arms follow the pumping keypoints in all eight frames, legs on theirs --
the v44 held-out arms are gone ON THE FOOTAGE, not only on the rulers. Bent linemen 13 (400-414) and 4
(360-374): the crouch survives the 20 deg spine box exactly as under 30 (the fold is at the hips) and
both track their keypoints. So the tighter spine box costs nothing visible; its numbers: spine total
p90 95 -> 87 deg, max 142 -> 101, past 90 on 11.0 -> 8.2 % of records, +0.2 px on the fit's own
reprojection. Cross-view scorer pending; if it holds, spine_max_deg 20 becomes the default.

**Spine box 20 DENIED by the cross-view scorer (2026-09-16 17:30).** Same six worst ids, scored at the
timeline's placement in both cameras (off-axis hinge share; joint jitter p90; endzone lower joints
p50/p90; sideline limbs p50/p90):

    temporal 3.0 + torso 30 (shipped)   0.9 %   0.0554   19.8 / 30.8   10.0 / 23.2
    + spine box 20                      3.3 %   0.0652   19.9 / 36.8   10.9 / 26.9

Worse on every column: with the spine boxed at 20 the fit fakes the crouch with the hinges (off-axis
0.9 -> 3.3 %) and the legs lie 6 px worse in the endzone. The strips were right that the crouch
survives; the scorer says the rest of the body pays for it. The fit's own +0.2 px understated the cost
by 30x -- the third time the fit's own reprojection has hidden what the other camera sees (memory
corrections-must-beat-what-they-correct, case 10). Mono2DConfig.spine_max_deg stays 30. The spine past
90 deg on 8-11 % of records is left as real crouching until a strip shows one that is not.

**v45 delivered (2026-09-16 17:25):** the pre-snap line crop from the rendered frame 74 (scratch
review_v45/line_30.png) shows every red shirt on the Kansas City side and no white one inside their
line -- id 82 (08w) is red on the footage, affirmed. 720p clip in diag as play_001_v45_hifi_720.mp4.

**FOUND ON THE FOOTAGE: the smoothers break at a half turn (2026-09-16 17:50).** The loop's first
footage-first pass on the two flailers the ruler still names (ids 0 and 38, 18-19 % of their standing
frames with a hand or foot past 0.10 m/frame) went: flail frames listed per id (scratch
probe_flail_frames) -> 05q strips at those frames -> the drawn legs leave confident keypoints. Id 0
(BAL 21, standing still at the snap, sideline keypoints at confidence 1.0 on every leg joint, no endzone
view): drawn legs splayed 45 deg on frames 302-312 and again 366-374 while the CACHE's own legs
reproject 3-5 px onto the same keypoints (scratch probe_id_legs: lower joints p50/p90/max 3.8/4.5/4.7
px over 280-340; 3.1/6.4/8.2 over 356-390). Id 38 (KC 55, a lineman): drawn legs collapse on 446-458,
cache 4.4-6.5 px. So the damage is made AFTER the fit, in the timeline, and it is smooth.

Mechanism: a body with its back to the sideline camera has |global_orient| near pi (id 0: 2.99 ->
3.19 rad over 286-322, crossing pi at 302-308). The keyframe SLERP (interp_axis_angle) returns
scipy's canonical vectors (|v| <= pi), so the vector FLIPS SIGN at the crossing, and the component-wise
orientation Gaussian (sigma 4, shipped v48) averages antipodal vectors: the timeline's |go| collapses
3.1 -> 0.64 and the yaw swings 87 -> -53 -> 178 -> 125 -> 90 deg over 300-314 (scratch
probe_orient_flip), sideline lower joints 30 px / upper 45 px against 3-5 px either side. The
body_pose Gaussian is exposed the same way in principle (joints never near pi in practice). The jitter
ruler cannot see it (a smooth 14-frame sweep), the hinge ruler cannot (the pose is fine, the frame is
wrong), the census cannot; only the footage and the sideline reprojection at the timeline's placement
did. Exposure on the whole clip (scratch count on the cache, keyframes canonicalised as the SLERP
does): 12.4 % of keyframes within 0.3 rad of pi, 46 sign flips; the three ids with the most flips are
9 (the runner, 11), 0 (10) and 38 (7) -- exactly the runner's flailing arms of v44 and the two flailers
left in v49. Each flip corrupts ~2 sigma frames either side.

Fix (commit 6d4c898): timeline.unwrap_axis_angles re-expresses each row as the representation
nearer the previous row (a rotation by a about u is a rotation by 2 pi - a about -u; the rotations are
unchanged) and build_timeline applies it to global_orient AND body_pose before the smoothers
(unwrap=True; False = v49). Tests: a sweep through a half turn is continuous after unwrapping; the
plain Gaussian on it is off by > 30 deg (negative control), the unwrapped one < 2 deg. A/B at the
timeline's placement on ids 0 / 38 and the play-wide rulers: below. Also seen in the cache: the fit's
own free vectors reach |go| = 1289 rad (205 turns, harmless to the rotation) and 14 keyframe pairs jump
> 3 rad in the raw vector (the two-view seed's representation vs the chain's; the temporal term at 3.0
fights those) -- a fit-side hypothesis for later, small population.

**Unwrap A/B at the timeline's placement (2026-09-16 18:05; scratch probe_orient_flip2) -- ADOPTED.**
Sideline reprojection of the drawn body (lower joints / upper limbs, px) and its yaw, v49 timeline
(no_unwrap) vs the fix (unwrap), on the frames the strips showed:

    id 0, 300-314   no_unwrap lower 5 -> 31 -> 6, upper 6 -> 45 -> 4, yaw 76 -> -53 -> 178 -> 90
                    unwrap    lower 3.0-4.8, upper 1.7-2.8, yaw 83-88 throughout
    id 0, 364-380   no_unwrap lower 8 -> 37 -> 8, yaw 89 -> -180 -> -53 -> 67
                    unwrap    lower 4.6-9.8, yaw 82 -> 71 (a real slow turn)
    id 38, 446-460  no_unwrap lower 12 -> 44 -> 36, upper 6 -> 48 -> 20, yaw 131 -> 171 -> -118 -> 44
                    unwrap    lower 4.9-9.6, upper 2.8-8.7, yaw 117 -> 96

Play-wide, every drawn body on 300-460 (n 3176): joint jitter p90 0.0352 -> 0.0326, sideline limbs
p50/p90 7.3/16.3 -> 7.1/15.0, endzone lower joints p50/p90 51.8/128.7 -> 51.8/129.5 (unchanged; that
ruler counts every body against whatever the endzone has under its id, so its level is not the six-id
scorer's). Better on every ruler the change can reach, neutral on the other camera: shipped as the
default (commit 6d4c898). 07l v50 and render v47 (chained behind v46; v46 is the v49 timeline and is
the before-footage) follow; the strips of ids 0, 38 and 9 from v47 are the affirm/deny.

Retracted (18:00, twenty minutes after writing it): I wrote here that the keyframe-turn ruler
(scratch probe_keyframe_swings, "turns > 45 deg 0.25 -> 0.03 %") differenced raw vectors and so
counted representation flips as 360 deg turns. Reading the probe: it measures body_pose joints as the
RELATIVE rotation's magnitude (ra.inv() * rb), representation-free, and never touches global_orient.
The temporal-weight number stands as measured. The lesson stays for any ruler that does difference
raw vectors; none of the shipped ones do.

Spine past 90 deg -- who and where (cache, live play, 18:20): every id with more than 11 such
keyframes stands within 2.3 m of the line of scrimmage (11, 12, 3, 17, 19, 166 on the Kansas City
side at 1.1-2.3 m; 13, 4, 1 on the Baltimore side at 0.6-1.0 m; 84) -- the two lines, crouching for
real, as the 13 / 4 strips showed. Off the line: the runner 9 (11 keyframes at 100-114 deg around the
handoff and the tackle) and 5 (two frames). The "29 % of frames past 90 deg" is the linemen; the
non-lineman spine hypothesis is DENIED by the population and closed.

Fit-side exposure (scratch probe_raw_jumps): 14 keyframe pairs on the clip jump > 3 rad in the raw
global_orient vector; 11 of them are pure representation jumps (|dv| ~ 6.28 = 2 pi, relative rotation
0-4 deg, the fit's own sideline error 3-8 px either side: the data term won and the temporal term's
2 pi residual cost nothing visible). The exceptions: the runner 9 at 326-338 (three jumps, sideline
13-18 px around them against 3 px outside), 28 at 472-474 (9 -> 27 px, post-whistle), 78 at 642-660
(a broken post-whistle track, hundreds of px, |go| 14 rad). A geodesic temporal residual (the relative
rotation's angle instead of p - prev) would remove the artefact; small population, needs a whole-play
refit and the cross-view scorer -- queued behind the v47 review.

**07l v50 = v49 + the unwrap (2026-09-16 18:15).** Only the joint rulers can move (the unwrap touches
rotations, not placement): jitter p50/p90/p99 0.0106/0.0354/0.112 -> 0.0104/0.0326/0.069, speed p90/p99
0.091/0.218 -> 0.085/0.141; steps (live 7, none over 0.6), root jitter, census 1.50 / 2.51, hinges 0/0
all unchanged. The p99s are the number to watch: a third of the worst joint motion on the live play
was the half-turn artefact.

Flail ruler under the unwrap (scratch probe_flail, v49 -> v50): standing body-frames with a hand or
foot past 0.10 m/frame 80 -> 48 of 2680 (3.0 -> 1.8 %); id 38 19 -> 7, id 0 28 -> 10, the runner's 2
gone; spine > 90 deg 1062 unchanged (the linemen), collars 0. What is left: 0 (10), 13 (8), 38 (7),
166 / 3 / 204 (4 each) -- next strips.

Strips under the unwrap (05q, sideline, scratch review_v50; the same frames as the v49 strips): id 0
at 300-314 -- the drawn skeleton sits on the keypoints in all eight frames, legs straight, the splay
gone; id 0 at 370-377 (frame by frame) -- the skeleton follows a real quick step with a teammate
crossing behind him, the drawn left leg a few px off the green at 372-376 and nothing like the
45 deg kick of v49. The two v49 flail windows of id 0 are AFFIRMED fixed on the footage. What the
flail ruler still counts on him (373-378 ankle 0.11-0.24 m/frame; 388-391 wrist 0.10-0.13) is that
step and a hand; on 38 and 13 the remaining wrist flags are 0.10-0.14 m/frame (3-4 m/s) hand-fighting
at the line, which the 410-424 strip of 38 already showed following the keypoints. The 0.10 m/frame
threshold now sits at the level of real hands; the flail ruler has done its job on this play.

Review tools promoted from the scratchpad (2026-09-16 18:15): `scripts/05v_render_strips.py` cuts
per-id strips from the RENDERED frames through the render's own follow camera (`--spec PID:START:COUNT:STEP`,
pass the 05k eye offset / fov) and contact sheets (`--sheet FRAMES`); `scripts/07m_measure_fit_reprojection.py`
prints a pose cache's own per-frame reprojection of one id (lower joints / upper limbs px, knee angles) --
read it beside the 05q strip of the same frames: if the cache is on the keypoints and the drawing is not,
the timeline made the damage. Tests in tests/test_review_tools.py. The loop per flagged id is now: flail /
07l names the id and frames -> 07m (is the fit right?) -> 05q strip (footage) -> 05v strip (render) ->
affirm/deny.

**Geodesic temporal residual: measured, NO EFFECT, closed (18:20).** Ids 9 / 0 / 38 / 28 refitted with
`--temporal-geodesic` (505 records) against the shipped cache, cross-view scorer at the timeline's
placement: joint jitter p90 0.0448 vs 0.0449, endzone lower p50/p90 13.8/20.4 vs 13.7/20.3, sideline
limbs 7.6/15.2 vs 7.7/15.4; the runner's own sideline error on 316-344 identical frame by frame (07m).
The 2 pi residual was harmless because the data term wins it every time. `Mono2DConfig.temporal_geodesic`
stays False; the raw jumps stay in the cache as a curiosity.

**v47 delivered and AFFIRMED on the render (19:15).** v46 = the v49 recipe (07l v50's cache, the
v49 timeline), v47 = the same cache with the half-turn unwrap; 720p clips in diag. Per-id strips cut
from both renders through the follow camera (scripts/05v, scratch review_v46 / review_v47/strips):
id 0 (Baltimore 21) at 300-314 -- v46 spins him through a full garbage turn over twelve frames
(side-on, bent double, crouched, back again), v47 shows a man standing with his back to the camera
in all eight frames. Id 38 (Kansas City 55) at 444-458 -- v46 has him collapse to the turf and lie
with his legs up, v47 has him upright in a blocking stance stepping sideways. These are the two
bodies the user's "funky angles, flailing" would have named on this stretch, and the fix is on the
footage, not only on the rulers.

The runner at the handoff (05q strips 322-336, both cameras): sideline -- the drawn body follows the
keypoints with a whole-body offset of ~15 px on 324-334 (a motion-blurred 100-px body between the
linemen); endzone -- id 9 has endzone rows on 322-324 only, and there the green keypoints sit on a
BLURRED figure passing between two linemen (the runner at speed, half-hidden), then nothing until 336
where his placement projects behind the quarterback. Hypothesis: the endzone term (weight 0.3) on those
two blurred endzone frames, carried by the temporal chain, is the 13-19 px; test = refit id 9 one-view
only and read 07m on 316-344 (running).

Result (18:30): one-view only, the runner's own sideline error on 324-336 drops 13-19 -> 3.5-9 px (lower)
and 4-17 -> 1.5-4.5 (upper) -- but his knees go 62-115 deg -> 3-27 deg: a sprinter drawn with straight
legs laid along the sideline ray, the one-view failure the cross-view scorer closed on 09-15. The endzone
has two blurred frames here and nothing to score the legs against. Neither cache oscillates the knees
stride by stride, so both are the 2D ambiguity; the two-view stays (its legs bend, its 15 px is the
price). NOT adopted; the runner at the handoff is left as it is.

**Live hops on the footage (05q strips at 07l v50's two worst live steps, 18:25).** Id 40 (BAL) at
387-394: the drawn body stands on EMPTY TURF between two Baltimore men with no green keypoints -- the
sideline has no detection of him on those frames (scratch probe_unseen_frames: seen by the endzone
only), so his ground point is the endzone's own foot point, which is blind along the field's long
axis (x) at 88 m; the hop (0.30 m/frame at 390-394) is that point wandering. Id 203 at 366-373: an
eight-frame fragment on the Kansas City line with NO TEAM (drawn in the default kit -- yellow in 05q),
sideline-only, its skeleton offset from the pile. Population on the live play: 3615 body-frames, 153
endzone-only (ids 40: 33, 37: 36, 38: 30, 17: 18, 74: 16, 198: 8, 15: 5), 8 teamless (203 only), 93
drawn with no detection in either camera (gap fill; the long ones are 153 gliding 3.15 m over 435-454
and 4 gliding 2.13 m over 444-457 in the pile, both under the 30-frame bridge). None of the six worst
live steps is on an unseen frame: they are endzone-only (40) and sideline-only fragment (203) frames.
Hypotheses: (a) on endzone-only frames slide the endzone point along the ENDZONE's own ray to the x
interpolated from the id's sideline-seen frames (its projection into the endzone is unchanged, the
blind axis comes from the camera that sees it) -- rulers: live steps, endzone lower joints (must not
move), census (must not move), metres moved; (b) a teamless fragment shorter than the rider length is
not drawn; (c) the gap bridge 30 -> 10 frames for the two gliders (measure the census cost).

(a) built as `render/blind_axis.hold_blind_axis` (slide along the endzone's own ground ray to the
sideline-interpolated x; the endzone image of the body is unchanged by construction), wired into
load_play_timeline (`blind_axis=`), unit-tested. First A/B (18:35, per-frame cap 4 m, one-sided hold
within the 30-frame window): 635 body-frames slid on the clip (median 0.60 m, max 3.27); live steps
7 -> 9 -- WORSE; census 1.50 -> 1.50 with both teams nearer eleven (KC 10.98 -> 11.08, BAL 11.42 ->
11.32); endzone lower p50 51.8 -> 51.3, sideline and jitter unchanged. What happened: id 40's
387-395 ghost is GONE -- slid onto the man he is, the dedupe box then takes him as the copy he was
(a twelfth Baltimore body), which is the census gain; but a per-frame cap let an eight-frame
endzone-only fragment (198, 453-456) flicker on and off the slide at 0.4 m/frame, and a one-sided
hold pinned it to a sighting a second old. Rebuilt: a run of endzone-only frames slides whole or not
at all (its median slide against the cap) and a one-sided hold reaches 6 frames, not 30; re-measuring.
(b) built as `timeline.orphan_ids` (a teamless id drawn on <= 40 frames is dropped; census unaffected
by construction), unit-tested, wired after the rider rule; measured with (a)'s rerun.

Rerun (18:50): with (b) in both arms, the per-run hold still LOSES -- live steps 5 -> 7, census
1.50 -> 1.60 (KC 10.98 -> 11.09), endzone/sideline/jitter unchanged, 611 body-frames slid -- and id
40's path on 385-395 moves by only 0.1-0.3 m: the endzone's own x drift there (-27.1 -> -25.2 in ten
frames) agrees with the sideline sightings on both sides, so he is RUNNING at 6 m/s and the ghost on
the strip is a lateral offset the hold cannot touch. (a) REJECTED, kept opt-in
(`load_play_timeline(blind_axis=True)`) with its numbers in the module. (b) ADOPTED: live steps
7 -> 5 (203's two gone), census unchanged. Commit 6cb2a1a.

The absolute step ruler counts sprinters: four of the five live steps left are id 40's strides at
0.26-0.33 m/frame (8-10 m/s, a defensive back in coverage) and none of the five is a step that
disagrees with its neighbours. New ruler `motion_rulers.jerk_steps` (commit c299ddb): a contiguous
step more than 0.15 m longer than the median of its neighbouring steps (3 each side) -- a hop, not a
stride; 07l prints it beside the absolute count (which stays for teleports). 07l v51 (orphan rule in)
follows; the hop count on the live play is the number to carry from here.

**07l v51 = v50 + the orphan rule (18:55).** Live steps > 0.25: 7 -> 5 (203's two gone), every other
ruler identical to v50 (root jitter live p90 0.0119, census 1.50 / 2.51, hinges 0/0, joints p90 0.0326,
p99 0.069). With the hop ruler: hops full 38, LIVE 4 -- id 37 0.27 m (+0.21 over its neighbours) @376,
166 0.25 (+0.18) @300 (the snap), 74 0.23 (+0.17) @419, 17 0.18 (+0.15) @414; id 40's strides are
not among them. Those four frames are the next strips.

Id 40 on the footage, three windows (19:00). Sideline 377-384 (no sideline detection either): the
drawn body stands ~1 m along the field from the real man in every frame. Sideline 395-400: at 395-396
still off him; from 398 the sideline sees him and the body sits on him. ENDZONE 386-393 (where he IS
detected): the green keypoints are on the man and the drawn body is offset ~40 px up-left of them --
further along the endzone's depth and a little across -- so the endzone-only placement does not even
land on the endzone's own keypoints. The ~1 m is made somewhere between the endzone foot point and the
drawn body: the refit placement (place_from_refit, 1 m reach), the run smoother (smooth_xy over a run
that mixes endzone-only and sideline frames), or the foot point itself (a running man's lifted foot).
Measuring per frame at the timeline (both cameras' lower joints, default vs no refit placement).

Result (19:00, scratch probe_id_timeline_reproj): id 40's lower joints reproject 86-112 px off the
ENDZONE keypoints on every frame 374-400, endzone-only and two-view frames alike, and turning the
refit placement off changes nothing (identical to the pixel) -- the offset is in the ground point
itself or the run smoother, not the refit. Isolating placement from the legs' pose next: the drawn
pelvis against the keypoints' hip centre, per camera (scratch probe_pelvis_offset, ids 40 / 37 / 74 / 17).

The four live hops on the footage (05q strips, 19:00): 37 at 372-379 -- a lineman in the pile, the
sideline keypoints steady on him while the drawn body sits 30-40 px off and drifts, the hop at 376
is the drawing moving, not the man (a two-camera id: 41 both / 44 sideline / 36 endzone frames).
74 at 415-422 -- drawn ~60 px left of the pile with no sideline keypoints until 421, then still
60 px off them: an endzone-only stretch handed to the sideline with its offset. 166 at 297-304 -- the
snap, a lineman firing off with the sideline keypoints gone; the body shifts a little. 17 at 410-417
-- keypoints on him to 413, then none; the crop jump at 415 is 05q switching from the box to the
projection for its crop centre, the body moves 0.18 m. All four are placement offsets of 0.2-0.3 m
at camera-source transitions in the line or the pile: the hop population is now four frames of 3600
on the live play, none over 0.3 m. The thread worth pulling is the OFFSET: bodies drawn 30-60 px
off their own sideline keypoints on frames that have them, next to unseen stretches (37, 74). A
population probe is running: sideline-seen body-frames' lower-joint px at the timeline placement,
binned by distance to the id's nearest unseen frame.

Result (19:05, scratch probe_seen_offset; 3185 sideline-seen body-frames on the live play, lower
joints px at the timeline's placement):

    distance to the id's nearest unseen frame    n     p50    p90
    0-2 frames                                  216    10.1   22.7
    3-6                                         231     8.7   14.8
    7-15                                        285     7.9   12.8
    16+                                        1342     6.3   14.1
    the id is never unseen on the play         1111     4.6    8.5

The two frames beside an unseen stretch are twice as far off their own keypoints as a frame of a
body every camera always saw, and the error decays with distance: something carries the unseen
frames' offset into the seen ones. Suspects, in order: the run smoother (timeline.smooth_xy, window
9, averaging seen with unseen points), the gap fill, the ankle anchor window (15). A/B running with
the smoother replaced by the identity (bins, root jitter, hops, census).

Placement alone (19:10, scratch probe_pelvis_offset: the drawn pelvis against the keypoints' hip
centre, px, drawn minus keypoints): id 40 in the ENDZONE is 70-78 px ABOVE his own hips on every
endzone-only frame 374-396 (further along that camera's depth) and still 34-38 px above, 55 px left,
on the two-view frames 398-400 -- the depth snap inherits it. Id 37 (two-view, the hop at 376/377):
endzone dx jumps 32 -> 67 px across the image between 376 and 377 while the sideline dy goes 20 -> 12
-- the body moved ~0.7 m along the sideline's depth in one frame: the depth snap changing its answer.
Id 74: endzone 45-50 px BELOW his hips (nearer the endzone camera) and 66 px right on 414-417, the
sideline 12 px below on 421-424. Id 17 (in the line, both cameras): 2-6 px, fine.

So the endzone-derived depth is off by ~1-2 m in both directions on running and piled men, and the
depth snap carries it onto the sideline's ray. Mechanism hypothesis: the endzone ground point is
the ANKLE keypoints' ray met with the ankle plane (ankle_ground, z = ANKLE_Z_M). That camera sits
low behind the goal line (~6 deg elevation at 88 m), so a foot lifted 0.4 m in a stride puts its
ground point ~3 m too deep, and a foot in a pile the other way; the sideline at ~24 deg is four
times less sensitive. The HIPS sit at ~0.95 m running or standing -- a 0.1 m spread -- so an
endzone ground point from the hip centre met with the hip plane would err ~1 m at worst instead of
3. Test: endzone target points from the hips (z = HIP_Z_M) instead of the ankles, everything else
unchanged; rulers = the endzone pelvis offset over every frame with endzone hips (p50/p90, the
population ruler of this defect), the sideline pelvis offset (must not move), live hops, census,
root jitter.

Smoother A/B (19:15, scratch probe_seen_offset_ab): with timeline.smooth_xy replaced by the identity
the bins barely move (0-2 frames: p50 10.1 -> 9.3, p90 22.7 -> 21.3; never-unseen 4.6 -> 4.7) while
root jitter live p90 goes 0.012 -> 0.043 and live hops 4 -> 67. The smoother is NOT what carries
the offset -- the ground points beside an unseen stretch are already off before smoothing, which
is what the endzone-depth hypothesis predicts (a two-view frame beside an endzone-only stretch
takes its depth from the same endzone foot point). DENIED; the smoother stays.

Lowest-ankle foot point A/B (19:25, scratch probe_lowest_ankle; the defect's own ruler = the drawn
pelvis against the endzone keypoint hips over every live frame that has them, n 1889): shipped mean of
both ankles |d| p50/p90 43.4/121.6 px; lowest ankle for the endzone 43.9/121.6; for both cameras
43.4/121.6 -- NOTHING moves, and census / root jitter get a hair worse (1.50 -> 1.53/1.55, p90 0.0120
-> 0.0126/0.0129). DENIED: the lifted stride foot is not the mechanism. The population number is the
finding: the drawn bodies sit 43 px from their own endzone hips at the median and 122 at p90, against
4.3 / 9.9 in the sideline -- a general endzone misplacement, not a few running men. Next: is it a
BIAS (one direction: the endzone camera itself, or a systematic depth rule) or scatter (per body)?
Vector statistics by views / team / depth band / id running.

Bias vs scatter (19:35, scratch probe_endzone_bias, 1889 live body-frames with endzone hips): the
overall median vector is small (dx -15, dy +6 px) but only 29 % of frames are within 20 px and 47 %
within 40 -- scatter, not one bias. It is structured: by team Baltimore dx -34 / Kansas City -1; by
field position the Baltimore side of the line (x < -26) dx -45 to -50 while the Kansas City side
(-26..-22) is +1 -- a horizontal shift in the endzone image is a y error in the field (the sideline's
DEPTH axis), so bodies on the defence's side stand ~1 m off along the sideline's depth while their
sideline hips match at 4-6 px. Two-view frames (n 1616) carry it (dx -17.5, |d| p50 42.5), so the
depth snap is not curing it there. Per id: 9 is 528 px off in the endzone (its endzone rows are
another man -- a mispair), 74 / 40 / 0 / 7 / 2 at 54-88 px with sideline 3-6 px; 3 / 11 / 5 / 13 / 15
at 8-17 px are fine. Next ruler: triangulate the hip centre on two-view frames (both cameras' hip
keypoints) and compare with the timeline's xy in field metres, by team and band -- if the y
disagreement is the ~1 m, the sideline's depth (foot point + snap) is the mechanism and the
triangulated hips are the fix for paired frames.

Gap bridge A/B (19:35, scratch probe_gap_bridge): bridge 30 (shipped) drawn 3607 live body-frames,
93 unseen by either camera; bridge 10: 3561 / 53, live hops 4 -> 4, steps 5 -> 5, census 1.50 ->
1.32, root jitter p90 0.0120 -> 0.0116; bridge 4: hops 6, steps 12, census 1.36 -- too short. Bridge
10 wins on every ruler measured; the ruler it lacks is POPS (a body vanishing and reappearing),
which no ruler counts yet -- counting them for 30 vs 10 before adopting.

Pops (19:50, scratch probe_pops; an id drawn, absent 1-30 frames, drawn again, live play): bridge 30
= 34, bridge 10 = 32. Not worse on the one ruler it lacked, better on census, jitter and unseen
body-frames, equal on hops and steps: bridge 10 ADOPTED as `timeline.FILL_GAP_FRAMES` (commit
a11555b; the span and anchoring rules keep MAX_GAP_FRAMES 30). Goes into 07l v52 with whatever the
hip A/B decides.

**Hip triangulation vs the timeline (19:45, scratch probe_tri_hips).** 1717 live two-view frames
have a confident hip pair in both cameras; the hip centre triangulated from the two rays lands at
0.77 m (p10 0.60, p90 0.92 -- crouching to standing), the rays pass within 0.13 m of each other at
the median, and the point reprojects onto the hips at 5.3 px (sideline) / 8.4 px (endzone). 1541
(90 %) pass a ray gap < 0.5 m and hip height 0.5-1.4 m. The timeline's point against them, field
metres: dx +0.04 (|dx| p50 0.07, p90 0.18), dy -0.10 (|dy| p50 0.20, p90 0.58); by side of the
line, Baltimore's side dy -0.33 to -0.38 (x < -26), Kansas City's +0.01; worst ids 74 +0.62, 0 -0.49,
7 -0.48, 2 -0.34. So the geometry is fine and the sideline foot point + depth snap is what stands
0.2-0.6 m off along the sideline's depth, one way on the defence's side. Built: render/tri_hips
(triangulated_hips with the two gates, place_on_triangulated_hips), wired into load_play_timeline
behind `tri_hips=` (off until measured), tests. A/B next: endzone pelvis offset (the defect's ruler,
expect 42 -> ~10 px on paired frames), sideline pelvis offset (control), endzone lower joints,
hops, steps, census, root jitter.

First A/B (20:00, scratch probe_tri_ab): 4483 body-frames placed on triangulated hips (median move
0.15 m, p90 0.34) -- and the endzone pelvis offset does NOT move (43.4 -> 43.6 px p50, 121.6 p90),
sideline 4.3 -> 4.3, while live hops go 4 -> 2 and root jitter p90 0.0120 -> 0.0117, census equal.
The ground point is not what is drawn: place_from_refit then replaces it with the refit record's
own transl on every fitted frame within 1 m (5038 body-frames on the clip), and the two-view fit's
transl is what stands 43 px off the endzone hips -- its endzone weight is 0.3 and its place term
pulls toward the OLD ground point it was fitted with. So the triangulated hips only reach the render
where the refit is absent. Next: keep the triangulated point on its frames (skip the refit
placement there) and measure again; the memory says the refit placement halves root jitter, so
that ruler decides with the endzone offset.

Second A/B (20:10, the triangulated point kept through the refit placement; both arms on the new
10-frame bridge): endzone pelvis |d| 43.4 -> 44.2 px (p90 121.6 -> 123.5), sideline 4.3 -> 4.9,
live hops 4 -> 7, steps 5 -> 9, census 1.32 -> 1.43, root jitter p90 0.0116 -> 0.0217. REJECTED
hard: 4483 frames on the triangulated hips beside 2853 on the refit's transl, the two sources 0.2-0.6
m apart, so the run alternates between them wherever a hip pair's confidence dips -- jitter doubles,
and the endzone number cannot fall while half the frames are still the old point and the smoother
blends the halves. Same shape as the ankle-vs-box switching of 09-15, same remedy: not the points but
a windowed MEDIAN offset (triangulated minus placed) per id, applied to every frame
(`tri_hips.anchor_ground_to_tri`, window 15, support 3), after the refit placement. Measuring.

Third A/B (20:25, the anchor): 5872 body-frames shifted by a MEDIAN OF 0.04 m (p90 0.09) -- and the
endzone pelvis offset is still 44.3 px, sideline 4.3 -> 4.9, hops 4 -> 6, steps 5 -> 9, census 1.32
-> 1.40, jitter 0.0116 -> 0.0122. REJECTED (three forms of tri_hips lost; the flag stays off). The
number that matters is the 0.04 m: at the point where the anchor acts (after the depth snap and the
refit placement), the placed ground points already agree with the triangulated hips to 4 cm at the
median -- yet the drawn bodies end 0.2-0.6 m from those hips. So the divergence is made AFTER that
point, inside build_timeline (gap fill, the run smoother, the dedupe) or in how the state's xy is
turned into a drawn pelvis. Capturing the ground handed to build_timeline and comparing it per frame
with the final xy and the triangulated hips for ids 7, 2, 0, 74 (running).

**RETRACTION (20:35): the endzone side of every scratch ruler written tonight was half a second off.**
The probe found ground -> final xy 0.01 m (build_timeline moves nothing) and ground -> triangulated
hip +0.29 m for id 7, i.e. the placed point was already "0.3 m off" -- and the anchor had said 0.04 m.
Reading the frame arithmetic: the trusted tools (05t, 05q, the cross-view scorer probe_cache_in_timeline)
take endzone clip frame = timeline frame + offset (offset = -15); the probes written tonight from
probe_orient_flip2 onward used timeline frame - offset, thirty frames (0.5 s at 59.94 fps) the other
way. A man moving 2 m/s is a metre from where those keypoints say, a standing lineman is not -- which
is exactly the "structure" I read as a defence-side depth bias. INVALID: the endzone columns of the
unwrap A/B and the blind-axis A/Bs (their sideline / jitter / hops / census columns and the render
strips stand), the pelvis-offset table (id 40 "70 px deep"), the bias-vs-scatter analysis, the hip
triangulation and all three tri_hips A/Bs (they compared a body with keypoints from another instant;
the "0.2-0.6 m" was the man's own motion), the lowest-ankle A/B's endzone column, the
id_timeline_reproj endzone column. VALID: the unwrap (sideline, jitter, strips, render), the orphan
rule, the fill bridge, the blind-axis rejection (steps and census), the geodesic and spine-20
decisions (the cross-view scorer indexes correctly), the seen-frame offset bins (sideline only).
Re-measuring with frame + offset: the pelvis offsets, the bias table, the triangulation. The lowest
ankle is re-measured too (its ruler was the broken one). tri_hips stays off and its docstring is
corrected once the numbers are real.

**Corrected numbers (20:50, frame + offset).** Endzone pelvis offset over 2062 live body-frames with
endzone hips: |d| p50 9.8 px, p90 19.0, 92 % within 20 px, 97 % within 40; two-view frames 9.5 / 18.2,
endzone-only frames 15.8 / 60.3 (n 133); no team or field-side structure (dx 0.4, dy 7.4 overall).
Hip triangulation on 1883 two-view frames (all pass the gates): rays meet within 0.06 m, hip height
0.84 m, reprojection 2.6 / 4.1 px, and the timeline's placement is 0.04 m (x) / 0.02 m (y) from it
at the median, 0.08 / 0.05 at p90, no id above 0.10. THE PAIRED PLACEMENT IS RIGHT. The endzone-depth
thread is closed; what remains of it is the endzone-only stretches (133 body-frames, p90 60 px, id
198's eight frames at 67 px), a population too small to chase before play 2. render/tri_hips is kept
as a ruler (the triangulation and its gates, docstring rewritten with these numbers); its placement,
keep-set and anchor code are deleted (commit below). The lowest-ankle A/B is re-run on the corrected
ruler for completeness.

Unwrap A/B re-measured on the corrected ruler (20:55): endzone lower joints p50/p90 12.4 / 21.4 (v49
timeline) -> 12.4 / 20.8 (unwrap), sideline limbs 7.2 / 16.2 -> 7.1 / 15.0, joint jitter p90 0.0354 ->
0.0327 (n 3169 body-frames). The unwrap is better on every valid column, the other camera included;
the earlier "endzone unchanged at 51.8 / 129" was the broken index (its level alone should have
said so: 129 px at p90 is a body length).

**07l v52 = v51 + the 10-frame fill bridge (21:00; the v48 render's timeline).** Live steps 5, hops
4 (37 @376, 166 @300, 74 @419, 17 @414), root jitter live p90 0.0119 -> 0.0115, census live 1.50 ->
1.32 (KC 10.78, BAL 11.34), whole clip 2.51 -> 2.42, hinges 0/0, joints p50/p90/p99 0.0106 / 0.0327 /
0.0689 unchanged, 13958 body-frames drawn (v51 14127). Render v48 = this timeline on the same cache,
strips auto-cut to scratch review_v48.

**Pops, classified (21:05, scratch probe_pop_classes).** The 32 pops on the live play are all GAP pops
within one id -- absent for 2 frames at the median, back 0.17 m away -- and not one is a fragment
handover (no id end is followed within 6 frames and 1 m by another id's start). They sit on ids in
the line and the pile (166: 5, 40: 4, 11 / 13 / 204: 3 each). Separately, 12 ids START mid-play with
no predecessor (37 @340, 38 @345, 74 @366, 40 @368, 80 @377, 39 @384, 167 @411, 197 @435, 84 @437,
198 @453 ...) and 7 END with no successor -- bodies appearing from nowhere and vanishing for good,
the rules admitting an id late (span, endzone-only, edge) or a fragment ending. A two-frame
absence with the body 0.17 m away is not a detection gap (the bridge is 10 frames): a timeline rule
removes the body on those frames -- the dedupe box when two bodies pass within DUPLICATE_M is the
suspect. Probing which rule and how close the nearest other body stood.

Lowest-ankle foot point on the corrected ruler (21:10): endzone pelvis |d| 9.8 / 19.0 (mean of both
ankles, shipped) -> 10.1 / 20.1 (lowest, endzone) / 10.1 / 19.8 (both cameras); census 1.32 -> 1.37,
root jitter p90 0.0116 -> 0.0122 / 0.0125; hops 4 -> 3, steps 5 -> 4 / 3. Worse on the ruler it was
built for and on census and jitter, better on two counts of small hops: DENIED for good. The mean of
both ankles stays.

Pop cause (21:15, scratch probe_pop_cause; the ground handed to build_timeline captured, then the
drawn states): two classes. (A) 18 pops where the id is NOT in the ground on the missing frames and
both neighbours are sideline-only (1, 11, 13, 19, 195, 82, 166 x5, 167, 197): a 1-5 frame hole in a
long-lived sideline id's detections; the bridge fills it, then dedupe_frames deletes the filled
frame because it is unanchored (no sideline detection of this id on that frame) and another body
stands inside DUPLICATE_M -- the rule built for fragment tails (a second id riding a detected man)
firing on a momentary dropout of the man himself. (B) 14 pops where the id IS in the ground and the
nearest other drawn body is 0.06-0.44 m away on the missing frames (204, 171 for 8 frames, 40, 17):
two ids on one man, the box killing this one on the frames it is filled and sparing it where it is
detected -- twin flicker. Remedy for (A): a filled frame whose id has a sideline detection within a
few frames on BOTH sides is a hole, not a tail, and stays anchored. Remedy for (B) is the twin
question (08o folds twins on the tracks; a stretch-consistent dedupe in the timeline would be the
render-side answer). Building (A) first; rulers = pops, census, hops, steps.

(A) built as `timeline._holes_by_frame` + `dedupe_frames(holes=)` (HOLE_REACH, commit c59c01b): an
interpolated state whose id the sideline detected within 6 frames on both sides is kept outright.
A/B on the live play (21:30, scratch probe_hole_ab): pops 32 -> 10, drawn body-frames 3561 -> 3601,
live hops 4 -> 4, steps 5 -> 5, census 1.32 -> 1.32 (KC 10.78 -> 10.96, BAL 11.34 -> 11.41), root
jitter p90 0.0116 -> 0.0129, p99 0.0606 -> 0.0692, duplicates dropped 1251 -> 1042; reach 12 takes
pops to 4 but the census to 1.45. ADOPTED at 6: twenty-two fewer vanish-and-reappear events per
live play for 1.3 mm/frame^2 of root jitter at p90. 07l v53 and render v49 (chained behind v48)
carry it. The 10 pops left are class (B), the twins.

**07l v53 = v52 + the hole rule (21:40; the v49 render's timeline).** Live steps 5, hops 4 (same
four), root jitter live p90 0.0115 -> 0.0125 (p99 0.0607 -> 0.0680), census live 1.32 (KC 10.96,
BAL 11.41), whole clip 2.42 -> 2.46, hinges 0/0, joints p50/p90/p99 0.0107 / 0.0329 / 0.0689,
14131 body-frames drawn (+173 over v52). The jitter cost is the readmitted filled frames; pops are
the number that moved (32 -> 10) and 07l does not print them yet -- probe_pops does.

**Twins on the live play (21:45, scratch probe_twins; drawn pairs within 0.45 m on >= 5 frames).**
Ten pairs. Two are twins beyond doubt: 166-204 (KC/KC, 23 frames 406-431 at 0.15 m, both sideline-
detected on 11 of them) and 1-40 (BAL/BAL, 19 frames 396-415 at 0.13 m, both detected on 9) -- two
sideline boxes 0.13-0.15 m apart on one man for twenty frames, each drawn as a body. The rest sit
at 0.27-0.43 m in the pile (19-166, 197-204, 19-37, 13-197, 82-166, 4-167, 17-204, 37-166), where two
real men can stand that close. 08o folds twins found by ankle rays at 0.03-0.11 m; these two are just
above it. Render-side rule to measure: a same-team pair within 0.2 m on >= 8 consecutive live frames
is one man, and the id with fewer drawn frames loses the stretch (the rider rule for long ids);
rulers = pops, census (a twin was a ghost in the count), hops, drawn body-frames.

A/B post-hoc on the v53 timeline (scratch probe_twins_ab; clock 20:25 -- the timestamps written into
this file since "18:05" ran ahead of the real clock by 60-80 minutes, I estimated instead of reading
`date`; the order of events is right): twin 0.2 m / 8 frames drops 111 body-frames on the clip (live:
40 at 399-413, 166 at 417-431 -- the two verified twins -- plus 21, 82 pre-snap and 60 / 75 / 197 /
205 after the whistle): census live 1.32 -> 1.20 (KC 10.96 -> 10.84, BAL 11.41 -> 11.32), live hops
4 -> 4, steps 5 -> 5, pops 10 -> 11 (the loser vanishes for its stretch and returns), drawn 3601 ->
3567. 0.25 m also takes 196 pre-snap (152 frames, census 1.19); 0.2 / 5 adds 168, 171 (139 frames).
ADOPTED at 0.2 / 8 (`timeline.TWIN_M`, `TWIN_MIN_RUN`, wired after the orphan rule): a body drawn twice
on one man for fifteen frames is worse than that copy vanishing. 07l v54 = v53 + twins; the v49 render
starts after v48 and picks this up, so v49 = v54's timeline.

**07l v54 = v53 + twins (clock 20:40).** Live steps 5, hops 4 (unchanged four), census live 1.32 ->
1.20 (KC 10.84, BAL 11.32), whole clip 2.46 -> 2.44, joints unchanged (p99 0.069); 111 twin body-frames
out (40 399-413, 166 417-431 on the live play).

**Ids starting mid-play, characterised (scratch probe_late_starts; team count over the five frames
before and after the start, nearest same-team body at the start).** Hole-fillers: 37 @340 (KC 9 -> 10)
and 38 @345 (KC 10 -> 11), both admitted by the endzone after the sideline lost them in the line --
right. Twins at birth: 171 @331 (0.13 m from a teammate), 84 @437 (0.12 m), 80 @377 (0.23 m), 197 @435
(0.30 m) -- they start on top of a drawn teammate and drift apart, too briefly for the 0.2 m / 8-frame
twin rule. Twelfth men: 40 @368 (BAL 11 -> 12, 1.7 m from anyone), 74 @366 (KC 11 -> 11.8), 39 @384
(25 frames), 167 @411 (10 frames, KC -> 13), 198 @453 (BAL 11 -> 12, endzone-only) -- real detected
players under a label the count already has, the run-surplus thread the census memory closed on
09-13 (no removal route that does not delete a real man). Recorded as the population; not pursued.

**v48 delivered (20:42): diag/play_001_v48_hifi_720.mp4 = v47 + the orphan rule + the 10-frame
bridge (07l v52's timeline).** Run sheet 344-454 (scratch review_v48/run_sheet.png): eleven white
and eleven red bodies through the run, the pile forming at 404-424 and breaking up by 444, nothing
floating, nothing lying down. v49 (+ hole rule + twins, 07l v54) started 20:44; its six strips and
a second review batch (ids 2, 7, 12, 3, 15, 30, 74, 37, 17, 40, 4, 11 and a 5-frame-stride sheet of
the whole live play) cut automatically when it lands.

**v49 delivered (21:33): diag/play_001_v49_hifi_720.mp4 = v48 + the hole rule + the twin stretch
rule (07l v54's timeline); the render log shows both firing (203's nine frames, 111 twin frames).**
Phase sheets (scratch review_v49/phases): pre-snap 100-300 -- the formation holds through the
cadence, a deep safety top-left, the Kansas City line in stances, the two split receivers, no white
body in the red line, the line filling in at 260-300 as the men set; post-whistle 460-640 -- the
crowd bunching bottom-right, lone bodies walking, a white body down at 640; nothing floating or
lying down on the play itself. Id 166 (a KC lineman) at 330-344: in a block, the line seen from the
side so the linemen overlap in the image (the camera's angle, not a defect). Whole-play sheet at
5-frame stride and twelve more per-id strips are cutting (review_v49/more).

v49 read (21:40, scratch review_v49/more): the 33-frame sheet of the live play at 5-frame stride --
the line holds 300-396, the pile forms 400-420, the play breaks to the near sideline 420-460, no
body floats, no team swaps, nothing lies down before the tackle. Id 2 (a Baltimore defender, 330-358):
standing then a natural arm raise, smooth. Id 74 (KC, 410-424): in a tackle scrum under two white
bodies, plausible. Id 40 (BAL, 380-394): running with bent knees and swinging arms -- and a second
white body drawn ON him at 380-386, the 1-40 twin before the stretch the 0.2 m rule takes (396-415):
the twins at 0.2-0.45 m still draw two bodies on one man for a few frames. Measuring TWIN_M 0.3 /
0.35 post-hoc (census decides: at 0.3 the pile pairs 19-166, 197-204, 19-37 come into reach, and
those may be two men).

Wider twin radii, post-hoc on the v54 timeline (21:50): 0.3 / 8 drops 51 more body-frames, 0.35 / 8
87, 0.3 / 12 43 -- ALL of them pre-snap (21 197-212, 196 132-209) or after the whistle (205, 76, 168,
198); on the live play nothing changes (drawn 3567, census 1.20, pops 11). The 1-40 double at 380-386
is not a within-0.3-m-for-8-frames case: they touch and part. TWIN_M stays 0.2 / 8; that double is
recorded as the residue.

Id 11 on the v49 strip (420-434) still vanishes on 422 and 428. Trace (scratch probe_id11_hole): the
sideline sees him 410-420 and again from 429 -- an eight-frame hole, one past HOLE_REACH 6, so the
bridge fills 421-428 but the dedupe deletes 422 and 427 as unanchored interpolated bodies near a
detected one, and 428 (endzone-only) falls to the one-view box (13 stands 1.49 m across). Measuring
reach 8 and 10 against 6 (reach 12 cost census 1.32 -> 1.45 earlier).

Reach A/B with the twin rule in (22:00): reach 6 pops 11 / census 1.20 / jitter p90 0.0122; reach 8
pops 7 / 1.19 / 0.0124; reach 10 pops 5 / 1.21 / 0.0126; hops 4 and steps 5 throughout. Eight wins
on pops and census: HOLE_REACH 6 -> 8. 07l v55 and render v50 (with a strip of id 11 at 420) follow.

**07l v55 = v54 + reach 8 (22:10; the v50 render's timeline).** Live steps 5, hops 4, root jitter live
p90 0.0120 (p99 0.069), census live 1.19 (KC 10.86, BAL 11.32), whole clip 2.47, joints p99 0.069.

**Facing vs motion, a new ruler (22:15, scratch probe_facing):** on the 260 live body-frames with
root speed > 0.1 m/frame, the angle between the body's forward and its velocity is 43 deg at the
median, past 90 on 17.7 %, past 135 on 3.5 %. Who runs "backwards": 6 (a Baltimore back, 12 frames
at 135 -- a backpedal), 166 / 19 / 5 (Kansas City linemen at 93-116 -- pass sets, retreating
squared up), 13 at 72; the runner 9 faces his motion at 13 deg over 132 frames, 3 / 11 / 204 / 74 /
40 under 60. Every past-90 case is a man who really moves backwards or sideways. No defect; the
ruler stays as a check for other plays.

**THE FEET SKATE (22:25, scratch probe_skating -- a new ruler).** On the 244 live body-frames where a
body moves faster than 0.1 m/frame, the slower ankle's WORLD speed is 0.94 of the pelvis speed at the
median and a foot is planted (under 0.3 of it) on 1 % of frames. A real runner plants one foot about
half the time: the foot stands still on the turf while the body passes over it. Every moving body on
the play glides with its legs cycling too little -- the runner 9 (132 frames, ratio 0.93, planted
2 %), 40 (1.00), 19 / 166 / 74 (1.00), 3 (0.86), 11 (0.56 -- the least bad). This is the "moonwalk"
of fitted motion and a viewer sees it without a ruler. Where is the plant lost? Either the sideline
keypoints themselves never plant (the detector on a 100-px blurred sprinter) or they do and the
pipeline damps the legs (the fit's temporal term at 3.0, the stride-2 keyframes with SLERP between,
the sigma-2 Gaussian). Measuring on the runner: each ankle keypoint's image speed against the hips'
image speed, beside the same for the drawn body (scratch probe_plant).

Result on the runner (322-372, 49 frames): the ankle KEYPOINTS' slower foot moves at 0.91 of the hips'
image speed (planted under 0.3 on 8 % of frames); the drawn body's at 0.77 (4 %). The keypoint ankles
jump 0.2 -> 15 px/frame from one frame to the next on a body whose hips move 3-6 px/frame: the
detector's ankles on a blurred 100-px sprinter carry no stance phase, so the fit cannot have one,
and the smoothers do not add or remove one. The skating is a DETECTOR LIMIT of the 2D keypoints;
the remedy is on the gait side: detect stance from the fitted leg cycle (the drawn ankle's world
speed dips to 0.3-0.4 of the pelvis speed every 8-15 frames on the runner) and lock the planted
foot to the turf with a two-bone leg IK while the pelvis keeps its smoothed path -- the leg then
extends behind the body as it passes over the foot, which is what running looks like. Cost to
measure: the sideline lower joints (the locked foot leaves its noisy keypoint by the skate distance,
0.1-0.3 m = 8-24 px), joint jitter; gain: the planted share (1 % -> ~40 %) and the render strip of
the runner. Building it opt-in.

Built (`render/foot_lock.py`, commit fba3c97: stance = the drawn ankle under half the pelvis speed for
2-8 frames, the foot pinned to the segment's median xy, hip and knee re-solved by a numpy-FK
least-squares; tests pass) and measured post-hoc on the live play (scratch probe_footlock_ab): 22
stance segments over all ids, 90 frames, the pinned feet moved 1-6 cm at the median (10 cm at most),
planted share 1 % -> 2 %, ratio p50 0.94 -> 0.94, sideline lower joints 9.9/16.2 -> 9.9/16.2, joint
jitter p90 0.062 -> 0.068. NO EFFECT: the criterion reads stances off dips the drawn legs do not
have -- they hardly cycle -- so it finds a few shallow ones and moves a foot a few centimetres. The
lock is left opt-in as a tool (it does what it says when a stance exists); the defect needs a stance
the data does not contain. That is a gait model: cadence from the body's speed (about 2 Hz at 8 m/s),
alternating stance phases, and leg motion synthesised to plant each foot -- motion synthesis over the
fit, which is a product decision (fitted legs that skate vs animated legs that plant) and a day of
work. Recorded as the next hypothesis class; the skating share goes into 07l as a standing ruler.

07l v56 (same timeline as v55, the skating ruler added to the report, commit 16261e5): skating p50 0.94,
planted 1 % of 244 moving frames; everything else as v55.

**Two decisions for the user (both product, both recorded, neither taken):**
1. Post-whistle: the render runs to frame 647; after ~500 the bodies are the crowd and the pile
   with no mechanism left to fix (the steps over 0.4 m are all after 500). Ending the clip at the
   tackle (05k `--limit` or a frame cap) is one flag.
2. Skating: fitted legs glide (planted 1 % vs ~50 % real). The only fix is a gait model over the
   fit -- cadence from speed, alternating stances, legs synthesised to plant, blended below the
   hips -- i.e. animated legs on fitted bodies. A day's work, and a different kind of footage
   (the legs would no longer follow the detector). Not started without a yes.

**v50 delivered (22:35): diag/play_001_v50_hifi_720.mp4 = v49 + HOLE_REACH 8 (07l v55/v56's
timeline).** Id 11 at 418-432 on the render strip: present on every frame, moving through the scrum
-- the two vanishes of v49 (422, 428) are gone on the footage. AFFIRMED. Delivered clips tonight:
v45 (82 re-teamed, rider rule), v46 (the v49 recipe, the before-footage), v47 (+ the half-turn
unwrap), v48 (+ orphan rule, 10-frame bridge), v49 (+ hole rule, twins), v50 (+ reach 8).

Twins at birth, measured post-hoc (22:45, scratch probe_birth_twins; an id whose first frame stands
within 0.3 m of a same-team body loses its frames while it stays on him): 6 body-frames (80 one
frame, 84 five), census 1.19 -> 1.15, pops 7, hops 4; at 0.4 m 16 frames (197 too), census 1.17.
Measured and NOT adopted: six frames is below what a viewer can see and not worth a rule; revisit
if play 2 shows the class at scale.

**Where the loop stands at 22:45.** Every mechanism the footage named tonight that had a
measurable fix is shipped and affirmed on the render (v47-v50); the rulers that remain on the live
play -- steps 5, hops 4 (all 0.18-0.27 m at source transitions), pops 7, census 1.19, joints p99
0.069, flails 1.8 % (real hands) -- are at the floor of what the rules can reach. What is left is
the two product decisions above (clip end, gait model), the runner's handoff 15 px (ambiguous
between two fits), and the twelfth-man thread (closed 09-13). Next play is the next signal.

**RESUME PLAN (machine off 2026-09-16 ~10:10; nothing running that matters).** Shipped state on disk:
poses_refit.json = the two-view hard-hinge cache (v47; .pre_tw3 is its copy), timeline with Gaussian
sigma 2 on body_pose and sigma 4 on orientation (report v48), renders v39..v44 in diag. Scratch caches
for the eight worst ids in `scratch/`: refit_tw3.0.json (temporal 3.0), refit_unseen5.json (unseen-limb
hold 5), refit_tw3u5.json (both).
  1. DONE before the shutdown (10:12): the combination scorer landed. At the timeline's placement,
     eight worst ids: C 0.144 / endzone 19.8-28.0 / sideline 9.8-22.2; temporal 3.0 0.079, off-axis
     0.5 % / 19.9-29.7 / 10.1-24.0; unseen 5 0.114 / 19.8-27.9 / 9.7-21.6; BOTH 0.079 / 19.9-29.7 /
     10.1-24.0 -- identical to temporal 3.0 alone: at that weight the temporal pull already holds an
     unseen limb, and the extra multiplier adds nothing. Winner: `--temporal-weight 3.0` alone.
  2. Whole-play refit with the winner: `PYS scripts/05p_refit_mono.py --play-dir P --two-view
     --endzone-weight 0.3 --temporal-weight 3.0 --workers 6` (~100 min; back
     up poses_refit.json first). Then `07l --tag v49 --joints` and the cross-view scorer with the new
     poses_refit.json as the extra argument; ship if joints improve with steps / root / census / endzone
     not worse than v48 (live 11, root live p90 0.0128, 1.65, endzone lower 19.8 / 28.0 on the worst ids).
  3. If shipped: set Mono2DConfig.temporal_weight default to 3.0 (unseen_temporal_mult stays 1.0),
     commit with the table, render v45 (`scratch/launch_v44.sh` with v45 substituted).
  4. Open product decision for the user: end the render at the tackle (~frame 500); post-whistle steps
     are the crowd and have no mechanism to fix.

**Local repair step 1, hold-through, measured and REJECTED as a default (2026-09-15).** Stretches
where the raw fit's max joint speed exceeds 0.25 m/frame (merged within 3 frames, padded 2): 85 on
the live play. Replacing a stretch's body_pose by the SLERP between its clean boundary poses, then the
shipped clamp + Gaussian (jitter p50/p90/p99; reprojection sideline p50/p90, endzone p50/p90):

    shipped              0.019/0.144/1.11   9.2/21.0   6.5/15.9   0 frames
    hold <= 12 frames    0.018/0.132/1.11   9.3/21.1   6.5/16.8   418 frames, 31 stretches too long
    hold <= 24           0.018/0.120/1.13   9.4/22.0   6.5/17.3   626, 18 skipped
    hold <= 48           0.018/0.118/1.08   9.4/22.3   6.5/17.5   664, 17 skipped

The same trade curve as the sigma sweep (0.01 of jitter for ~0.5 px), and the p99 does not move: the
garbage is in the LONG stretches (17 of 85 run past 48 frames), where holding is a mannequin. A
post-hoc repair cannot make a good pose from a bad one; the fix belongs in the fit -- a temporal term
across neighbouring keyframes and joint limits inside the optimiser, so the pose that matches the
keypoints is also one a body can hold. That is the next build, not another smoother.

**A false bug report avoided, worth recording.** I suspected this 2024 play had been resolved against the
2025 roster, because `roster.py` reads names only from a `player_name` column (the 2025 schema) while the
2024 file stores them under `full_name`, and yet identities carry names. The check disproved it: "Swayze
Bozeman" is in the 2024 table (KC 55, LB, DEV). My query had printed `player_name` for both files, so the
2024 names were never displayed at all. The schema difference is real; the conclusion drawn from it was not.

Worth recording separately: the probe's baseline census reproduces the ad-hoc figures recorded for v38
(snap 2.82 vs 2.85, play 1.98 vs 1.98, run 1.79 vs 1.79, aftermath 3.44 vs 3.46), so the two instruments
agree and the census numbers in this document are reproducible rather than one-off.

**The sideline DETECTOR's share of the deficit is closed**: those men cannot be separated in that view.
What is not closed -- and what an earlier version of this section asserted too broadly -- is that the last
geometric route is gone. A render can also delete a body it already holds, which is a different mechanism
from a camera that never resolved one, and `dedupe_frames` above is exactly that and still unmeasured.
Play 1's honest body-count error is 1.98 a frame over the play.

**And the "worst body-frame" is not a fit or pairing fault at all.** id 17 has records on only frames 458
and 460 in that entire stretch: an isolated two-frame island where the sideline ground point (-26.3,
+3.9) sits **4 m** from the endzone's (-22.9, -1.2), with the cached translation matching the sideline
exactly. The sideline track produced a ground point on those two frames only, and produced a wrong one.
A tracking artifact, already excluded from the ruler by the 0.6 m ray-miss gate -- not something the
fits or the joins can repair.

Three times now the geometry has proposed something the footage refused -- a body supposedly drawn on
the camera rig (it was a drone in the plate), players supposedly cut off by the sideline crop (nothing
was clipped), and now the endzone supposedly over-segmenting the line. **Look at the frame before
believing a count.**

Baltimore's 15-18 is the other half, and two new faults sit in it. **Officials are voted onto a team**: id 85
stands motionless at (-10.1, +5.2) through frames 300-400, 8.6 m from any sideline body, and id 87 at
(-51.7, -0.4) is 17.7 m behind everyone -- officials wear white, so the saturation vote reads them as
Baltimore. And **one body can carry different teams in the two cameras**: endzone id 86 (KC) is sideline id
82 (BAL) at 0.55 m, endzone 99 (KC) is sideline 6 (BAL) at 0.49 m, endzone 91 (BAL) is sideline 3 (KC) at
0.68 m. The kit vote is per camera and per id, so nothing forces the two views of one man to agree; the vote
belongs on the merged id, after pairing, not before it.

**The cause is the two-view gate.** Simulating `two_view_pass`'s filters per player showed the
cross-view error is monotone in the fraction of a player's frames that got the triangulated ankle
anchor:

    id 12  187 frames reach the two-view fit, 180 anchored (96 %)  ->  endzone   9.3 px
    id  5  118                              , 116         (98 %)  ->  endzone  16.2
    id 17   62                              ,  31         (50 %)  ->  endzone  20.1
    id 25   34                              ,  12         (35 %)  ->  endzone  14.5
    id 19   93                              ,   4         ( 4 %)  ->  endzone 142.2

But id 19 *did* reach the two-view fit on 93 frames, and the fit is innocent: on these five players
its own medians were **sideline 14.6 px and endzone 4.9 px** -- better in the second view than the
first. `_two_view_job` gated validity on `first_rms`, the fitted camera alone, at
`--reproj-px-max 20`. 126 of 328 frames failed that sideline gate with excellent endzone error, and
**not one of id 19's 93 frames survived it**, so all 170 of his frames fell through to the one-view
pass, which reported `id 19: 170/170 frames, reproj 13.1 -> 4.0 px`. That four-fold "improvement" IS
the defect: `merge_into_refit` skips a frame whose `valid` is False, so it is absent from the cache,
counts as uncovered, and the one-view pass refits it to the sideline alone -- where 4 px is trivially
reachable by sliding the body along the ray that camera cannot see. The gate's own comment records
the original reasoning ("a weak second view's residuals are large by design, endzone 17 px at weight
0.3"), which inverted when the endzone became an equal partner at weight 1.0. Nobody re-measured the
rule when its premise changed.

`--two-view-gate first|worst|mean` now exposes it, default unchanged. Note that **`worst` is
stricter and would reject id 19 too**; `mean` is the setting that keeps a frame whose second view is
good. And pixels are not comparable between these cameras -- the endzone's px per metre differs from
the sideline's, so 4.9 px there may be metrically better than 14.6 px here -- which means the
principled form of this gate is metric, not pixel. The deeper fix may be simpler still: a frame the
two-view pass attempted should never be handed to the one-view pass, because a one-view fit is
strictly worse in 3-D whatever its pixels say.

**Measured, on five players, with the harness validated first.** The baseline arm (`--refit
<nonexistent>`, otherwise v34's invocation) reproduces the v34 cache under 05t to the last digit, so
a difference in any arm belongs to the flag. Then:

    five players, 05t                     baseline   --anchor-max-miss 0.6   --two-view-px-max 40
    two-view frames kept                   202/328          201/328               327/328
    frames anchored on the ankles          179/328          290/328               179/328
    endzone p50 / p90 / p99 (px)      17.4/150.8/382.6   17.4/150.9/382.7      16.2/ 24.9/153.7
    endzone frames over 100 px          118 (34.8 %)     118 (34.9 %)            15 ( 4.3 %)
    id 19 endzone p50 (px)                 142.2            142.2                  16.5
    sideline p50 / p90 (px)              3.8 / 13.8       3.8 / 13.8            5.6 / 30.5

`--anchor-max-miss 0.6` is a **null result** and an instructive one: it anchored 111 more frames on
the point both cameras agree about and moved nothing that survives -- the anchor decides how often a
player is exposed to the gate, not what the gate does. `--two-view-px-max 40` collapses the
catastrophic tail, and the sideline pixels rise, which is the correct trade rather than a cost: that
number was never evidence, it was the symptom of a body placed to satisfy one camera.

**The second ruler, in metres, is unambiguous** -- gap from the point both cameras put the feet:

    id (ankle miss)   along-ray p50 base -> px40     across-ray p50
    id 19 (0.36 m)        1.04 m  ->  0.05 m          0.02 -> 0.01
    id  5 (0.19 m)        0.02    ->  0.02            0.00 -> 0.00
    id 17 (0.16 m)        0.25    ->  0.25            0.13 -> 0.13
    id 25 (0.34 m)        0.06    ->  0.06            0.03 -> 0.03
    id 37 (0.19 m)        0.02    ->  0.02            0.02 -> 0.01
    pooled                p50 0.07 -> 0.04, p90 1.12 -> 0.23

The broken player improves twentyfold and **not one other player moves a centimetre**, which is the
adoption test passed: beaten on a ruler that is not the one being optimised, with no regression
elsewhere. v35 applies it to the whole play and re-scores with 05t, the census and both overlays.

**v35, the whole play, and it holds.** The two-view pass now keeps **1787 of 1793 frames across 24
players** (its own medians: sideline 6.7 px, endzone 4.3) instead of throwing the informative ones to a
one-view refit:

    whole play, ~1820 body-frames both cameras see     v34        v35
    endzone p50 / p90 / p99                        8.7/25.3/341.3   8.8/16.7/140.3
    endzone frames over  20 px                       197 (10.9 %)     92 ( 5.0 %)
    endzone frames over  30 px                       178 ( 9.8 %)     55 ( 3.0 %)
    endzone frames over 100 px                       129 ( 7.1 %)     24 ( 1.3 %)
    sideline p50 / p90                                3.4 / 11.0      3.4 / 12.7
    sideline frames over 20 px                         6 ( 0.3 %)    147 ( 8.1 %)

and in metres, over 1680 paired body-frames on 23 players: |along-ray| **p90 0.21 -> 0.05 m, p99
2.51 -> 0.34 m**, id 19 1.04 -> 0.05, and **0 of 23 players worse by more than 0.10 m** -- every other
player identical to the centimetre. The catastrophic tail is cut fivefold, nothing regresses in metres,
and the sideline's own pixels get worse, which is the trade rather than the cost. `TWO_VIEW_PX_MAX = 40`
is now the default in 05p and carries these numbers in its comment.

(08f declined to re-split the teams on the new cache -- "no clear two-way split: largest gap 9 at 189,
123 below, 2 above; refusing" -- so the labels are v34's. Refusing beats guessing, and it is the same
one-sided saturation histogram the kit work has hit before.)

**The census did not move, and that is the useful part.** v35 scores `|KC-11| + |BAL-11| = 3.02` bodies
a frame from the snap against v34's 2.95, with KC mean 11.7 and BAL 12.0 and only 4 frames of 361
exactly eleven and eleven. Placement improved fivefold on its own ruler and the body count got very
slightly worse, which settles that these are independent axes: the two-view bound fixes where a body
stands, not whether the right bodies exist. **The census is now the binding problem.** What the render
threw away on the way to that number:

    frames beyond an id's sideline span left out   2110
    endzone-only ids left out                        34
    edge-clipped one-view ids left out               25
    sideline dwellers left out                       20
    bodies behind the offence (officials)            ids 85, 88, 106, 107

which corrects something written above: the officials that the kit vote labels Baltimore are **already
excluded from the drawn bodies** -- id 85, the motionless referee at (-10.1, +5.2), is in that list. The
vote is still wrong about them (white stripes read as Baltimore) and that wrong label still reaches
anything reading `identity_resolved.pkl` directly, but they were never inflating the census. The surplus
of 15-18 "distinct places" for Baltimore is a count of RAW ids, before these rules.

Note also `endzone-only ids left out: 34` sitting beside the finding above that 2-4 endzone-only ids a
frame are men no sideline body is within 1.5 m of -- the occluded linemen. The rule that drops them is
the same one protecting the count from double-drawing unpaired duplicates, which is exactly why the fix
has to be pairing (join the ids by ankle-ray agreement) rather than relaxing the rule.

**The depth snap is exonerated and re-measured as a win.** It is not the source of the along-ray
error: it never moves id 19 (1.09 m before, 1.09 m after; worse on 3 frames of 107, better on 1).
Where it fires it is strongly right -- id 5 0.40 m -> 0.05 m of along-ray error (better on 115 of
117 frames), id 25 0.60 m -> 0.11 m (better on 19 of 33).

**Two hypotheses killed, and a trap.** `two_view_pass` takes its endzone offset from
`poses_tri.json` rather than `clip_offset`; a disagreement would silently mis-pair every player, and
they agree (both -15). The edge cross-fade of `global_orient` did not fire on the records in question
(the anchor it would blend toward sits at 71 deg while the cache holds -172). And 05p's resume path
reads: if `poses_refit.json` holds "mono", start again from `poses_refit_fused.json` -- a backup only
written when `poses_refit.json` existed at the start, which the pipeline deletes first, so the file
on disk is from 09-09 and a plain re-run would build on a two-day-old two-view cache computed with
the old endzone camera. **Check that backup's mtime before trusting it**; pass `--refit
<nonexistent>` to reproduce the pipeline's empty-blob state.

**Pose, place or facing: how to tell.** `05s` holds a record's `body_pose` and re-solves only what
you allow to move. On id 9's frames 258-270 (rms px): cache 18.7/25.1/29.5/22.9, translation-only
12.2/19.6/26.2/19.1, `global_orient`+translation **3.2/3.2/3.3/3.6**, and a 180 deg turn 28-39. The
limbs were never wrong -- the heading was, drifting +80 deg across twelve frames while the man ran
straight. This is local: only 16 of 5099 frames exceed 20 px in the sideline.

**Measured and rejected before building it:** a velocity-heading prior. 51 % of 1778 body-frames over
1.5 m/s face more than 90 deg from their own travel direction, which looks like a catastrophe and is
mostly correct football -- turning those bodies 180 deg beats the cache on 0-1 % of them. id 12 holds
163 deg while sliding at 85 deg (7.6 px as cached, 49.5 px flipped) because he is a lineman blocking,
and defensive backs backpedal the same way. Such a prior would wreck exactly the players it appears
to save.

### v34: the gap fill lands, and the count turns out to be fragmentation (2026-09-11)

08q recovered 607 sideline boxes inside gaps in existing tracks, from detections
the 0.35 threshold had rejected, and the keypoints were re-read for them. At the
snap Kansas City goes from 8 drawn bodies to 9 of 11, and 27 players triangulate
from both cameras instead of 25. The overall count error did not move: 2.93 ->
2.95 bodies a frame from the snap.

What that says is where the error really lives. Play 1's eleven Kansas City
players are carried by 32 separate sideline track ids, and Baltimore's by 51. A
player is tracked, lost, and picked up again as a NEW id -- which produces both
halves of the error at once: a hole while nobody holds him, and a surplus where
the old id and the new one overlap. Filling the small holes inside a track cannot
reach either.

By phase of play (bodies drawn, truth 11 and 11):

    at the snap (280-340)   KC 8.8   BAL 11.7
    the run     (340-460)   KC 12.2  BAL 11.8

The next piece is joining fragments into one identity per player. Position-only
stitching is a measured dead end (2026-09-07: it welded a wrong pair for every
right one). What is new since: the kit read from the pose torso, the role and
build per player, depth-corrected positions, and the one-place-at-a-time veto in
the pairing. A stitch gated on those is worth measuring against the census.

### Counting bodies per team: the ruler, and what it says (2026-09-11)

Eleven a side is the one number in this problem that is known without labelling
anything, and nothing was checking it. Per frame, from the snap onward, play 1
draws |KC - 11| + |BAL - 11| = 2.9 bodies wrong on average, and exactly eleven a
side on 2 frames of 361. The median hides two different faults pulling opposite
ways:

- AT THE SNAP Kansas City is three short. The sideline detector has 8 boxes for
  their 11 players; the endzone camera has 11. The men it loses are the linemen
  in the pile at the line of scrimmage. The render can only draw what the
  sideline tracks, so those three are simply absent.
- LATER IN THE PLAY both teams run one or two OVER, as the tracker re-acquires a
  player under a new id and both are drawn.

Tried and rejected (2026-09-11): keeping an id beyond its sideline span wherever
the sideline has no body within 1.2 m of it. It is the right idea for the first
fault -- four Kansas City players are seen by the endzone through a sideline gap
-- but it admits a ghost for every player it recovers: 2.93 -> 3.01. The
capability stays in endzone_only_rule (side_ground=), unused.

The honest read: the count is limited by the sideline DETECTOR's recall in the
trenches, which is upstream of everything in the render. That is the next place
to look, not the render's rules.

Measured there (2026-09-11, yolov8x at imgsz 1920 on play 1's sideline, twelve
frames): person boxes at the detector's 0.35 threshold against lower ones --

    frame   0.35  0.20  0.15  0.10
    160-400  19-23  19-26  19-26  19-27   (the play: 22 players + 2-4 officials)
    440-600  26-42  28-54  31-63  34-76   (the camera pans, the crowd enters)

At frame 300, the snap, the two boxes that appear only below 0.35 are a Kansas
City lineman half hidden behind another, at confidence 0.17 and 0.13 -- exactly
the men the census says are missing. But a blanket threshold drop is not the
answer: late in the play it triples the detections as the sideline crowd comes
into frame. The shape of the fix is to fill the GAPS OF EXISTING TRACKS: where a
track has no box on a frame inside its own life, look for a low-confidence
detection near where the track should be, and take it only there. That keeps the
threshold high everywhere else and creates no new identities.

### The endzone overlay, and what it showed (2026-09-11)

The user asked which camera the overlay uses. It used the SIDELINE alone, and
that is a blind ruler for the one axis that matters most: the sideline looks
down the line of scrimmage, so a body a metre too near or too far lines up in
its image exactly as a right one does. 05q now draws into either camera (the
endzone clip runs at f + clip_offset, so the label says which film frame it is)
and `--cam both` puts them side by side. Colours: GREEN the detector's 2-D
keypoints in that camera's film, RED a fitted Kansas City body, WHITE a fitted
Baltimore body, cyan the track id.

The first endzone frame showed three things the sideline overlay had hidden:

1. Two officials and a sideline marker are drawn as Baltimore players. All the
   drawn endzone-only ids are non-players: 85 and 89 and 108 are officials
   (85 wears 83 on his back, so the jersey reader gave him Baltimore's Qadir
   Ismail) and 109 is a marker. The frustum test in endzone_only_rule keeps
   endzone-only ids the sideline could not have seen, which is right for a wide
   receiver and wrong for the referee standing 14 m behind the ball.
2. The counts are wrong: the render draws a median of 9 Kansas City bodies and
   13 Baltimore ones, where the truth is 11 and 11. Bodies per team per frame is
   the sharpest ruler this project has had; it was never being looked at.
3. Several one-view bodies sit up to a metre off in depth, visible only here.

### Where play 1 stands at v31 (2026-09-10, late)

Sent: v30 (role builds, the pairing veto) and v31 (twins folded). Fits on the
v31 ids: two-view 1577 frames on 22 players at 6.1 px sideline / 4.1 px endzone,
one-view 3487 frames at 2.6 px, population median against a body's own keypoints
3.1 px (v28b 3.8, v30 3.2). The render draws 67 players over 647 frames.

The bodies that still sit worst against their own keypoints, all one-view:
id 13 at 12.9 px over 193 frames -- a defensive lineman in a three-point stance
whose LEFT arm reads 33-49 px off with keypoint confidence 0.26-0.46 while his
right arm sits at 10-15 px (the down hand is under his own body and the detector
barely sees it); then 76 (14.5 px, 34 frames), 40 (12.4, 20), 55 (9.7, 11).
Nothing there looks like an identity fault any more -- the fits are on the right
men, in the right poses, with an arm that cannot be seen.

Everything tried against the remaining arm jitter has now been measured and
rejected: the left/right flip filter, holding an unseen limb at its previous
pose, per-joint outlier rejection, the restart, anatomical bounds, and the
side-agnostic residual. The arms move when the detector loses them, and no
prior built out of the previous frame fixes that.

### Two ids on one body: the feet tell, the boxes do not (2026-09-10)

The render stood two avatars on one man in play 1's trenches. The sideline
tracker held the left tackle as ids 18 and 19 for 139 frames and the quarterback
as 16 and 22 for 58, which shows structurally too: six interior linemen on a
five-man line, two quarterbacks under centre.

Box geometry cannot find them. From the sideline the line of scrimmage is seen
end-on, so two men standing a metre apart in depth overlap in the image exactly
as a duplicate does: ids 14 and 27 (two Baltimore players, one behind the other
in the footage) sit 0.19 box heights apart with IoU 0.66 and any box rule flags
them. The ankle keypoints' rays taken to the turf resolve depth and separate the
two cases cleanly -- over play 1's same-team sideline pairs the duplicates sit
0.03 and 0.11 m apart and the next pair 0.80 m. Each id's ankle track is filled
across gaps of up to 4 frames first: the pose detector suppresses one of two
overlapping boxes, so a twin pair carries ankles on the same frame only 3 times
in 58, and the raw test (both ankles, same frame) read 18 & 19 as 1.30 m apart
off those three frames alone.

`tracking.twins` + `scripts/08o_merge_twins.py`: same team, no disagreeing
jersey numbers, feet within 0.20 m, boxes overlapping 0.3 IoU, 20+ frames
judged. 0.35 m was the first threshold and it folded the endzone's quarterback
into the lineman beside him (0.29 m at the feet, two men in the footage) --
that is the calibration of this rule. The merge is per BODY, not per camera:
ids are global, so a merge found in one camera moves the other camera's rows too
(the first cut folded endzone 113 into 22 while the sideline folded 22 into 16,
splitting one person's two views). Where the merge leaves two boxes of one
camera on a frame, the less confident is dropped, so the merged id holds one box
per frame. The stage runs between the repair and the roles, under numpy 1, and
08c --from-cache follows it.

### The cameras' ankle RAYS, not their ground points, say whether a pair is right (2026-09-10)

Chasing the motion man's arms at 262-270 produced a false lead worth keeping.
The two cameras' ankle ground points (each camera's ankle ray taken to z = 0.08)
disagree by 1.11 m in x for him where every other player's agree to 0.20 m, which
looks like a mis-pair. They are not: the two rays miss each other by 0.15 m. The
sideline camera is nearly horizontal, so a ray that passes within a hand's width
of the truth lands a metre away on the turf, and the ground-point difference is
mostly the z assumption. The ray miss is the honest test: 3189 of play 1's frames
have confident ankles in both cameras, they miss by 0.15 m at the median, and 1 %
by more than 0.6 m. `05p --two-view-max-miss` (default 0.6 m) leaves those frames
to the one-view pass.

What the arms actually are: at 262-272 the temporal keypoint filter throws out
his left elbow and wrist (the crowd), so no camera constrains that limb and only
the L2 pose prior holds it. `Mono2DConfig.unseen_temporal_mult` holds an unseen
limb at the pose it had instead. MEASURED at 6 (ids 9 and 12): the runner's arm
speed p90 in his own frame went 10.8 -> 11.7 m/s and the footage strip is
unchanged. It does not help: the limb is unseen for two or three frames at a
time, so holding it only delays the snap back, and the pose it is held at came
from the frame the detector had already lost. OFF by default. If this is picked
up again, the thing to hold an occluded limb with is a real pose prior (a
learned one, or the player's own pose distribution over the play), not the
previous frame.

### Builds by role, and one id that held two men (2026-09-10)

The user asked two things of v28: fix "joint overlap / player confusion" by giving
each player a fixed set of joints traced frame to frame (occluded ones inferred from
SMPL-X's priors), and explain why a defensive lineman and a cornerback come out the
same body. Both were measured before anything was built.

**Builds.** The roster build (height -> beta 0, weight -> beta 1 via mesh volume)
reached only the ids the jersey OCR named; unnamed "P<id>" bodies were 1.85 m with
no weight. `identity/roles.py` + `scripts/08n_role_builds.py` read the role off the
pre-snap formation: the snap is the first frame after which fewer than 40 % of the
bodies stand still for 12 frames (play 1: 300); the formation is set where 70 % stand
still (62); the line of scrimmage is the midpoint of the two teams' crouched lines
(x = -24.0 m; the play's BAL 24 is -23.8). Offence on the line: the five distinct
bodies nearest the formation's centre (two ids within 0.7 m are one body) are the OL,
a stance beyond them a TE, else WR; an upright body on or just behind the line within
2.5 m of centre is the passer; the backfield within 7 m is RB (QB if on the centre's
line and standing). Defence: a stance on the line = DL, standing on the line within
9 m = LB (edge / walked-up), second level within 4 m of centre = LB, else DB; a
defender more than 0.8 m past the ball is mis-teamed (no role). Named ids take their
roster position; a track that starts within 90 frames of a same-team roled track's
end, within 2.5 m + 0.06 m/frame of where it ended, inherits its role if it beats the
runner-up by 1 m. Builds are the KC/BAL 2024 medians (OL 1.96/143 kg, DL 1.91/134,
LB 1.88/107, DB 1.83/91, WR 1.83/89, TE 1.96/111, RB 1.78/98, QB 1.87/96), written
into identity_resolved.pkl (frozen dataclass: `dataclasses.replace`), which both 05p
and the timeline read. Play 1 v30: 26 roles from the formation, 28 from the roster,
9 inherited, 31 unnamed ids built. Known limits: with a lineman hidden behind the
centre the tight end is the fifth body (OL build, 32 kg over); adjacent linemen are
0.3 m apart laterally in the sideline ground points, so duplicate tracks on one
lineman look like two bodies.

**Confusion, measured.** A ruler (`scratchpad/kp_conflict.py`): project every
fitted body's joints into the camera and count detected keypoints that sit within
12 px of ANOTHER body's same or mirrored joint while more than 8 px from their own.
Sideline: 0.1 % of 57k keypoints (0.5 % near another body at all); endzone 0.5 %.
The runner's occluded 262-268 has no stolen keypoints: it is occlusion. The user's
per-player joint set is what the per-box detector + temporal term already give; the
prediction-gated assignment was NOT built because there is nothing for it to fix at
the keypoint level. One outlier: id 17 had 28 % (sideline) / 34 % (endzone) of its
keypoints more than 40 px from its own fit.

**The outlier was an identity merge.** The v28 strip for 17 at 190-211 shows the drawn
body on the quarterback's keypoints at the centre's place: sideline id 17 held the
centre AND the quarterback standing 0.7 m behind him for 165 frames (old sideline 16
merged into 17 by the lag -15 re-pairing), plus two short fragments; sideline 21 and
15 and endzone 3, 17, 42 likewise (1883 duplicate rows, keypoint sets of 22-25 rows
that 05p skips). The greedy loop's continuation waiver ("the same person within 1 m")
let both sideline tracks hold the endzone centre track and the union-find merged
them; the fit then sat 46-59 px between the two bodies through 142-374, and the edge
cross-fade dragged even the one-view frames onto the centre's placement.
`global_ids_checked` (tracking/pair_by_appearance.py) refuses a union that would put
two same-camera tracks overlapping in time beyond a short continuation (45 frames,
half the shorter span) into one id; 08i prints the drops (play 1: 4). Re-running the
repair needs the `*_oldids` caches (they carry tracks_oldpair.parquet's ids; the live
caches' merged ids cannot map back by boxes): `08i --keypoints keypoints_2d_oldids.parquet
--keypoints-ids tracks_oldpair.parquet`, copy the oldids caches into place, `08m --old
tracks_oldpair.parquet` (scratchpad `p1_v30_repair.sh`). After it: 0 oversized keypoint
sets, 50 duplicate rows (continuations), the centre a crouched body at the line, the
quarterback his own chain (16 -> 22 -> 33 -> 49). The veto also keeps genuine duplicate
tracks (two tracker ids on one lineman) apart; that is the linker's problem, to be
fixed by box overlap, not ground distance, and measured by bodies drawn vs boxes.

**Versions.** v29 = v28b + role builds (poses only; runner strip unchanged). v30 =
v29 + the vetoed re-pairing (fits running at the time of writing).

### The endzone camera was never on its paint; the clips are 15 frames apart (2026-09-09, evening)

Three days of pose-side probes (priors, endzone weights, joint limits,
bounds tables) could not put the runner's legs under him. The footage
overlay's second look (`diag/overlay_ez2_0.3`) showed the two-view records
40-60 px off the player although their own sideline reprojection was 3 px:
the fit had floated 0.2-0.5 m off the turf to satisfy both cameras and the
box-point pin at once. Chasing that:

- **Ray-miss ruler** (`scratchpad/miss_ruler2.py`, now
  `calibration.endzone_paint.player_rulers`): per frame, the two cameras'
  rays through every confident keypoint of every paired player -- their
  miss distance, and the heights the closest points give hips and ankles.
  Play 1 as exported: miss 0.21 m at the median, 0.45 m while the endzone
  pans (frames 255-290), triangulated hips 1.7 m, ankles 0.3-0.8 m.
- **The endzone camera's paint** (`scratchpad/ez_paint_overlay.py`,
  `diag/ez_paint/`): the projected yard lines 40-85 px off the painted
  ones on EVERY frame, rolled, the hash rows 150 px off their columns. The
  camera came from the players' feet against the sideline's placement
  (from_players) with the mount centre a grid PRIOR at (60, 0, 20).
- **Rejected on the way**, each measured: refine_paint on the endzone
  (rotation + focal, centre held): grid 56 -> 28 px and the rays miss 5x
  WORSE (0.21 -> 1.06 m; a pencil of parallel lines does not see the
  lateral axis). The ray miss alone as an objective: degenerate, the fit
  rotates the camera up so the rays meet 5-30 m in the air. Yard lines +
  rays + an ankles-on-turf prior with the centre free: the lines
  registered one line off at the far end (nearest-line assignment, 60 px
  start on 60 px spacing), fits rolled the grid. A field crown: the two
  cameras' box-bottom ground points differ by a constant, not by distance
  from the centre line -- no crown.
- **What works** -- `scripts/08l_endzone_paint.py`
  (`calibration.endzone_paint`): yard lines clustered by their row at the
  image centre and registered by consecutive counting from the red end
  zone's edge (the goal line is the first line BELOW the red, which stops
  25 px short of it; its own white edge and the letters make lines above),
  the hash-mark dashes (grey > 150 -- they are thin and blurred -- RANSAC
  per column) for yaw and lateral position, the mount centre solved once
  over 12 frames, rotation and focal per frame from the neighbour's camera
  with two registrations tried and the paint picking, deltas smoothed over
  9 frames. Play 1: centre (60, 0, 20) -> (88.3, 0.6, 20.8), focal 9.9k ->
  15.2k, paint 2.3 px, dashes 1.3 px on 501/510 frames. The players, which
  the fit never saw: ray miss 0.213 -> 0.135 m, ankles +0.33 -> +0.05 m,
  hips 1.06 -> 0.83 m; the pan frames 0.10-0.19 m. Applied to play 1;
  original kept as `cameras_endzone_players.npz`. 08l refuses to write
  when the players do not confirm the paint.
- **The clip offset.** With the camera right, 05n's offset search sat at
  the edge of its range (-3, then -10 when widened, monotone): its ruler
  scores every player's reprojection and standing players meet at any
  offset. `scripts/05o_clip_offset.py` (`pose.clip_offset`) uses only
  (frame, player) pairs with sideline ground speed >= 3 m/s: a clear
  minimum at -14/-15 frames (0.110 m; 0.29 at +14, 0.53 at -70). The
  endzone clip runs 15 frames (0.25 s) AHEAD of the sideline where +3
  behind had been assumed. A runner at 8 m/s is 2 m from himself in 15
  frames -- that WAS the runner (id 9 paired 2.4 m off; 0.74 m re-paired
  at lag -15). The pairing, the triangulation's 42 % ceiling, the "two-view
  wobble" and the 1 m pairing ambiguity all carried this.
- **Re-pairing without re-running the GPU stages**: `scripts/08m_relabel_caches.py`
  (`tracking.relabel`) carries keypoints and pose caches to the new ids by
  their boxes (raises when the boxes differ; refuses under numpy 2 -- the
  caches are numpy-1 pickles).
- **The box-point anchor is biased** (2399 paired frames whose ankles
  agree to 0.3 m): the sideline's box-bottom ground point sits 0.31 m from
  the triangulated ankle midpoint (0.54 m pre-snap, the stances), the
  endzone's 0.44 m along its depth; the sideline's ankle ray at z = 0.08
  sits 0.06 m off (pre-snap 0.32). 05p's two-view pass now anchors on the
  triangulated ankles where both cameras see them (pin weight 10 -> 2);
  the one-view pass keeps the box (a separate change, measured next).

Pipeline: stages `endzone_paint` (after check), `offset` (after keypoints),
`repair` (08i --lag, 08m, 08c --from-cache; before tri); 05n reads
clip_offset.json. v26 = all of the above + 05p --two-view --endzone-weight
1.0 on the re-paired play (`scratchpad/p1_v26.sh`, `p1_v26b.sh`); strips in
`diag/overlay_v26/`, clip `diag/play_001_v26_hifi_720.mp4`.

**v26 sent (2026-09-09 20:20)** with the runner strip: the first version with
the runner's legs under him (frames 260-269); 272-281 and 245-257 still
flail. Two-view: 20 players, 1314 records, sideline 7.1 px / endzone 4.8 px;
1221/1546 frames anchored on the triangulated ankles. Probes on the runner
after v26, each judged on the strips against v26's (`diag/overlay_probe_*`):

- Keypoint filter judging by the nearest pair (kept the runner's ankles):
  WORSE -- the detector's left/right labels flip for 2-4 frames on 11 % of
  his frames (arms; legs 8 %) and the window-3 median had been catching
  those. Reverted.
- A refit from the rigid start when the warm-started frame is worse than
  10 px: WORSE -- lower rms with wider legs on one-view frames; the warm
  start is the regulariser the depth ambiguity needs. Off (RESTART_PX inf).
- Anatomical joint-range prior at weight 10 in the two-view fit: WORSE --
  83/222 two-view frames accepted, sideline 24.6 px.
- A flip-aware left/right rule (`keypoint_filter.fix_lr_flips`, 05p
  --fix-lr): swaps a limb group's labels where the swapped ones continue
  the last two frames' prediction and the given ones contradict it; finds
  826 flips on play 1 (the runner 111/441 frames). Probe running.

Also measured: only ~9 of 22 players carry an endzone partner per frame.
15 endzone tracks sit 14-15 m from every sideline track (the bench); 21
endzone tracks clash on kit with every candidate within 2.5 m (BAL #0's
270-frame endzone track has no white sideline track within 2.5 m: the
sideline does not track him there); the rest are fragments. Pairing v27:
numbers outrank a kit clash, a same-person continuation shares the span,
median 2 m / lateral 1.2 m gates on ankle ground points -> 37 pairs, the
worst at 1.33 m but one.

**Later the same evening (v27 running, GPU paused at the user's request at
21:28 with the v27 render at 154/324 frames -- resumable):**

- The side-agnostic left/right residual (Mono2DConfig.lr_symmetric, 05p
  --lr-symmetric): no help on the runner (probe v27sym). His bad frames
  258-270 have bad sideline keypoints (another player covers him; the fit
  cannot reach them, 10-35 px), not flips. Off.
- The 9-frame pose smoothing was a moving MEAN of axis-angle vectors: it
  smeared the runner's legs (limb reprojection 19 -> 25 px p50, 44 p90; a
  leg swung out where the stride turned) while halving a lineman's jitter
  (limb speed in the body frame 1.24 -> 0.50 m/s). Now a moving MEDIAN,
  window 7: the runner as the fit put him (19 px), the lineman 0.66 m/s.
  Ruler: `scratchpad/smooth_ruler.py`.
- The remaining 19 px on the runner: the fit used the regressor's betas
  (~1.72 m) and the timeline rendered the roster build (1.85 m default for
  an unnamed id) -- every rendered body sat ~20 px above its own
  keypoints (`diag/runner_f242_labelled.jpg`). 05p fits with the roster
  build now (render.roster_shape.roster_builds; --no-roster-betas to
  compare). Probe v27roster measures it.
- The pipeline's refit_mono runs the two-view fit by default
  (ENDZONE_WEIGHT=1.0; ONE_VIEW=1 for the v25 mode).

- **The fit put the ankle joint on the turf; the render puts the sole on it.**
  SMPL-X's ankle joint sits 0.08 m above the sole (a pointed toe 0.19 m),
  so every fitted body stood through the turf and the render lifted it
  6-15 px above its own keypoints (`diag/runner_f242_labelled.jpg`: the
  whole skeleton shifted up, the pose identical to the record). The roster
  build was not it (probe v27roster: 20.6 px). `fit_mono2d.sole_height`
  (ankles - 0.08, feet - 0.02) is the ground term and the rigid start's
  placement now: the runner's skeleton sits on him (probe v27sole), the
  lineman's limbs 4.7 -> 2.3 px. The ankle-ray ground anchor (ankle at
  0.08) and the ground term agree at last.
- The pose median holds only where a component turns less than 0.5 rad in
  the window; a fast limb stays raw (runner limbs 17.1 -> 14.7 px).
- A tighter one-view gate (8 px rms, --one-view-px-max): no help -- the
  runner's occluded frames (258-270) pass an all-joint rms gate while their
  arms alone sit 10-35 px off. Gate stays 20.
- **v28 poses (CPU, 22:59-23:24):** two-view 21 players, 1813/1859 frames,
  sideline 5.9 px / endzone 3.9 px (v27: 6.5 / 4.6); one-view 3097 frames at
  2.7 px (3.2). Strips `diag/overlay_v28/`: the lineman (12) and a walking
  BAL player (2) sit on their keypoints to the pixel; the runner (9) is on
  his body except 258-270 (occluded, arms follow bad keypoints). The render
  waits for the GPU (`p1_v28_render.sh`).
- **The runner's 258-270 were the two-view EDGE, not the occlusion.** The
  one-view pass cross-faded its fit toward the bordering two-view record over
  12 frames -- body pose included (w = 1 - d/13) -- and pulled the pose
  toward that record at ten times the usual init weight. A sprinter got a
  stride phase from 0.2 s later: f260 wrists 73 / 59 px off while the fit
  itself sat at 3 px; every probe of the evening (filter, restart, joint
  limits, flip rules, gates, per-joint rejection) looked at those frames and
  could not move them. The edge now carries its HEADING and placement only
  (blend_params(pose=False), EDGE_BLEND_POSE): f260 arms 40 -> 8.5 px, 264
  38 -> 14, 266 36 -> 19 (probe v28edge; the rest is the occlusion, arms
  filtered out and held by the temporal term). Per-joint rejection
  (--joint-reject-px) stays off: no effect there.
- **v28b poses done (01:46):** two-view 21 players, 1813/1859 frames,
  sideline 5.9 px / endzone 3.9 px; one-view 3097 frames at 2.6 px. Strips
  in `diag/overlay_v28/`. `poses_refit.json` is v28b (v28a kept as
  `poses_refit_v28a.json`, v27 as `poses_refit_v27.json`). The hi-fi render
  is the only thing left and needs the GPU: `bash scratchpad/p1_v28_render.sh`
  (~50 min), then send `diag/play_001_v28_hifi_720.mp4` and the runner strip.

- Prepared, CPU only: `scratchpad/p1_v28_poses.sh` (the full 05p with the
  roster build and the sole on the turf, teams, strips) and
  `p1_v28_render.sh` (the GPU render, on request).

Open after this: the one-view anchor from the ankle ray (ruler above);
the pairing's ground points from ankles (its 2 m gate feels the 0.8 m
box bias; ids 14, 25, 73, 78 still 2.5-3 m mis-paired at lag -15); 08c
named 28 ids after the re-pair (check against the previous run); frames
past 510 (the play over, 8-45 pairs) stay poor on every ruler.

### The footage as the ruler (2026-09-09)

The user on v24: still jittery, arms all over the place, compare against
the All-22 footage. `scripts/05q_overlay_footage.py` projects the timeline's
bodies (the states 05k renders, placed the same way) through the sideline
camera onto the footage as skeletons, with the 2-D keypoints as dots;
`--player ID --start F --count N --step S` gives one body over consecutive
frames. `diag/overlay_v24/`. What it showed at once:

- Refit-placed bodies 1.5-3 m toward the camera on some frames (ids 2, 4:
  feet 50-76 px below the real ones): `place_from_refit` accepted a
  record's pelvis up to 3 m from the box point; MAX_REFIT_SHIFT_M is 1.0
  now (the box point sits 0.52 m from a right pelvis at the median).
  Median over all states: ankles 13 px above the box bottom, pelvis 0.24 m
  beyond it.
- A lineman in his stance (id 12, frames 130-151): the sideline keypoints
  do not move, the skeleton's torso and arms swing frame to frame. The
  fused refit reprojects into the sideline camera at p50 8.4 px, p90 15
  (arms p90 19) -- the triangulation's own level, 10 cm on a 140 px body
  -- and the 3-D fit follows every spike of it. The keypoints are the
  stable signal; fitting them directly in both cameras with the temporal
  term, instead of triangulate-then-refit, is the next step.
- The QB (id 5, frames 180-201, standing still in the shotgun): the
  keypoints sit on him, the skeleton's feet reach 40 px below his shoes --
  the two-view refit places him 0.2-0.5 m toward the camera, because the
  endzone track paired to him is Pacheco's (the pairing's ~1 m ambiguity
  between two backfield players), so both his triangulated pose and his
  placement mix two people. The one-view fit would put him on his own box
  point (unbiased against a right pelvis, 0.52 m at the median).

Two experiments, judged on the same rulers (sideline reprojection per
record, the merged cache's jitter, the boundary ruler, and the id 12 /
id 5 strips): `05p --two-view` (every two-camera player fitted straight
to both cameras' keypoints, 05f's records replaced; probe on ids 12 and 5:
sideline 8.1 px, endzone 8.5 -- the views disagree by that much through
the calibration, no fit hugs both) and `05p --one-view-only` (every body
from the sideline alone; the render camera sits on the sideline's side,
where the one-view depth error is foreshortened). Results:

- two-view direct fit (00:19-00:48): 23 players, only 1440 of 1976 frames
  under the 20 px gate (the two views disagree past it on a quarter of the
  frames), sideline 8.6 px / endzone 8.7 -- the same as 05f's chain; the
  id 12 strip is pixel-for-pixel the v24 one: the wobble is the two views'
  disagreement, and averaging the views cannot remove it. Not adopted.
- one-view only (00:48-01:18): 52 players, 5166 records, sideline
  reprojection p50 2.0 px, p90 4.8. The strips (`diag/overlay_oneview/`):
  the lineman's skeleton holds still on him, the QB stands on his own feet;
  a walking corner (id 2) still had his skeleton 60 px below his feet --
  the timeline placed him on the two-camera AVERAGE (1.9 m endzone offset
  on that pair) and refused the record's pelvis against it, and the fit
  itself let the reprojection drag the pelvis 0.38 m off the box point
  along the ray (place_weight 1). ADOPTED with two changes: the timeline
  places every id the sideline sees on the sideline's own point (100 % of
  records placed, median shift 0.38 -> ?), and place_weight 10 holds the
  pelvis on the box point (the run with it, 03:07-03:28: 5166 records at
  1.9 px p50 / 4.7 p90; the corner's skeleton now stands on his shoes,
  `diag/overlay_oneview2/player_2_f300.jpg`). Pipeline: ONE_VIEW=1
  (05p --one-view-only); the triangulation and 05f stay as the validation
  reference. The two-view depth is given up where the pair was right; the
  render camera on the sideline's side foreshortens that error.
- Even with the pelvis held, every skeleton's feet sat ~15 px below the
  shoes (the QB, the corner, the lineman alike). Measured: the detection
  box ends 16.7 px below the lower ankle keypoint at the median (138 px
  boxes); the ankle is 6 px above the sole, so 11 px is the detector's
  margin -- 0.078 of the box height -- and through the camera it is ~0.15
  m toward the lens for every body. `ground_positions` now takes the foot
  at bbox_y2 minus 0.078 of the box height (BOX_MARGIN_FRAC); the fit and
  the timeline share it. Run 3 (the fit on the corrected points, redone
  after a session death): 5166 records at 1.9 px; the corner's and the QB's
  skeletons now sit on their bodies with the feet on the shoes
  (`diag/overlay_oneview3/`). v25 (16:39, `diag/play_001_v25_hifi_720.mp4`,
  sent; strip against v24 `diag/p1_v24_v25_strip.jpg`): 24 bodies a frame,
  postures follow the footage, no arm arcs on the line. **v25 is the
  deliverable**; the pipeline reproduces it with ONE_VIEW=1 (default).

**The runner toward the camera** (2026-09-09 evening, strips on v25's
cache, `diag/overlay_v25b/player_9_f260.jpg`): id 9 running toward the
sideline camera gets a collapsed skeleton from frame 266, legs splayed at
the lens, while the keypoints on him look like a runner. One view cannot
tell short, foreshortened legs from legs pointed at the camera, and
nothing forbade the impossible angles. `pose.pose_bounds`: per-component
body_pose bounds at the 2nd/98th percentile of the two-camera refit
records (2963, 24 players); the one-view records sat outside them on 54 %
of components. As a soft prior (`bounds_weight`): weight 1 does not fix
the runner (the reprojection still wins), weight 10 neither: the legs
still splay (`diag/overlay_bounds_w10/`) -- the two-camera records the
bounds come from allow a hip abduction of 1.0 rad themselves (their own
noise), so the range forbids nothing that matters. Not adopted. The
principled fix is the second camera: the endzone sees motion along y
across its image, exactly where the sideline is blind; at a LOW weight it
would act only where the sideline's constraint is null. Measured (probes
on ids 9 and 12, strips in `diag/overlay_anat/`, `diag/overlay_ez0.3/`):
- anatomical joint limits (hips, knees, ankles, elbows; weight 10): the
  runner straighter for two frames, legs still out from frame 275.
  Insufficient alone.
- the endzone at weight 0.3 in the two-view fit: the runner's legs are
  UNDER him on every frame, the arms a little wide; the lineman holds
  still with legs and torso on the keypoints, but his head is drawn as a
  spike up and back -- the endzone's head keypoint (the back of a helmet)
  pulled it. Only 95 of 194 frames passed the combined 20 px gate (the
  endzone's residuals are 17 px by design at that weight). Two changes:
  the gate is the sideline's reprojection alone, and the second camera
  does not vote on the neck and head. Re-probe: PENDING.

**The keypoints themselves jump.** Sideline wrists (confident ones):
frame-to-frame motion p50 1.6 px, p90 7.5, p99 44 px, max 94 -- a
left/right swap or a miss for one frame -- and 30 % of wrists sit under
0.5 confidence (hips and knees never do). Every fit that follows the
keypoints throws the arm there and back. `pose.keypoint_filter.
reject_outliers`: a confident point further than 18 px from the median of
its neighbours' pairwise midpoints (constant motion of any speed sits on
every midpoint; a one-frame spike does not) gets confidence 0 -- the fit
ignores it and the prior holds the joint. Play 1: 1.4 % of confident
keypoints rejected (wrists 3.3 %), wrist p99 44 -> 17 px, max 94 -> 37.
Applied in 05p (both passes) and 05n.


### v23: the user's four notes on v22 (2026-09-08 evening)

The user, on v22: (1) jitter and shakiness, players glitching in and out;
(2) the QB the wrong colour, players near each other switched around;
(3) every build the same, linemen no bigger than corners; (4) which play
is this. Measured and answered:

**Glitching in and out** (ruler: a drawn id's runs between rendered
frames): 223 disappear/reappear events over 54 drawn ids, gaps of 4-16
frames, 781 vanished frames -- 293 with a ground point and dropped by the
dedupe, the rest interpolated through a short detection gap and then
dropped by the dedupe as the neighbour's copy (linemen 0.8 m apart). Fix
(`timeline._anchored_by_frame`): a state whose id the sideline detected
within MAX_GAP_FRAMES (30) of the frame is anchored, never a duplicate;
only ids unseen for longer dedupe.

**The QB** is sideline id 5 for the whole play (frames 14-628), kit votes
42 red / 2 white, team KC, drawn red throughout; what is wrong on him is
the NUMBER: the endzone track paired to him read 10 on 7 of 7 crops, and
#10 is Pacheco, who stands beside Mahomes in the shotgun -- the pairing
cannot tell two backfield players 1-2 m apart along the endzone's depth,
and the name follows the endzone's number. Open.

**Switched around**: the kit margin per detection flips for good inside
10 of 74 sideline tracks (sustained runs of the other kit, 20-135
confident detections): the linker handed the track to another player.
`tracking.split_by_kit` cuts a per-camera track where the smoothed kit
sign holds for >= 15 confident detections (a blip of 6 is a shadow); 30
cuts on play 1 (13 sideline, 17 endzone), stage `split` (08k) between
link and identity, so numbers, partners and avatars are per player.

**Builds**: every avatar had the regressor's near-neutral girth, only the
height from the roster. SMPL-X beta1 sets girth (+-27 % of mesh volume
per +-2) almost without touching stature; the neutral body is 77 kg at
1010 kg/m^3 (BMI 26), so a roster weight is a target mesh volume.
`roster_shape.betas_for_height_weight` meets both (Travis Jones 1.93 m /
152 kg -> beta1 +2.7, 150 kg; Marquise Brown 1.75 / 77 -> -0.3). All 21
named ids carry a weight; unnamed ids keep the default build.

**The play**: BAL @ KC, 2024 week 1 (the Thursday opener, 5 Sep 2024), KC
on offence, 2nd-and-20 from the Baltimore 24; the next clips in the
folder are KC 3rd-and-9 and 4th-and-9 from the 13, so the snap gained
11 yards. Named on it: Mahomes' track carries #10 (see above), Pacheco,
Noah Gray, JuJu Smith-Schuster, Humphrey, Taylor, Suamataia; BAL Stephens,
Cooper Jr., Ojabo, Travis Jones, Agholor.

v23 = the script from the link stage with the split stage, the anchored
dedupe and the weighted builds. Its chain: 30 cuts (151 -> 181 per-camera
tracks), OCR 41 of 181, 36 pairs (30 two-camera ids, 0 cross-kit, 10
paired a frame), 28 named (5 shared names demoted, the kicker vetoed), 24
players triangulated at 65 % / 7.8 px, refit 0.096 m, 3603 one-view
records at 3.6 px, 25 states a frame.

The anchoring alone over-corrects: a ruler of VISIBLE defects per pair of
rendered frames (a body vanishing with no body within 0.6 m of where it
was; two bodies within 0.4 m) on v23's caches -- v22's rule 155 vanishes /
108 doubles; anchoring alone 39 / 360 (a second fragment id interpolated
on top of its player's detected body is now kept); anchoring with the
rule that an interpolated state within INTERP_DUP_M of a detected one is
still a duplicate: 46 / 123 at 0.4 m, 55 / 121 at 0.5. Adopted 0.4. v23's
render carries the anchoring alone (360 doubles); v24 = v23 + the radius.

**Shakiness as rendered** (ruler: the timeline's interpolated poses at
the render stride through the FK, second differences per rendered step,
hands and feet): p50 38 mm, p90 267 mm on v23 -- fused and one-view
records alike, i.e. limbs jumping a quarter metre between rendered
frames at the p90, where a sprinting limb's true second difference is
a few cm; first differences (real motion) p50 36, p90 147. A zero-phase
moving average over the interpolated axis-angles (`timeline.
smooth_axis_angles`, POSE_SMOOTH_FRAMES source frames):

| window | jitter p50 / p90 (mm) | motion p50 / p90 (mm) | 2nd/1st |
|---|---|---|---|
| off | 38 / 267 | 36 / 147 | 1.08 |
| 5 | 21 / 150 | 33 / 140 | 0.65 |
| 9 | 15 / 95 | 29 / 119 | 0.50 |
| 15 | 10 / 65 | 24 / 94 | 0.41 |
| 21 | 7 / 50 | 20 / 81 | 0.35 |

Adopted 9 (0.15 s): the jitter halves at the p90 and 81 % of the p90
motion stays; past that the smoothing eats strides. v24 carries it.

v23 rendered 20:17 (`diag/play_001_v23_hifi_720.mp4`; stills against v22
in `diag/p1_v22_v23_strip.jpg`): the linemen are visibly bulkier and the
backs slimmer (the roster builds), 25 states a frame with the anchoring
alone (some doubles at fragment overlaps, as the ruler said). v24 = v23's
caches + the 0.4 m radius + the 9-frame pose filter, render only (21:04,
`diag/play_001_v24_hifi_720.mp4`, sent; stills against v22 in
`diag/p1_v22_v24_strip.jpg`): 24 bodies a frame, 28 named, 34 ids with a
roster weight, 366 tilt-clamped states (78 before the pose filter: the
average of neighbouring axis-angles leans a few bodies past the limit for
a frame; harmless, the clamp catches it). **v24 is the deliverable.** The
QB (sideline track 5, the lone back 5.5 m behind the line) is drawn red
throughout and carries #10 from the endzone's 7-of-7 reads; whether that
body is Mahomes with a misread 15 or Pacheco could not be settled from
the crops -- the user is asked.

### v21: a wrong pair sawtoothed by a metre a frame (2026-09-08)

Ruler: second differences of the timeline's state xy per frame.
Box-placed ids p90 18 mm (smooth_xy does its job); record-placed ids p90
27 mm but p99 1365 mm -- 250 of the 253 triples over 0.5 m belong to id 9.
Id 9 is a WRONG PAIR: the sideline track runs 28 m along y (a receiver),
the endzone track stands still; the two cameras' ground points sit 14 m
apart at the median. The pairing accepted it as a number match because
its offset statistic is the mean difference vector over the overlap,
which averages out when tracks cross (mean 2 m, per-frame up to 15 m);
the fused refit then triangulated two different people, the timeline
averaged the two cameras' points (7 m from either), refused the refit's
placement against that average, and interpolated between refused
records at the odd frames: a 1 m sawtooth every frame, and the avatar
named Noah Gray (#83 read on the endzone track). 27 two-camera ids: median
distance p50 1.04 m; 4 over 2 m, 2 over 3 m (ids 9 at 14.2, 37 at 4.2).
Now: `pair_by_appearance` gates on the median per-frame distance
(MAX_MEDIAN_DIST_M 4.0; the test has crossing tracks with a shared
number); `render.pair_rule.mispaired_ids` guards a play-dir paired before
the gate (the timeline drops the id's endzone rows and says so);
`place_from_refit` interpolates only between records it accepted. Play 1:
ids 9 and 37 dropped from the fused cache and refit one-view (05p; the
fused cache rebuilt by 05f under numpy 1 first -- a copy written from the
numpy-2 venv does not load in the smplx venv). Result: the timeline's xy
second differences for record-placed ids p99 1365 -> 115 mm, one triple
over 0.5 m instead of 253; the merged cache's jitter p99 260 -> 213 mm,
hands/feet p90 103 -> 94 (the wrong pair's triangulated poses were among
the worst). v21 = v20 + this (10:58, `diag/play_001_v21_hifi_720.mp4`,
sent to the user; stills against v20 in `diag/p1_v20_v21_strip.jpg`: id 9,
the receiver, now stands where the sideline sees it; the rest identical).
**v21 is the deliverable and the pipeline's reproducible baseline** --
every rule above is in the script. PROVEN by v22 (11:00-18:04 with a
power-off in between; `scripts/pipeline_play.sh` resumed at the render):
the script from the link stage with all of today's rules -- per-camera
link, OCR 33 of 151 ids, 27 pairs with the median gate (25 two-camera ids,
0 cross-kit, 9 paired a frame), identity 21 named with the same three
demotions and the kicker vetoed, 22 players triangulated at 66 % / 8.4 px,
damped refit 0.096 m, 3609 one-view records at 3.5 px, 23 bodies a frame,
4 default-posed. `diag/play_001_v22_hifi_720.mp4`; stills against v21 in
`diag/p1_v21_v22_strip.jpg` are the same formation. **v22 is the
baseline** (v21 was assembled on v14's caches; v22 is the script).

### The sideline camera through the rendered span (2026-09-08)

Grid distance of play 1's refined sideline track at 16 sampled frames:
2.7-6.6 px from frame 20 to 620, then 69 / 99 / 93 px at 670 / 720 / 770.
The tracks (and the render) end at 660: the late drift the memory warns
of is the post-whistle pan-out, outside the rendered span. No action.

### v20: the timeline anchored the model's origin, 0.35 m from the pelvis (2026-09-08)

Found while chasing the last 0.35 m of the boundary jump (it survived the
placement weight and the sideline-only ground points): the SMPL-X rest
pelvis sits at (0.003, -0.351, 0.012) in model axes, and neither
pose.forward_kinematics nor the smplx model rotates that root offset --
the pelvis is always rest pelvis + transl, in WORLD axes. `placed_vertices`
put the model's origin at a state's xy, so a body with no refit record
(one-view sideline records, default poses) stood 0.35 m from its box
point along -y, toward the sideline camera; a record-placed body (xy =
transl) stood at its fitted pelvis. The two disagreed by 0.35 m, so a body
popped by that at every record boundary, and every earlier comparison of
"transl" against a ground point carried the constant: the "pelvis 0.46 m
off" in the 05p validation, the box point's "-0.32 m y bias" in the
ankle measurement (against the true pelvis: bias x -0.01, y +0.04, p50
0.52 m). Now `placed_vertices` anchors the pelvis joint at xy,
`place_from_refit` places records at transl + rest pelvis (per betas,
`rest_pelvis_xy`), and 05p interpolates the fused PELVIS across and beyond
a span. Play 1 re-run: the pelvis jump at the 71 boundaries 0.35 -> 0.05 m
(p90 0.14), the merged cache's jitter p99 408 -> 311 mm; the orientation
still turned 17 deg (p90 32) and body_pose 0.9 rad at a long-gap edge --
one view's keypoints against the two-view pose along the depth -- so a
one-view block now cross-fades into the fused block it borders
(`fit_mono2d.blend_params`, per-joint slerp, w = 1 at the edge to 0 a
max-gap away): boundaries pelvis 0.01 m, orientation 2 deg (p90 4),
body_pose 0.10 rad; the merged cache's jitter p99 260 mm and hands/feet
p90 103 against the fused-only cache's 255 / 102 -- the one-view records
no longer add roughness. v20 = v19 + pelvis anchoring + cross-fade (10:13,
`diag/play_001_v20_hifi_720.mp4`; stills against v19 in
`diag/p1_v19_v20_strip.jpg`: the same 22 bodies, the recordless ones
shifted onto their feet points; no regression). v19 (09:28) = v18 + the
short-gap rule, stills identical to v18.

### v19: the one-view fill-in popped at every triangulation gap (2026-09-08)

Ruler: consecutive records (within 3 frames) of one player where the kind
changes between fused and one-view. Play 1 v18's cache: 22 players carry
both kinds, 577 one-view records sit inside a fused span (frames the
triangulation dropped), 187 boundaries, and at a boundary the pelvis jumps
0.43 m (p90 1.27), the orientation 35 deg (p90 122), body_pose 3.0 rad --
a two-view avatar pops half a metre and spins whenever the triangulation
blinks. The one-view fit placed the pelvis on the box-bottom point and
started from the regressor's pose, agreeing with neither fused block. Now
(05p, fit_mono2d.prev_seq): inside the span the ground point is the fused
pelvis interpolated, the start pose the nearest fused record's, the warm
start at a block boundary the fused params; beyond the span the box point
carries the end's offset, decayed. Measured: 0.43 -> 0.37 m (p90 1.27 ->
0.78), 35 -> 18 deg; with strong inside-span weights (place 10, init 1.0)
0.36 m (p90 0.43), 18 deg -- the pelvis holds, the pose does not. The
decisive ruler: second differences at triples mixing fused and one-view
records against fused-only triples of the same players: 227 mm vs 18
(hands/feet p90 778 vs 140), whatever the weights. One view's keypoints
disagree with the two-view pose along the camera's depth and no weight
settles it; for a gap of a few frames the interpolated two-view pose is
the answer. The gaps: 132 inside spans, p50 3 frames, 77 % under 12. Now
05p leaves one-view frames inside a fused gap of <= max_gap (12) frames to
the timeline's interpolation and fills only the long gaps, and
`place_from_refit` interpolates the refit translation across such gaps.
Play 1 re-run: 105 one-view frames left to the interpolation, boundaries
187 -> 71 (long-gap edges and span ends), the merged cache's jitter p99
763 -> 412 mm and hands/feet p90 144 -> 110 (the fused-only cache: 102).
At the remaining edges the pelvis still jumps 0.36 m: the placement pull
beyond the span was the default 1.0; it now follows the decayed offset
(10 at the edge, 1 five max-gaps out).

### v18: the two-view bodies shook; the fused refit damped (2026-09-08)

Ruler: second differences of the refit's world joints per frame (a smooth
motion has them far below the first differences). The fused refit (05f,
plain per-frame least squares, warm start only) came out p50 19 mm, p90
101, p99 717, hands/feet p90 319 mm a frame -- limbs jumping a third of a
metre between frames at the p90 -- from triangulated joints (05n) at p50
18 / p90 62 / p99 293: the fit AMPLIFIED the input's spikes. Two causes in
`fuse_smplx.fit_single_frame`: no temporal term, and soft_l1 at f_scale
1.0 m, which is plain least squares for anything under a metre. Now
`SMPLXFitConfig.temporal_weight` (pull of body_pose and orient toward the
previous frame's solution, consecutive frames only) and `f_scale`; 05f
defaults 0.3 / 0.1 (the library's defaults stay 0 / 1.0). Play 1 sweep:

| refit | p50 | p90 | p99 | hands/feet p90 | rms to the joints |
|---|---|---|---|---|---|
| plain (v14-v17) | 19 mm | 101 | 717 | 319 | 0.087 m |
| temporal 0.3, f_scale 0.1 | 17 | 65 | 255 | 102 | 0.096 |
| temporal 1.0, f_scale 0.1 | 16 | 59 | 182 | 77 | 0.102 |

The rms rises because the damped fit no longer follows the spikes it is
scored against; the p90 lands at the input's own level (the fit stops
amplifying). Synthetic: a 0.40 m one-frame wrist spike followed 0.28 by
the plain fit, 0.16 damped; a real 0.40 m step followed to 0.26 within
four frames. v18 = v17 + this (05f then 05p re-run) + the identity rules
(08:07, `diag/play_001_v18_hifi_720.mp4`): 22 bodies a frame, 68 tilt
clamps, stills against v17 (`diag/p1_v17_v18_strip.jpg`) identical in
formation, #21 the only visible number either way; the damping is a
motion property, judge it on the clip.

### v16: the duplicate rule was eating six players a frame (2026-09-08)

Every render since v8 drew 16 bodies a frame and it passed as normal. An
audit of the path from boxes to states on v15's caches: the sideline has
21 boxes a frame, 20 ids after the exclusion rules, 32 ids with a ground
point; the timeline drew 16 and `dedupe_frames` had dropped 7511 states
(11.6 a frame); 47 sideline ids over 4056 id-frames were present and
undrawn. The rule treated any one-view state within its camera's
depth/across radii (4.0 m along the depth axis, 1.5 m across) of a kept
state as the other camera's copy -- built for play 2's endzone ghosts
strung along x -- and for a sideline-only id the depth axis is y, so the
linemen a metre apart along the line of scrimmage were "copies" of each
other. Now (timeline.dedupe_frames): a state whose id the SIDELINE
detected in that frame is never a duplicate (its boxes in one frame are
different people); an id the endzone alone sees that frame dedupes within
the endzone's depth/across radii of a kept state; an interpolated frame
within 0.9 m. Measured: 23 states a frame (p10 20, p90 25), 3396 dropped,
7 sideline id-frames undrawn, nearest-neighbour distance p5 0.47 m.
Render v16 = v15 + this rule (06:19, `diag/play_001_v16_hifi_720.mp4`,
strip `diag/p1_v15_v16_strip.jpg`): median 23 bodies a frame; the whole
offensive line is back at the snap (v15 drew four red bodies at 1.5 s, v16
the pack of eleven) and the 7 s frame shows a full formation. No visible
duplicates in the stills. v16 is the deliverable.

The states the new dedupe keeps that the sideline does not see in that
frame, per frame on v16: 1.1 interpolated (gap fills), 0.9 endzone-seen
PAIRED ids, 0.2 endzone-only ids the frustum test lets through. The paired
ones are one id: 68 = an endzone track of 510 frames (125-634) paired by
08i to a sideline fragment of 18 frames (606-642), drawn 474 frames from
the endzone alone -- a second copy of a player the sideline tracks under
other ids, metres away along x, past the dedupe radii. Id 40 the same
shape (endzone 125-634, sideline 398-526). `endzone_only_rule.
beyond_sideline_span` (in the timeline before the exclusion rules): an
id's frames beyond its sideline span by more than MAX_GAP_FRAMES (30) are
dropped where the sideline could see the spot, kept where it could not
(outside its image). Play 1: 1588 ground frames dropped, 22 bodies a
frame, dedupe drops 3396 -> 1841. v17 = v16 + this + the tilt prior (07:17,
`diag/play_001_v17_hifi_720.mp4`): 22 bodies a frame, tilt-clamped states
1158 -> 73, stills against v16 (`diag/p1_v16_v17_strip.jpg`) the same
formation without the endzone tails; no regression.

### v15: the one-view bodies refit to the keypoints (2026-09-08)

Half of the rendered bodies are one-view (the sideline alone) and took the
monocular regressor's pose, which sits near the mean pose: measured on v14,
their joints move 0.21 m/s in the body frame (pose only, orient and transl
zeroed, `fit_mono2d.body_frame_speeds`) against 1.0 m/s for the
triangulated bodies -- mannequins gliding across the turf. The 2-D
keypoints (05m) exist for every tracked person and carry the articulation.
`pose.fit_mono2d` + `scripts/05p_refit_mono.py` (stage `refit_mono`, after
`refit`): per frame, least squares over SMPL-X (body_pose, orient, transl)
to the sideline keypoints through that frame's camera, the lower ankle on
the turf (the depth one view cannot see), the pelvis over the box-bottom
ground point, an L2 prior, a pull to the regressor's pose, a temporal
term; started from the regressor's pose and HEADING (a 12-heading search
picked the mirrored heading on a symmetric body and the fit then extended
the arm toward the camera instead of raising it: 0.31 px vs the truth's
1.09, wrong by 0.5 m -- the synthetic test in `tests/test_fit_mono2d.py`).
Records merge into 05f's cache; the fused record wins; the 05f cache is
kept as `poses_refit_fused.json` and a re-run starts from it. A frame
whose fit reprojects over 20 px keeps the regressor's pose.

Probe (ids 0 and 8, 343 frames): reprojection of the regressor's placed
pose 34 px -> 2.5 px; body-frame joint speed p50 0.21 -> 0.55 m/s (p90 1.3
-> 2.3). Cost 0.6 s a frame (least_squares max_nfev had been max_iter x 10
and a noisy frame ran 400 iterations); stride 2 (the timeline interpolates
axis-angles between records, 05k renders at stride 2), 6 workers.
Full run on v14's caches (04:32-04:43): 42 players with sideline keypoints
outside the fused refit, 3718 of 3721 frames fit (stride 2; 3 over the 20 px
gate); reprojection median 35.6 -> 3.3 px; body-frame joint speed p50 0.17
-> 0.56 m/s (p90 1.34 -> 2.98). The one-view bodies now move about half as
much as the triangulated ones instead of a sixth. The unobserved joints
(spine, collars, feet: 8 of 22) sit on the prior and pull the median down;
the reprojection is the keypoints' own noise at 130 px bodies.
`diag/play_001_v15_hifi_720.mp4` = v14 + this stage (nothing else changed).
Jitter (body-frame joint series, median second difference over median
first difference between records): fused refit 2.16 at step 1 and 2.16 at
step 2, one-view refit 1.70 at step 2 -- the one-view fits are no rougher
than the triangulated ones (both are noise-dominated at 8-15 mm a record),
so no extra smoothing on the mono records.
Validated against the fused refit (`05p --validate`: the mono fit run on
the frames the triangulation covers, scored against it; 365 frames at
stride 8): pelvis-aligned joint error p50 0.18 m for the regressor's pose
AND 0.19 m for the mono fit -- the reference's own noise floor (the fused
refit's rms to the triangulated joints is 0.087 m and those joints carry
~0.1-0.2 m at 8 px from 100 m), so the 3-D ruler cannot rank the two; the
mono fit's measured gains are 2-D (43 -> 3 px on those frames) and, with
the tilt prior, the lean. Without the prior the one-view fits leaned 34
deg (p50) against the truth's 16 (one view trades lean against depth);
`tilt_weight` 3 -> 21 deg, 10 -> 20 deg (|diff| 7, none past 60, p90 joint
error 0.31 -> 0.26 m), adopted at 10 past 20 deg. The pelvis of a one-view
body sits 0.46 m (median) from where the two-view fit puts it: the
box-bottom ground point against the triangulated pelvis -- the placement
ambiguity of one view, not the pose. HONEST CORRECTION to the motivation
above: the fused bodies' 1.0 m/s "body-frame motion" is mostly per-frame
jitter (8 mm a frame, second differences twice the first), so the
regressor's 0.17 m/s was not a sixth of real articulation; what the
keypoint fit adds is what the stills show -- strides, bent arms -- and the
2-D agreement, not a 3-D number.
On the real one-view set the prior is a wall: 3718 records lean p50 20,
p90 21, p99 21 deg (the fit wants more lean everywhere and stops at the
free angle), where the fused bodies spread 16 / 39. The lean points toward
the sideline camera 40 % of the time (cos > 0.5; fused 23 %) and away 21 %
(fused 12 %): a camera-ward bias, the depth ambiguity, not a travel lean
(cos with the travel > 0.5: 40 % vs 29 %). Untried: a quadratic prior from
zero tuned to the fused spread instead of a free angle and a wall.
Heading (`--validate`, 365 two-view frames, the fit's facing against the
fused fit's on the ground plane): |diff| p50 21 deg, p90 51, facing the
wrong way (over 90 deg) 3 % -- the regressor's heading prior plus the fit
gets the facing right on 97 % of frames; with the tilt prior the
pelvis-aligned joint error is 0.16 m against the regressor's 0.17.
Render (05:21): 69 posed players (v14 55), 3 default-posed, 16 bodies a
frame as before. Stills against v14 (`diag/p1_v14_v15_strip.jpg`, temporal
crops `diag/p1_v14_v15_motion.jpg`): the one-view bodies articulate --
running strides, bent arms -- where v14's walked upright; no flailing, no
broken bodies. v15 is the deliverable.

### v11, v12, v13: the endzone track decided by triangulation (2026-09-08)

Three runs of play 1 through the same chain (per-camera link, OCR,
appearance pairing, identity, pose, keypoints, tri, refit, hifi), differing
only in the endzone track:

| | v11 | v12 | v13 |
|---|---|---|---|
| endzone track | interpolated, mount y=4 | footage (08h), from y=0 | interpolated, y=0 |
| players ruler (m) | 1.68 | 1.38 | 1.41 (v12's re-solve) |
| pairs / paired per frame | 23 / 9 | 23 / 6 | 24 / 10 |
| players triangulated | 18 | 15 | 21 |
| observable joints passing / reproj | 60 % / 6.7 px | 58 % / 10.4 px | 62 % / 7.9 px |
| refit rms | 0.087 | 0.091 | 0.085 |
| named | 29 | 25 | 26 |
| bodies per frame | 16 | 16 | 16 |

The footage-driven track loses on the geometry that matters although it
passes the players ruler and holds pixel-static players to 1 cm: the
feet-fitted anchors are locally right for the paired players, and a
single pose fitted from a capped nearest-neighbour objective is not. It is
opt-in now (`ENDZONE_TRACK=1`). The mount's across-field refinement stays
off (y = 0 pairs more, triangulates more). **v14 is the current deliverable
and the pipeline's reproducible baseline** (v13 plus the pairing overlap
floor at 6 frames, 67dab0d): 29 pairs, 27 two-camera ids, 0 cross-kit, 22
players triangulated at 63 % of observable joints and 8.4 px, 26 named, 16
bodies a frame. `diag/play_001_v14_hifi_720.mp4` (v13 kept beside it; strip
`diag/play_001_v10_v11_v13_strip.jpg`).

Open, in value order: two-view coverage (10 of 22 paired per frame; the
sideline OCR reads no numbers at 130 px bodies, so pairing rests on kit
and position), fragments (74 sideline tracks for ~29 people), the pose
of one-view bodies (monocular). Officials inside the field are NOT among
the unnamed white sideline ids on play 1 (all eight darkest are Baltimore
players, `diag/p1_v13_white_unnamed.jpg`): the sideline does not track
them as separate bodies, so the kit fault reduces to the boundary staff
and the sideline official, both excluded.

### v11: the script end to end, and what the fresh run taught (2026-09-07 evening)

The first fresh `--fresh --from-paint` run of play 1 through the new stage
order failed twice before it ran: the launcher seeded play 1 from itself
(`SEED_FROM=play_001`; circular, a 52 px grid where the free solve gives
9 px) and the numeral ruler crashed on an empty strip from a wild
candidate (`yard_numbers.read_line_strips`, guarded, 9800092). Then it ran:
paint (candidate 1: (42, -102, 42.5), 11.8 deg, rulers 0.97/0.96, players
1.86 m, grid 9.2 px), refine, shift, endzone, check, endzone_track
(REFUSED, chain continued: 23c83f8), link (153 camera tracks), identity
(OCR 34 of 153 read, 23 pairs of which one on a number, 21 two-camera ids,
0 cross-kit, 29 named), pose, keypoints, tri (18 players, 60 % of
observable joints at 6.7 px), refit, hifi.

The sideline reproduced to the pixel against v10's cameras. The endzone
did not: the re-solve's refinement moved the mount to y = 4 on a height
tiebreak, and on the whole play that is worse (players 1.68 m at y = 4,
1.56 m at y = 0, 1.35 m with the footage-driven track from y = 0), so
`08h` could not beat the interpolated track from that start (1.77 m one
start, 1.82 m nine starts, 1.73 m seeded from v10's pose). The across-field
refinement is off (`REFINE_DY_M = (0.0,)`, bc09480): the feet do not see y
on 14 anchor frames; the synthetic justification was noiseless. v12 re-runs
the chain from the endzone stage with y held at the seed.

### The pipeline's stage order since feabdfc (2026-09-07)

`scripts/pipeline_play.sh <play-dir> <side.mp4> <end.mp4> <los> [--fresh]
[--from-paint]`, markers `.done_<stage>`; env RED/WHITE, SEED_FROM,
GRID_PX, FUSE, FIT, DIAG:

paint (08, --from-paint) -> export (08b from recon) -> refine (08e) ->
shift (08d --apply) -> endzone (08 --sideline-from, then 08b again) ->
check (08d rulers + LOS; two-of-three) -> **endzone_track (08h: the
endzone camera from the footage's motion; refuses to write unless the
players get closer)** -> **link (08b --cameras cameras.npz --pairing track
--pair-gap 0: camera tracks keep their ids; the export's tracks kept as
tracks_export.parquet)** -> field (05l; LOOK at the PNG) -> **identity
(08c OCR per camera track -> 08i pairing by number then kit -> 08c
--from-cache names on the paired ids; BEFORE the pose stages because
pairing changes the ids and every pose cache is keyed by id)** -> pose_s
-> pose_e -> keypoints (05m) -> tri (05n) -> [fuse, FUSE=1] -> refit (05f,
6 joints / 50 % frames) -> [fit, FIT=1] -> teams (08f) -> hifi (05k, 720p
encode into diag) -> render (05d). The script's header comment still lists
the old order; it is edited only when no instance runs. v11 is the first
fresh run through this order (2026-09-07 20:14).

### Play 1 v9 and v10 delivered (2026-09-07)

| | v8 | v9 | v10 |
|---|---|---|---|
| endzone track | interpolated anchors | footage (08h) | footage (08h) |
| link | per frame, old cams | per frame + kit gate | per-camera tracks, appearance pairing |
| two-camera ids / cross-kit | 62 / 24 % | 84 / 23 | 21 / 0 |
| players triangulated | 47 | 36 | 15 |
| observable joints passing | 65 % | 66 % | 57 % |
| named | 31 | 49 | 33 |
| timeline bodies / frame | ~29 | 21 | 17 |
| excluded ids (edge / ghosts / dwellers / striped) | 23 / - / - / - | 30 / 32 / 26 / 0 | same rules |

Files: `diag/play_001_v9_hifi_720.mp4`, `diag/play_001_v10_hifi_720.mp4`,
strips `diag/play_001_v8_v9_strip.jpg`, `play_001_v9_v10_strip.jpg`. v10 is
the cleaner baseline (right kits, no ghosts, no staff, one body per
sideline track); its cost is two-view coverage, so most poses are
monocular. The 17 bodies per frame are the sideline's 22 tracked ids minus
excluded non-players -- what the sideline frames.

Measured and not adopted today: same-camera stitching with kit and number
vetoes (welds one wrong pair per right one on named fragments); depth
whitening before linking (fewer tracks by welding on the helmet set:
purity p10 0.49 -> 0.44, switches 191 -> 245); a wider kit gap for the
appearance pairing (22 -> 40 pairs, 8 -> 9 players paired per frame).
Sideline OCR reads 3 numbers to the endzone's 34 at 130 px bodies; an
upscale/band sweep was measured (`diag/sideline_ocr_exp.log`).

### Who is drawn: four exclusion rules, and v10's order (2026-09-07)

`play_timeline` now leaves out, in this order, each a set of ids: edge-clipped
one-view ids (`edge_rule`), endzone-only ids (`endzone_only_rule`, one avatar
per sideline track), sideline dwellers (`offfield_rule.sideline_dwellers`:
share of frames at |y| >= 23.5 m of at least 0.8 -- play 1: 26 ids, staff
in dark jackets and the sideline official, no roster-named player) and
striped officials (`offfield_rule.striped_ids`: torso gradient ratio 2.5
with a dark share 0.25; play 1: none beyond the dwellers). Measured
against the roster-named ids: none touched.

Appearance pairing (`tracking/pair_by_appearance`, `08i`): per-camera
tracks pair on the jersey number read in both cameras (3.5 m), else on
kit agreement (2.0 m), never across a kit or number conflict, never on
position alone. Dry run on play 1 (kit only): 44 pairs, 0 cross-kit, 10
paired per frame (the per-frame link: 84 two-camera ids, 23 cross-kit).
v10 is the first play through the new order: `08b --cameras --pairing
track --pair-gap 0` (camera tracks keep their ids) -> `08c` (OCR per
camera track) -> `08i` -> `08c --from-cache` -> pose -> keypoints -> tri
-> refit -> hifi. The pipeline patch for these stages
(`scratchpad/patch_pipeline_stages.py`) adds `endzone_track` and `link`
stages after the check and moves identity ahead of the pose stages
(pairing changes ids; pose caches are keyed by id); apply it when no
`pipeline_play.sh` instance runs.

### The 42 % triangulation figure has a structural ceiling (2026-09-07)

On play 1 v9 (58 two-view players, 104k joint-frames): 8 of the 22 SMPL-X
body joints (spine x3, collars x2, feet x2; the head only through the
face, 37 % confident) have no COCO-17 keypoint and never triangulate, so
the raw share tops out near 64 %. Over the 14 observable joints: 61 % are
confident (>= 0.3) in both views and 68 % of those reproject within 20 px
(median 13 px, p75 23, p90 37) -- 13 px at these focals is 0.15 m at the
player, keypoint noise not camera error. So the triangulation passes 66 %
of what is observable; `05n` now prints both shares. The endzone track
change moved neither (v8 42 %, v9 43 %). Chasing this further means better
2-D keypoints on 140 px people, not geometry.

### One avatar per sideline track; the pairing stays ambiguous (2026-09-07)

Play 1 with the footage-driven endzone track written (`cameras.npz`, old
track in `cameras_endzone_interp.npz`) and re-linked three ways on the same
cameras (`diag/p1_frame`, `p1_track`, `p1_track25`):

| link | ids | in both cams | cross-kit | paired / frame | one-view / frame |
|---|---|---|---|---|---|
| old cameras, per frame | 106 | 62 | 24 % | 15 | 14 |
| new cameras, per frame + kit gate | 176 | 84 | 23 (27 %) | 13 | 17 |
| new cameras, track pairing 2.5 m | 147 | 29 | 8 (28 %) | 9 | 23 |
| new cameras, track pairing 1.0 m | 166 | 17 | 6 (35 %) | 5 | 31 |

The better endzone track did not make pairing decisive: box bottoms give
about 1 m of depth error per camera and players stand 1-2 m apart, so a
third to a half of the players stay unpaired in each camera and the
cross-kit share of the pairs made is unchanged. The kit gate only bites
where both boxes carry a label (63 %), and the position-only linker joins
across kits over time. What the numbers do say: the timeline was drawing
both copies of every unpaired player -- 37 ids a frame, 18 endzone-only,
each a ghost at the endzone's poor x. `render/endzone_only_rule.py`: an id
the endzone alone sees, never two-view, is left out (same shape as the
edge rule); one avatar per sideline track, the endzone kept for
triangulation where it pairs. Play 1 v9 runs on the per-frame kit-gated
link with the new cameras: rulers are triangulated joints passing (v8:
42 %) and avatars per frame (v8: about 29 for 29 people plus ghosts).

Next for pairing: appearance at track level -- the jersey number read in
both cameras must agree and the kit must match -- before position decides;
the OCR runs after the link today (08c), so the order changes.

### 08h works once the camera-from-homography step is a ray fit (2026-09-06/07)

Every "pose fit refused" verdict on 08h was scored on broken motion.
Bisected on play 2 frames 353-360: a 0.4 px homography step came out of the
matrix route (K_t R_t = H^-1 K_ref R_ref, rows normalised, SVD polish) as
0.18 deg of rotation, 0.9 m on the ground at 300 m; the polish, not the
translation, set the rotation near the identity. `rot_focal_from_homography`
fits (rotation, log focal) to a pixel grid's rays instead: a pixel-static
player holds to 1 cm (commit a2fc4ee, tests in
`tests/test_endzone_track_rayfit.py`). Motion by consecutive-frame steps
composed outward (394-482 inliers per step); direct links to the reference
alias on the periodic yard lines at long gaps (play 1, 475-535: 7-21 px).
Composed steps drift over a play's second half (play 1: 7-90 px by frame
655, with the anchors right there), so the chain is anchored every 10
frames where a direct link's ray fit is under 2 px and blended between
anchors. First passing players rulers: play 1 1.19 -> 0.90 m (share within
1 m 0.43 -> 0.55) on the plain chain over 353 frames; play 2 2.35 -> 1.68 m.
Also measured: on play 2's reference frame the old endzone camera has the
field rolled 6 deg and the hash columns 200 px inside the painted ones
(`diag/p2_ref353_overlay.jpg`); paint alone cannot fix the focal at a 4 deg
lens (six points over 6 x 9 m at 300 m ran the solver to its bound), so
depth stays with the sideline's players.

### The endzone camera track invents motion; the pairing is the wound (2026-09-05, later)

Directive from the user (relayed): play 1 only until it works properly; no
compute on other plays. Play 5 was delivered (`diag/play_005_hifi_720.mp4`)
and its chain stopped; play 2 served as the diagnostic bench (CPU only).

**Kit labels in the TIME linker are dead** at any label quality: helmet set,
five plays, ten views (`07j`, `data/helmet/tracking_accuracy_kits.json`):
six views refused the saturation split (both teams in white, separation
1.2-1.4, the refusal in `team_color.saturation_split` at 1.6 / gap 50), and
on the four that split the hard gate was worse (purity p10 0.45 -> 0.37) and
a soft 1 m cost neutral to slightly worse. `08b --kit-link` defaults to off.
The kit's place is the cross-camera pairing veto: 53 % -> 17 % cross-kit ids
on play 2 (`--kit-link block` reaches 0 of 71 at +15 fragments).

**Why the pairing is a coin flip.** For the same player the two cameras'
foot points sit 1.4 m apart per frame with no bias; per frame only 10-12 of
22 players pair; most ids are one-view and render twice. Per-camera tracks
paired by trajectory (`tracking/pair_tracks.py`, `08b --pairing track`,
commit 471c6cf) found only 5 pairs on play 2, because the offset between
the cameras for one player is not noise but a slow swing: pre-snap a
lineman who is pixel-static in the endzone view (1 px) has a ground point
wandering 1.7-2.6 m, and a fixed endzone pixel's ground point swings 4 m in
a second while the sideline's holds to 0.1 m. The picture
`diag/p2_endzone_353_420.jpg` shows frames 353 and 420 of the endzone with
the SAME camera pose to the pixel while the old track claims 5.4 deg of
rotation between them. The endzone track (14 sparse paint solves, focals
jumping 5-10 % between neighbours, interpolated by 08b) invents motion.
The sideline is trustworthy (refined every frame; its players are static
to 0.04 m pre-snap).

**The fix in progress: `scripts/08h_endzone_track.py`.** Every endzone
frame registers into one reference through `calibration/endzone_mosaic`
(players masked, 170-1200 inliers, H near identity where the camera is
still); the motion between frames is the footage's; the absolute pose is
the open part. Tried and refused by the players ruler (median distance from
each sideline player's ground point to the nearest kit-consistent endzone
one, `08h` refuses to write unless it improves): (a) a per-frame focal
scale (no effect); (b) the mean of the old track's frames transported to
the reference (2.35 -> 3.07 m on play 2: the anchors scatter 3.7 deg, their
mean is not a pose); (c) a 4-parameter fit of the reference focal and
rotation on the players with the centre held (2.35 -> 2.61 m play 2, 1.18
-> 2.50 m play 1). The old track's per-anchor paint fits compensate a wrong
centre frame by frame, which a single pose cannot, so the running
experiment (`scratchpad/endzone_static_pose.py`) fits 7 parameters (focal,
rotation, centre) on a static pre-snap window where camera and players
both hold still, with restarts, then propagates through the mosaic and
scores the whole play. `08b --cameras cameras.npz` (commit da798e5) links
with per-frame cameras once a track passes. Play 1 is re-rendering
meanwhile with rule D's kits (keypoints -> tri -> refit -> hifi).

## What has been measured and rejected (do not re-propose without new evidence)

- **Merging "twin" tracks by box geometry (2026-09-10, play 1, sideline):**
  three ids look like one body under two ids (16 & 22 the quarterback,
  18 & 19, 21 & 28), so two avatars stand on one man. A rule of "box centres
  within 0.25 box heights and IoU >= 0.4 over 20+ common frames" catches them
  -- and also catches 14 & 27, which the footage shows to be TWO Baltimore
  players one behind the other (gap 0.19 heights, IoU 0.66). From the sideline
  the line of scrimmage is seen end-on, so bodies stack in the image at every
  depth: box geometry cannot tell a duplicate from two men in a line, and the
  ankle ground points do not separate them either (16 & 22 sit 0.96 m apart,
  the genuine pair 14 & 27 1.28 m). The one test that did separate them is
  the KEYPOINTS: 16 & 22 agree to 8 px (0.05 box heights), 18 & 19 to 50 px
  (0.40) -- but the pose detector's own suppression emits keypoints for only
  one of a twin pair on all but 3-4 frames, so the test almost never fires.
  Not adopted. Anything built here must be judged on bodies drawn vs bodies
  in the footage, per [[corrections-must-beat-what-they-correct]], and the
  v16 duplicate rule (ground distance) is the cautionary case.

- **The regressor's body pose as a cross-camera pairing cue (2026-09-08,
  play 1, 85 common posed frames, 885 same-id pairs vs 47k other-id):**
  pelvis-relative joint distance same p50 0.109 m, other p50 0.130, 30 % of
  others below the same-median; the nearest endzone pose is the true
  partner 18 % of the time (chance 4 %). A weak cue on its own; untested
  in combination with the position gate, and there is no pair truth on
  play 1 to test it with (the helmet set has both views labelled -- the
  instrument if this is ever pursued).

- **Ankle keypoints as the one-view ground point (2026-09-08, play 1, 2773
  two-view frames, judged against the fused refit's pelvis):** box bottom
  p50 0.55 m (p90 1.35), mean of the two ankle keypoints dropped 8 cm to
  the turf p50 0.53 (p90 1.45), the lower ankle 0.62. No gain. (The
  "opposite biases" first reported here, box -0.32 / ankles +0.28 along
  y, were the rest-pelvis offset: that comparison used transl, not the
  pelvis; against the true pelvis the box point is unbiased, x -0.01 y
  +0.04, p50 0.52 m.) The 0.5 m is the placement ambiguity of one view
  against the hips' triangulation, not the choice of pixel.

- Officials by shirt stripes (horizontal-gradient energy of the torso band): a
  continuum on real crops, players on top, at 140 and 200-260 px bodies.
- Per-frame camera refinement to the yard lines alone: the pencil of near-parallel
  lines trades rotation about their direction against focal length; it needs
  priors, a 3 deg / 20% reject and along-track smoothing, or it wanders 2 deg.
- The endzone camera refined to its own paint (2026-09-05, 623eab0, kept as
  `08e --cam endzone --orient-tol 25 --out`): the paint improves (play 1
  median grid 39 → 23 px) and the feet refuse it -- triangulated lower ankle
  p10 −0.01 → −0.45 m, ray agreement 7.3 → 10.2 px, joints passing the gate
  47699 → 9291. The endzone's paint at 23–27 px is not a ruler the geometry
  trusts, and its residual is not in rotation and focal. Two rulers, one
  disagreed, nothing applied. `cameras_endzone_refined.npz` is the record.
- The endzone PITCH from the ankle ruler alone (2026-09-05): +0.2° puts the
  play 1 lower-ankle median at the turf, but planted feet (p10) are already at
  the turf at 0°, the median is running players, and on play 2 no pitch moves
  both p10 and median onto the turf -- the float there is dispersion in a
  sparse triangulation, not a camera bias. A median is not a stature.
- Seeding the joint solve with another play's mount and letting it run
  (2026-09-05, 5e2a833): on play 3 the paint asks for a 21–53° lens from the
  seeded mount while the boxes say 12°; holding the centre there did not
  rescue it. The fault was upstream of the solver (next entry).
- The paint reader's row labelling with two rows only (2026-09-05, diagnosed
  by drawing the correspondences, `diag/play3_corr_394.png`): the far
  sideline's yard ticks and the far hash row get labelled as the two sidelines
  (+24.38 / −24.38), a 2.3× cross-field stretch and a 41° lens, while grid
  consistency, an x-instrument, still scores 0.95 and the ladder is right
  (consecutive 5-yd lines). Four-row frames label correctly and give the 12°
  the boxes ask for; the pooled candidate was the compromise (25°, players
  2.78 m), and play 4's rulers disagreement (hashes 1.82 vs numerals 0.99) was
  the same stretch. Fixed in dd1b95c by bounding the implied lens to the seed
  play's (×1/1.6..1.6) inside `assignment_is_possible`; a labelling's fit to
  its own lines never ranks above a plausible lens again.

Memory `measured-dead-ends.md` has the numbers. Headlines: endzone from paint
(dead); per-detection team labels and looser gates in the linker (worse);
generic and football-trained re-ID embeddings (neutral / worse than ImageNet
at the linker's question); a whole-field coverage gate (rejected the right
camera); numerals as the only ruler (constant was wrong — the hashes caught it).

## Open items, in value order (2026-09-10, after v30)

0. Twin tracks: DONE 2026-09-10 (scripts/08o, tracking.twins) -- see the
   entry below. Play 1 folded the left tackle (18 & 19) and the
   quarterback (16 & 22); 148 ids -> 146. Do NOT lower the 20-frame
   minimum to catch more: at 6 frames the rule folds ids 21 & 28 (0.02 m
   at the feet over 8 frames) and 14 & 32 (0.05 m over 6), and the footage
   shows 21 & 28 to be two Baltimore players one behind the other -- with
   few ankle frames the filled tracks agree by construction. The margin
   that makes the rule safe is many frames, not a tighter distance.
0b. Left/right label flips: the detector swaps a limb group's labels for
   one to four frames (play 1's motion man at 267, arms and legs at once).
   pose.keypoint_filter.fix_lr_flips catches those frames but swaps 3 % of
   all (player, group, frame) decisions and 13-24 % for the players who run
   at the camera, because the shoulders are only 16 px apart at the median
   -- the test runs at the noise level. MEASURED 2026-09-10 (05p --fix-lr,
   ids 9 and 12): the runner's arm speed p90 in the body frame fell 8.91 ->
   7.66 m/s and his reprojection p90 rose 16.5 -> 21.6 px (the fit stops
   following the flipped frames, so that ruler is biased against it); the
   lineman did not move at all; on the footage strips the arms are no
   better. Still off. Gate it on the pair's separation, or decide the flip
   in 3-D where the body's orientation is known.

1. Two-view coverage: 9-10 of 22 players a frame are paired; the rest
   stand on one camera's box point (0.5 m from a triangulated pelvis at
   the median) with a one-view pose (heading within 21 deg of the two-view
   fit at the p50, 3 % facing the wrong way). Measured 2026-09-08 (v22's
   tracks): of 30 sideline tracks of 30+ frames without a partner, 17 have
   no endzone track within 4 m and sit OUTSIDE the endzone image on 100 %
   of their frames (staff on the near sideline, excluded from the render
   anyway), 5 face another kit at their nearest endzone track, 4 are
   eligible but their partner is taken by a longer overlapping fragment, 2
   overlap no endzone track for 6 frames. The pairing is at its ceiling
   for what both cameras see; the ceiling is the endzone camera's 4-6 deg
   lens (frames 125-634 only, and it never holds the wide receivers) and
   the fragments. The regressed pose is a weak cue (18 % top-1); pair
   truth exists only on the helmet set.
2. Fragments: 73 sideline tracks for ~29 people; every stitching cue
   measured is a dead end (position, colour, betas, kit+number vetoes);
   the visible cost is small (one avatar swap at a track end within 30
   frames and 80 px on play 1) but each fragment is a separate identity.
3. Officials inside the field are not tracked as separate bodies (the
   detector/linker folds them in); boundary staff and the sideline
   official are excluded by position and stripes.
4. Compute hygiene: DONE 2026-09-11 with the user's go-ahead -- the 30
   superseded render_hifi_* directories and render_abs were deleted (play_001
   16 GB -> 765 MB); only render_hifi (v31) is kept, and every delivered
   version's 720p clip stays in C:\Users\sumedh\diag\play_001_vNN_hifi_720.mp4.
   appearance_v1/v2 (26 MB) were left. A render script that moves render_hifi
   aside before rendering should delete the old one once the new clip is sent.
   The 05d render is opt-in (RENDER_ABS=1).
5. Plumbing: 08c's --kicking-play is manual (the pipeline does not pass
   it); a kickoff/punt/field-goal play needs it or its specialists render
   unnamed.
6. Generalise across the ~160 downloaded play pairs (paint-solve dead zone
   35-55 deg handled by --vertical-deg 45; expect new failure classes) --
   NOT before play 1 is signed off (the standing directive).

## Environment

RTX 4080 Laptop, 12 GB, driver 595.71 / CUDA 13.2. `C:\venvs\nflgsplat`
(Python 3.14.7; torch 2.11 cu128, ultralytics 8.4, numpy 2.4.4, opencv 5,
easyocr) and `C:\venvs\smplx312` (Python 3.12; torch cu128, smplx, numpy
1.26, pyarrow, imageio, av). Venvs live at short paths on purpose
(`LongPathsEnabled=0`). Do NOT `pip install -e .` (pyproject pins numpy<2 for
paddle/mmcv); run with `PYTHONPATH` set to the repo. ffmpeg 9 via winget.
SMPL-X models under `data/body_models/smplx/` (license-gated). Rosters:
`scripts/fetch_nflverse_rosters.py` → `data/rosters/2024/`.

## 2026-09-17 (overnight): the clip ends when the play is dead; a gait for the legs

**Decisions taken by the user (04:00):** end the clip when the play is dead; build and test the gait
model; keep the loop running overnight, hypotheses -> tests -> footage.

**Play end (08x, commit feb29cd/b26fb91).** The share of moving bodies does not mark the death of the
play (scratch probe_play_end: 5-25 % of bodies move faster than 3 m/s DURING the play -- linemen
engaged, backs covering -- and 50-80 % after it, everyone jogging in). The ball carrier stopping does:
he is taken as the id that travels furthest from its snap position over the live window (play 1: id
9, 19.2 m), and the play is dead when he stops for 10 frames, his track ends, or the sideline last
sees him -- the 05q strip of 9 at 480-492 shows him wrapped and going down at 482-486 and on the turf
by 488, while the bridge kept a standing body on him to 493, so the last sighting caps it. 05k reads
<play-dir>/play_end.json and stops at end + a 30-frame tail; pipeline_play.sh runs 08x as a marked
stage before the rulers. Play 1: end 493 by the first rule, re-run with the last-sighting cap pending.

**Gait (render/gait.py, commit feb29cd; opt-in via 05k --gait / 05q --gait).** Where the pelvis
moves faster than 4.8 m/s the hip and knee rows of both legs are synthesised from a phase that
advances 2 pi per stride of travel along the body's forward axis (a backpedal cycles backwards):
stance with the foot planted by construction (the hip's forward reach runs linearly back over the
stance travel), a cosine swing with the knee to 1.3 rad, blended over 6 frames at the edges; the fit
keeps the torso and arms. First cut used the STEP length as the cycle (2.2 m at a sprint) and ran the
runner at 2.8 cycles/s, twice a sprinter's cadence -- the cycle is two steps, 4.6 m at 9 m/s, and the
ground share falls with speed (0.6 walking to 0.22 sprinting). Fixed before any footage. A/B on the
live play (skating ruler, sideline reprojection, jitter) running; render v51 = v50 + gait + the clip
end follows if it reads well.

Gait A/B, corrected stride (04:40, scratch probe_gait_ab; post-hoc on the live play): 18 ids get the
gait (the runner 148 frames / 3.8 cycles = 3 steps a second; 3, 6, 40, 74 ... for 15-30 frames each).
Skating ruler: slower ankle / pelvis p50 0.94 -> 0.88, planted 1 % -> 11 %. Sideline lower joints
9.9/16.2 -> 17.8/25.4 px (the legs no longer chase the noisy ankle keypoints; expected). Joint jitter
p90 0.062 -> 0.153 (a leg that really swings reverses hard at each end; the ruler was calibrated on
legs that did not). 11 % planted is short of a runner's ~50 %: diagnosing on the runner whether the
plant fails where the body moves across its fitted facing (43 deg median off the velocity on this
play) -- then the legs should swing along the velocity. v51 (gait as is + the clip ending at 523)
is rendering for the footage verdict regardless: a leg cycle that looks like running beats a ruler.

Play end, capped at the carrier's last confident keypoints (05:00): the box tracker outlives the
detector's pose (boxes on 9 to 493, keypoints to 483), so play_end.json now says end 483, clip to 513.
v51 went out under the earlier 523; the next render uses 513.

Plant diagnosis on the runner (scratch probe_gait_plant, 137 moving frames): where the fitted facing
is within 20 deg of the velocity (75 frames) the gait plants 24 % (ratio p50 0.75; the stance dips to
0.08-0.27 every ~15 frames, a 2-cycles-a-second sprint with 0.1-s stances -- physically right; the
< 0.3 criterion misses the stance edges, ~40 % is the ceiling at a sprinter's duty); where the facing
sits 45-90 deg off the velocity (53 frames, 39 % of his run) it plants 0 % -- the legs swung along
the fitted facing, sideways to the motion. The fitted yaw is the unreliable part (43 deg off the
velocity at the median on this play). Fix: the legs' plane turns onto the direction of motion
(`gait.leg_yaw`: a yaw about the pelvis's up axis composed with the hip flexion; a backpedal keeps the
plane and cycles backwards). Re-measuring.

Re-measured with the legs in the plane of the motion (05:20, commit 9f374ab): all moving bodies --
slower ankle / pelvis p50 0.94 -> 0.74, planted 1 % -> 21 %; sideline lower joints 9.9/16.2 ->
15.3/21.6 px (the facing-plane version cost 17.8/25.4: legs that swing where the man goes also sit
nearer his keypoints); joint jitter p90 0.062 -> 0.208 (the ruler now counts real swing reversals;
compare gait renders among themselves on it). The runner: planted 13 -> 27 %, and the stretch where
his fitted facing sat 45-90 deg off his velocity goes 0 -> 26 %. A sprinter's duty caps the
criterion near 40 %. v51 = the facing-plane gait (clip to 523), v52 = this one (clip to 513),
chained; the strips of the runner (326, 400, 470), 40, 0, 2, 6 and 38 decide.

**v51 delivered (15:36; the machine was off from ~05:30 to ~15:00 and the chain resumed): diag/
play_001_v51_hifi_720.mp4 = v50 + the facing-plane gait + the clip ending at 523 (255 rendered
frames, 69 after the play left out); legs synthesised on 2236 body-frames of 65 ids.** Strips: the
runner at 326-340 -- a stride cycle, one leg forward while the other drives back, the knee lifting in
swing, feet flat; at 400-414 the same with a forward lean; at 470-484 running into the tackle, then
wrapped. Id 40 (a Baltimore back) at 378-392 -- alternating legs with a high knee. At strip scale
the legs read as running, not gliding; the video is the user's to judge. v52 (legs along the motion,
clip to 513) is rendering; 07l gains --gait so the report scores what 05k --gait draws.

**07l v57 = v56's timeline with the gait applied (the v52 render's legs).** Skating p50 0.94 -> 0.75,
planted 1 -> 21 % of 244 moving frames; joints jitter p50/p90/p99 0.0106/0.0329/0.069 ->
0.0112/0.0501/0.263 and speed p90 0.085 -> 0.110 (legs that swing and reverse; from here the joint
rulers compare gait renders with gait renders); steps 5, hops 4, root jitter, census 1.19 unchanged
(the gait moves no pelvis).

Gait over the footage (05q --gait, scratch review_gait): the runner at 340-354 -- the synthesised
legs stride while the green keypoints sit within 10-20 px of them; running toward the camera the
swing leg projects as a sideways kick at mid-swing. Id 40 at 380-394 -- the legs stride hard (a
sprinter's 39 deg reach at 0.24 m/frame) on a body that stands the known metre off the man (the
endzone-only placement of 385-394), so the legs look right and the placement wrong. The gait is the
pipeline default from here (pipeline_play.sh: GAIT=0 draws the fitted legs; both 05k and 07l take
it), reversible with one variable.

Gait on the jogging band, measured and DENIED (16:05, scratch probe_gait_jog; RUN_M 0.08 -> 0.05 so
bodies at 3-5 m/s get the gait too): on those 545 live body-frames the fitted legs already plant 9 %
(the detector's legs are usable at a jog) and sit 8.5/14.6 px on the sideline keypoints; the gait
takes planted to 29 % but the legs to 21.4/37.3 px -- a synthesised walk over legs the footage
shows is a loss. RUN_M stays 0.08: the gait replaces legs only where the keypoints cannot carry
them (a sprint). Ankle rows level the stance foot from commit 2b9058b (v53 chained behind v52).

**v52 delivered (16:15): diag/play_001_v52_hifi_720.mp4 = v50 + the gait with the legs along the
motion + the clip ending at 513 (250 rendered frames, 8.3 s; 74 rendered frames after the play left
out).** Strips: the runner at 326-340 strides (334-340 one leg driving back while the other knee
comes up), id 40 at 378-392 strides with a sprinter's trailing leg at 390-392. The clip itself, tiled
every fourth frame from the pile break to the tackle: bodies run, the scene holds, the clip ends
with the pile forming. v53 (+ the levelled stance foot) is rendering. Next check: whether the fast
jog band (4.8-6 m/s) is better served by the fitted legs, which would move RUN_M to 0.1.

Fast-jog band alone (0.08-0.1 m/frame, 96 live frames, 16:40): the fitted legs skate there as at a
sprint (planted 1 %, ratio 0.89) and sit 7.5/15.8 px; the gait plants 17 % at 10.9/21.3 px. Against
the slower band (0.05-0.08: fitted legs plant 9 %, the gait costs 13 px), 0.08 is where the
keypoints stop carrying a stance and the gait starts paying for itself. RUN_M stays 0.08; the
threshold is measured on both sides now.

Arm swing on running bodies (16:50, scratch probe_arm_swing; frames where the gait is on): the
fitted wrists travel 0.4-0.6 m fore-aft per leg cycle (runner 9: 0.41 / 0.57 m, upper-arm flexion
range 16 deg; 6: 0.41 / 0.50; 19: 0.33 / 0.57 at 52 deg) against a sprinter's 0.8-1.0 m and 70-90
deg -- the arms pump at about half amplitude. Not the moonwalk: the fitted arms are on their
keypoints, and replacing them with an animation would trade footage for a guess, the same trade the
jog band lost. Recorded; a lighter temporal weight on the arms of sprinting ids is the fit-side
hypothesis if the video wants more pump.

**v53 delivered (16:49): diag/play_001_v53_hifi_720.mp4 = v52 + the stance foot levelled (250
frames, clip to 513).** The runner at 326-340: the stride with the stance boot flat on the turf; id
6 (a Baltimore back at 340-354, under 4.8 m/s so the fit's legs): a coverage shuffle, bent knees,
natural. v53 is the current best; v50 is the last fitted-legs render for a side-by-side.

Side by side for the video verdict: diag/play_001_v50_vs_v53_run.mp4 (the run, rendered frames
150-250 = timeline 300-500, v50's fitted legs left, v53's gait right, 960x540 each). The standing
directive of 2026-09-05 (play 1 only, no compute on other plays) stands until the user lifts it;
the next play was considered at 17:00 and not started.


## 2026-09-17 (evening): the snap was never at 300 -- RETRACTION of the live window and the clip end

**The footage, not a ruler, found it (17:20).** Full-frame overlays of both cameras (05q --frames 305
320 340 360 380 400 --cam both, diag/census_v53/) show play 1's formation still SET at sideline
frames 360 and 380 -- linemen crouched, the QB under centre -- and the line firing at 400. The snap is
~393-396, not the 300 that identity.roles.snap_frame wrote into identity_resolved.pkl and that
LIVE_LO, 08x, 07l and every "live window 300-460/483" number in this file inherited. Bodies moving
faster than 0.03 m/frame (1.8 m/s) per frame from the v53 timeline: at most 6 of 22 before 388 (the
man in motion, id 9, from 284; a shifting defence; two gliding fragments), 9-11 at 392-404, 18 of 24
at 416. The snap detector's 0.5 m/s "static" cut sits below the placement's own jitter, so "fewer
than 40 % static for 12 frames" fired 95 frames early.

**What that retracts.** (1) Every "live" ruler since 2026-09-13 ran on a window that was two-thirds
pre-snap: the census 1.19 "live", hops "live 4-5", the KC-at-the-snap deficit (300-344 = KC 9 drawn,
ids 37/38 arrive at 340/345 -- that IS pre-snap, the formation standing 50 frames before the ball
moves; two linemen missing from a set line for 0.8 s). The fixes judged on that window were also
judged on the footage strips, so they stand; their aggregate numbers do not mean what they said.
(2) The play is a PASS: the QB is in the pocket at 480 (endzone 465), the ball is downfield at 520,
the tackle is at ~640 by the 15-yard line, the source clip ends at 647. 08x's "ball carrier =
furthest traveller from his frame-300 position" picked the motion man (19 m, mostly pre-snap) and
"last seen" ended the play at 483 when he ran out of the sideline frame: **v51, v52 and v53 stop at
513, two seconds after the real snap, in the middle of the play.** The 16:49 "v53 delivered (clip to
513)" is withdrawn as a clip; the gait/unwrap/twin content of those renders is unaffected.

**The crowd is the ruler now (scripts/08x_play_end.py, rewritten).** snap = the frame from which at
least 0.4 of the drawn bodies (12+ drawn) move faster than 0.03 m/frame for 6 frames, less 4 (the
line fires a few frames after the ball moves); dead = that share under 0.4 for 30 frames from snap+60,
else the clip's end (NFL Pro cuts the All-22 at the whistle: the share never falls below 0.57 to the
last frame here). Play 1: snap 393, dead 660 (clip end), clip from 213 (START_BEFORE 180 = 3 s of
formation and motion). --carrier ID keeps the old stop rule for a named id; --end/--snap override.
07l's --lo/--hi now default to play_end.json's snap/end; 05k honours a "start". tests/test_play_end.py
(5). identity.roles.snap_frame is left as is (08n's pre-snap window [snap-130, snap-30] lands on
the same set formation either way on play 1) -- but it is wrong and must not be trusted on another
play; the 08x detector is the one to reuse. Memory: snap-frame-check-on-footage.md.

**Ghosts from the endzone lead-in, confirmed on the footage (17:05-17:40).** The census surplus on
Baltimore (11.53 on 300-483) is not kit-vs-label (only 37 of 5689 confident live detections
disagree with their id's team, all on endzone-only ids 91/42 that are never drawn). It is (a) the
30-frame reach before/after an id's sideline span, drawn from the endzone alone: id 198 at 453-482
stands on EMPTY TURF beside the tackle 2.4 m from its man (diag/census_v53/player_198_f453.jpg), id
40 at 368-396 a phantom defender 1-2.7 m beside the real one; the smoother turns the 2.4-2.7 m jump
at the join into a glide, so the hop ruler (3-frame excess) never saw it; and (b) fragments on men
already drawn under another id in the pile (194/40 at 0.20-0.25 m, 168/15, 84/13; 37/166 41 frames,
19/166, 19/37 in the KC line) -- real men, doubled, below the twin rule's 0.2 m / 8 frames; no radius
separates them from engaged linemen (thread open, the 09-13 finding stands). Held real men: 37 and
38 drawn 30 frames before their sideline spans join within 0.27-0.55 m and stand on their men.

**Rule (render/endzone_only_rule.py `hold_m`, HOLD_M).** For each side of a two-view id's sideline
span, the jump between the endzone's point at the frame adjacent to the span and the sideline's
own first (last) point for the man; over ``hold_m`` the whole side is dropped. First cut compared
every beyond frame to the join point and read a man's own running over a long lead-in as drift
(74: median 1.29 m over 314 frames, mostly his 125-390 pre-snap stretch) -- replaced by the join jump
before shipping. A/B on 300-483 with the per-frame cut, hold off -> 0.8 m: KC 10.82 -> 10.79, BAL
11.53 -> 11.21, frames at exactly 11/11 50 -> 69, frames with BAL >= 12 75 -> 33, pile pairs 9 -> 9,
live hops 5 -> 6 (74 at 418, 0.225 m: its lead-in now ends where the sideline starts). Join-jump A/B
below.

**Join-jump A/B (18:05), hold off -> 0.8 m on 300-483:** KC 10.82 -> 10.73, BAL 11.53 -> 11.36, exactly
11/11 50 -> 56 frames, BAL >= 12 75 -> 62, live hops 5 -> 4 (74's 0.228 m at 419 gone), pile pairs
9 -> 9. Dropped whole-play: 198 (jump 1.55 m, 43 frames, the ghost on the turf), 40 (1.55, 27: the
phantom defender's lead-in), 74 (1.14, 270: its pre-snap endzone stretch -- in the live window only
the 16 frames 366-369/409-420 that flickered between two old-rule drops, so KC -0.09 and one pop
fewer), 164 (1.22, 80), 170 (4.34, 58), 186/212/206. Held: 37 (0.31), 38 (0.50), 17 (0.41), 27, 30,
171. The per-frame cut dropped more (BAL 11.21, exact 69) by also dropping men for their own running;
the join jump is the honest version and ships: **HOLD_M = 0.8** (render/endzone_only_rule.py).

**v54 delivered (19:20): diag/play_001_v54_hifi_720.mp4 = v53 + the join-jump hold + the real snap
and the whole play (timeline 213-660 at stride 2, 224 frames, 7.5 s at 29.97).** The run sheet
(scratchpad/review_v54/run_sheet.png, 380-640) shows the line firing at 400, the pocket at 440-520,
the routes at 540-600 and a white body down on the turf at 640. Strips: 198 ABSENT at 452-480 (the
ghost on the turf is gone); the tackler 184 renders prone with arms forward at 640-652 and then
stands up at 654-658 while the footage still has him down; 55 dives plausibly at 604-638.

**The join-jump hold misses a GLIDING ghost (19:50).** Id 40's lead-in 368-397 is still drawn: its
lead-in joins the sideline within 0.67 m (kept), the 1.55 m jump the report showed was its TAIL
(527-553, dropped 27). The footage strip has 40 standing 1-1.5 m beside the real defender at
368-383 and on him at 398: the endzone's placement slides 2.7 m onto the man over the 30 frames,
and the join test only sees the last step. The per-frame test caught it and misread running men;
the join test passes it. What separates them is not metres at all: a real man is always under
SOME sideline detection box (even a merged one, as 37/38 were), empty turf never is -- the test
belongs in the sideline image (project the endzone point, ask whether a sideline box covers it).
Measured next.

**Bodies on the ground -- measured and NOT adopted (20:10).** Hypothesis: the tackled men are drawn
standing because the fit's tilt prior (weight 10 past 20 deg) forbids lying, so gate it by the box
(sideline h/w under 0.7: linemen's p5 is 0.87, the tackled men 0.46-0.58; 0.9 also caught crouched
linemen 4 and 17 pre-snap on 110 frames each). Built: Mono2DConfig.lying (the lean must be at least
60 deg), 05p --lying-aspect, timeline.lying_frames + build_timeline(lying=) skipping the tilt clamp,
tests/test_lying.py. Probe on ids 184/28/55 (55 has no fit at all: no keypoints pass the filter).
Same-recipe control vs lying prior, world tilt through the timeline, id 184 at 642-660: control
20-36 deg, lying 142/140/115/80/52/46/56 -- past horizontal, the pelvis inverted; the fit's own
upper-body reprojection 16.7 -> 21.2 px, lower unchanged; the footage strips (diag/lying_v54/) look
the same for both: a folded body along the man at 642-651, standing up at 654-657 while he is still
down. The shipped fit ALREADY renders the tackler prone (v54 strip 184 at 640-652) from a folded spine
at 24 deg of pelvis lean: the 24 deg I first read as "standing" was the pelvis, not the body. What
remains wrong is 654-660 (he rises early) in both. The prior ships opt-in (05p default 0); the
timeline's no-clamp on lying frames stays (it never fires on the shipped fits; a gate, not a change).
Also: a first per-frame tilt reading taken on the raw records (24 vs 102 deg) was in the cache's
frame, not the world's -- measure through the timeline's states.

**Box containment is not the ghost test either (20:00).** Endzone-placed lead-in frames projected
into the sideline image, "inside any sideline box dilated a quarter body sideways": 198 0.72 (it
stood beside KC 65 on empty turf: the neighbour's box covers it), 40 0.18, real men 37/38/17 1.00,
74 0.96, 164 0.94, 170 0.47. Nearest box bottom in body heights: ghosts 0.57-0.60, real 0.26-0.48
-- the same 1 m ambiguity as the metres. What does separate 40 is TIME: before the snap a set man
does not move, so a lead-in that stands 2.7 m from where the sideline finds him and slides in is
the endzone's error. Rule: before play_end.json's snap every beyond frame is measured against the
join point (per frame); from the snap on, the join jump. Test added; A/B below.

**Snap-aware hold A/B (20:45), 300-483, hold 0.8 m.** Join-jump only (v54): KC 10.73 / BAL 11.36,
exactly 11/11 on 56 frames, BAL >= 12 on 62, id 40 drawn on 30 of 360-397. Per frame before the
snap + join jump after: KC 10.73 / BAL 11.21 (pre-snap 10.32 / 11.00, live 11.14 / 11.42), exactly
11/11 on 69, BAL >= 12 on 33, id 40 on 1 of 360-397, live hops 4, pile pairs 9. KC pre-snap loses
half a man (74 and 164, real linemen whose endzone depth sat 1.1-1.3 m off their sideline join).
Tried "hold": keep those pre-snap frames AT the sideline's join point instead of dropping them (a
set man has not moved). Rejected: KC pre-snap 10.32 -> 10.39 only, BAL 11.00 -> 11.19, exactly
11/11 69 -> 54, and a new pile pair 1-40 (22 frames): 40's held lead-in lands on the man already
drawn as id 1 -- the lead-in was a second copy, and dropping it was right. PRESNAP = "drop" ships
(endzone_only_rule.PRESNAP; "hold" kept as the measured alternative). v55 launched 20:50 with it.

**The double body on KC 76 (37 + 166, 41 frames within 0.6 m) is a DEPTH error, not two boxes
(21:15).** Every pile id draws within 3-18 px of its OWN sideline hip keypoints (166 13 px, 37 3 px,
19 18, 204 9, 17 11, 1 7): both bodies sit on their detections in the sideline image, and those
detections are 88 px apart -- adjacent linemen stacked along the sideline's depth axis (field y).
The zoomed two-camera crop at 380 (diag/census_v53/line_zoom_380_400.jpg) shows it: in the
sideline 11, 166, 37 are three men in a row; in the endzone 37 is on 76 with keypoints and 166's
skeleton stands a metre beside 76 on turf with nobody under it. 166 is sideline-only there (its
endzone pairing is 472-562), so its y is the sideline's own foot-point depth, off by ~0.5 m in the
line cluster, which puts it inside 37's 0.6 m. Sideline box IoU (0.36) and keypoint distance (88 px)
cannot call it a duplicate because it is not one: it is a man drawn half a metre into his
neighbour. The fix is depth for one-view bodies in the line (an unpaired endzone id for the same
man is among the 68 endzone-only ids left out), i.e. the pairing thread of 09-08/09-10, not a
twin rule. Left open.

**The real baseline (21:30): 07l on the live window 393-660 (play_001_v55_plausibility.json,
the v54/v55 timeline).** steps > 0.25 m/frame live 114 (handovers 7), > 0.6 none; hops live 25 (the
old window said 4-5); skating: planted 13 % of 1207 moving body-frames (a runner plants ~50 %);
census live 2.55 (KC 11.04, BAL 11.49); joints (57 ids) jitter p50 0.026, p90 0.156, p99 0.58.
Every "floor" claimed on 300-460/483 was two-thirds pre-snap; the play itself has 25 hops and a
p99 joint jitter of 0.58 m/frame^2. This is the number to beat from here. 07l's window now comes
from play_end.json (a str/Path slip fixed 21:25).

**Hole hold (22:05): endzone-filled holes inside a sideline span follow the sideline's own line.**
The span rule holds a span's edges; inside a span the hole rule (HOLE_REACH 8) draws endzone-placed
frames, and the endzone's ground point is poor along the field: id 37 at 569-576 stepped 0.42-0.44 m
a frame for five frames -- the footage strip (diag/live_v55/player_37_f558.jpg) has him drawn on the
lineman two metres LEFT of the man the sideline resumes on at 573, who stood there throughout.
Rule (endzone_only_rule.hold_holes, HOLE_HOLD_M 0.8): a hole frame farther than 0.8 m from the
sideline's line between the hole's ends (holes up to 17 frames) or from the nearer end's point
carried at the sideline's own velocity over 4 frames (longer holes, the frames within reach 8 of an
end) takes that line. Live window 393-660: 110 frames held (median 1.31 m off, max 3.69); hops 25
-> 23, steps > 0.25 m/frame 114 -> 97, handovers 7 -> 4, KC 11.04 -> 11.10, BAL 11.49 -> 11.38; 37's
run of 0.4 m steps -> 0.03 (one 0.47/0.48 pair remains where the two ends' extrapolations meet
mid-hole). Shipped; v56 carries it.

**v55 delivered (22:35): diag/play_001_v55_hifi_720.mp4 = v54 + the snap-aware hold (id 40's pre-snap
phantom gone; 74's off-line pre-snap stretch gone).** Sheets over the live play at 6-frame steps
(scratchpad/review_v55/sheet_a.png 393-525, sheet_b.png 531-660): the line fires, the pocket holds
to ~520, routes to 600, the tackle at 639-657 with the tackler prone. Nothing standing alone on the
turf. A small black-clad figure deep at the top of 615-633 (an official or a teamless body) to check.

**Gait threshold re-tested on the real window, DENIED again (22:30, scratch probe_gait_runm).** RUN_M
0.08 (shipped): gait on 2135 body-frames, planted 14.2 %, ratio p50 0.79; 0.06: 3205, 18.5 %, 0.76;
0.05: 3860, 18.2 %, 0.78. Four points of planting for a thousand body-frames of synthetic legs on
bodies whose keypoints carry a jog (the 16:05 reprojection cost, 13 px, is unchanged by the window).
RUN_M stays 0.08.

**Next: the sideline's own double tracks.** Same-team id pairs whose SIDELINE boxes coincide (IoU
>= 0.6) on 8+ live frames -- one man, two tracker ids; the memory's "never by boxes" was about
engaged linemen at IoU ~0.4, these are 0.85-0.97 at the p90: 40-198 (27 frames 483-522: the
ghost's sideline track sits on 40's man), 61-64 (BAL, 27, p90 0.95), 197-186 (KC, 14, 0.96), 194-40
(13, 0.93), 12-170 (9, 0.87), 46-56 (9, 0.97), 194-198 (8, 0.85), 211-205 (14, 0.69). Footage strips
of 61/64 and 197/186 next, then a box-twin rule A/B on the live census.

**Box twins, measured and left opt-in (23:00).** timeline.box_twin_frames (same-team drawn ids whose
sideline boxes overlap IoU >= 0.6 on 8+ consecutive frames; the id with fewer sideline boxes loses
them), load_play_timeline(box_twin_iou=). Of the strong pairs, 61/64 and 46/56 are bench people at
the boundary (never drawn: diag/boxtwins/player_61_f578.jpg), so the rule reached 74 body-frames
whole play: 27 at 129-150, 195 at 181-205, 170 at 598-605, 198 at 489-509 (the ghost's sideline
track on 40's man). Live window 393-660: KC 11.10 -> 11.07, BAL 11.38 -> 11.31, exactly 11/11 34 ->
35, BAL >= 12 115 -> 109, pile pairs 18 -> 16 (289 -> 261 frames), hops 23 -> 25, steps > 0.25
97 -> 94. Real duplicates, a handful of frames, and two hops bought by the gaps it leaves.
BOX_TWIN_IOU = None (off); test in tests/test_timeline.py.

**An official drawn as a Baltimore player (23:20).** The small black figure at the top of v55's
615-633 is id 70: on the boundary line at the LOS for its whole 28-frame life (|y| 23.5-24.4,
x -25..-24, 611-638, kit margin -0.31, no jersey, no role), drawn 20 yards behind the play in its
fitted black-and-white appearance. The dweller rule's line was 23.5 m and 70 straddles it.
SIDELINE_M -> 23.0 (offfield_rule): among every id the v53 timeline draws, 70 is the only one with
half its frames beyond 23.0 (every roster-named player stays under 6 m median, none past 23.5).

**Despike (23:30): a median of the +-2 neighbours BEFORE the position Gaussian.** The hops the
rulers count on the live window are single-frame spikes the Gaussian spreads into a hop; a
median sees a spike as the odd one out and a real cut (every later frame moves the same way) as the
trend. timeline.despike_xy (excess DESPIKE_M, symmetric window only, holes left alone), applied in
build_timeline before smooth_xy; knob despike_m on load_play_timeline. Live window 393-660: hops 23
-> 2 (0.15) / 2 (0.25), steps > 0.25 m/frame 97 -> 63 / 65, handovers 4 -> 0, root jitter p90
0.050 -> 0.032 / 0.035; the two left: 79 at 544 (0.36 m) and 194 at 503. Second ruler (the bodies'
hips against their own sideline keypoints, off vs 0.15) below.
Second ruler (23:40): 3024 drawn body-frames with hip keypoints on 393-660, hips vs own keypoints:
off p50 7.3 / p90 16.0 / p99 40.0 px, 70 over 30 px; despike 0.15: 7.3 / 16.2 / 40.3, 68 over 30.
The spikes go and the bodies stay on their keypoints. **DESPIKE_M = 0.15 ships**; v57 carries it
with the dweller line.

**v56 delivered (23:50): diag/play_001_v56_hifi_720.mp4 = v55 + the hole hold.** Sheets as v55. But
id 37's strip (scratchpad/review_v56/strips/render_37_f558.png) flickers: absent 558, drawn 562-568,
absent 570, drawn from 574. Per frame in the current timeline: absent 545-559, drawn 560-568 on
endzone-only views with steps 0.30-0.46 m/frame, absent 569-573, sideline from 574. The hole is 29
frames; the hold moved the frames within reach 8 of the end (566-573) onto the sideline's line and
left 560-565 at the endzone's spot two metres away, so the glide moved rather than went, and the
held frames near the join were then deduped against the man's other id. Fix: in a long hole the
frames beyond reach of BOTH ends are left out (they were the endzone's blind depth with nothing to
hold to). v57 (launched 23:45, despike + dweller line) predates this; v58 carries it.
Measured (00:05, with the despike in): 110 held + 28 left out; live window hops 2, steps > 0.25
m/frame 63 -> 55, handovers 0, KC 11.10 -> 10.98 (the 28 blind frames), BAL 11.31; id 37 absent
until the sideline resumes at 574 and steady after (0.03 m/frame): no glide, no flicker. Ships;
v58 carries it after v57.

**Where the live joint jitter is (00:20).** 07l's "jitter p99 0.58" on 393-660 was measured WITH
the gait: the synthetic legs swing, and that swing is most of the p99 (the 09-16 note already had
0.263 for gait swings). Without the gait, the worst ids by 07l read: 212 p90 0.008, 206 0.000 (nothing
there), 185 0.10, 184 0.43 (the diving tackler at 655-658, interpolated between sparse records),
79 and 5 0.2 at 536-541 (a throw-side arm, real), and 66 0.84 at 581-582 -- a DEFAULT-POSED body
(no fit at all; 66 at 576-607 and 194 at 467-509 are the two on the play) whose heading comes from
its motion and started at the fallback 0 deg until the motion gate opened, then turned 104 deg in
one frame. Across the worst ids the jitter does not sit at view switches (p90 0.032 there vs 0.080
elsewhere) nor differ between record and interpolated frames (0.072 vs 0.081).
Fix: timeline.yaw_from_motion backfills the frames before the first known heading with it and
smooths the heading circularly over 5 frames (YAW_SMOOTH). Test added. Numbers below.
Measured (00:30): id 66 jitter max 0.84 -> 0.08, p90 0.21 -> 0.07; 194 unchanged (0.01). Ships;
v58 (queued behind v57) carries the long-hole fix and this.

**Box twins re-measured with the despike in, and SHIPPED (00:45).** Same rule, same pairs, live
window 393-660: KC 10.98 -> 10.95, BAL 11.31 -> 11.21, exactly 11/11 37 -> 41, BAL >= 12 109 -> 100,
pile pairs 18 -> 16 (285 -> 251 frames), hops 2 -> 2, steps > 0.25 55 -> 53; 82 body-frames left out
whole play (194 490-497 and 198 489-509 on 40's man, 170 598-605, 27 and 195 pre-clip). The two
hops it cost at 23:00 were spikes the despike now removes. BOX_TWIN_IOU = 0.6.

**v57 delivered (01:05): diag/play_001_v57_hifi_720.mp4 = v56 + the despike + the dweller line.** Id 70
(the official) absent on 611-638 (strip); 76 stays with the pile at 628-646 instead of sliding off
it; sheets as v56 otherwise. v58 (long-hole middles out, heading backfill, box twins) rendering.

**Bodies on the ground flail between sparse records: a wider pose Gaussian on lying frames (01:30).**
The one real flail left on the play was the diving tackler 184 at 655-658 (joint acceleration p90
0.43 m/frame^2, records on 7 of 19 frames). timeline.build_timeline: frames the box calls lying
(lying_frames, aspect 0.7) take the pose Gaussian at LYING_SIGMA_MULT x sigma, blended in over
LYING_BLEND 3 frames either side of a lying run. Without the blend, swapping the values on the
lying frames alone made a seam: id 1's isolated lying frames went p90 0.08 -> 0.55. With it, on
393-660 (55 lying body-frames, ids 1/28/34/35/36/55/82/184/206): lying p90 0.189 -> 0.079 (x3) /
0.031 (x5); 184 0.42 -> 0.09 / 0.03; 28 0.19 -> 0.02 / 0.04; 1 0.08 -> 0.15 / 0.19; all-frames p99
0.105 -> 0.098. LYING_SIGMA_MULT = 3.0 ships (the seam cost on 1 is the reason not 5). Footage strip
of 184 with it: diag/lying_smooth/player_184_f646.jpg.
The tackler's footage strip with the smoothing (01:40, diag/lying_smooth/player_184_f646.jpg): the
skeleton lies along the man from 646 to 660 -- he no longer rises at 654-658. v59 (queued behind
v58) carries it.

**Pre-snap per-frame hold only when the join is at the snap (01:55).** A join deep in the play is
where a man who has since run stands, and says nothing about where he stood set: 74 (join at
snap+28, 1.14 m off) and 164 were real linemen dropped for the whole pre-snap. PRESNAP_JOIN_MAX 10:
the per-frame test applies when the join is within 10 frames of the snap (40's phantom joins at
snap+5 and stays caught: 1 frame on 360-397), otherwise only the same-body test stands. Measured
on 300-483: KC pre-snap 10.32 -> 10.37, exactly 11/11 69 -> 68, BAL >= 12 33 -> 35, pile pairs 9,
hops 2 (0 live). A wash in numbers -- 74's pre-snap frames were mostly the OLD rule's (a sideline
body within 1.2 m in the line) -- kept because it is the right question to ask. Tests updated.
