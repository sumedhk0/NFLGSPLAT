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

## Open items, in value order (2026-09-08, after v21)

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
4. Compute hygiene: play_001 holds 12 GB, 13 superseded render/appearance
   dirs (render_hifi_v10..v20, render_abs, appearance_v1/v2) -- the user's
   call to delete. The 05d render is opt-in now (RENDER_ABS=1).
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
