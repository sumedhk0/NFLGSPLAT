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
off (y = 0 pairs more, triangulates more). **v13 is the current deliverable
and the pipeline's reproducible baseline**: `diag/play_001_v13_hifi_720.mp4`,
strip `diag/play_001_v10_v11_v13_strip.jpg`.

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
