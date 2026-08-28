# First end-to-end result — play_001

Broadcast video to free-viewpoint frame, with real players, on 2026-08-26.
Everything below ran on the local RTX 4080. No PACE.

## The chain

```
sideline.mp4 ──► detect + BoT-SORT track ──► tracks.parquet   (42 tracks >=100 frames)
             ──► mosaic calibration       ──► cameras.npz     (816 verified frames)
crops        ──► SMPLest-X-H32 (0.69B)    ──► SMPL-X joints, camera frame
joints + K,R,t ► place_on_field           ──► joints in FIELD metres
procedural_field + joints ► preview_cpu   ──► novel viewpoints
```

## Numbers

| | |
|---|---|
| skeletons placed | 147 across 8 frames, **0 rejected** |
| ground positions | x −28.0…−9.3 m, y −14.0…14.0 m — all on the field |
| stature | p50 1.42 m, p90 1.58 m |
| field | 264,600 gaussians at 0.15 m/px, drawn from spec in 0.3 s |
| render | 1280×720, ~0.7 s/frame on CPU |

## Why the statures are not 1.8 m

They are posture-dependent vertical extents, not standing heights, and this play
is mostly linemen in a three-point stance. The p90 is the number to read: the
most upright players reach 1.58 m ankle-to-crown, and a 1.9 m athlete measures
about 1.67 m that way.

The remaining gap is accounted for rather than assumed. An independent probe —
YOLOv8-pose lifted through the calibrated camera with a vertical-body
constraint, sharing none of SMPL-X's body model — gave p90 **1.58–1.59 m** on
the same footage. Two routes with different failure modes agreeing is the
evidence that the calibration and the ground assumption are both sound.

Posture also moves it in the direction physics predicts: pre-snap median
1.48 m, post-snap 1.21 m, because players lean forward when running and pile up
after contact.

## What this does NOT show

- **No mesh.** These are joints rendered as gaussian blobs. Avatars
  (`avatars/build_play.py`) are a separate stage and have not run.
- **Sideline camera only.** The endzone feed is calibrated but not fused;
  cross-camera correspondence remains unsolved (see below).
- **Field is layer 1 only** — geometry and markings from the rule book. Turf
  colour, mowing pattern, endzone artwork and lighting are venue-specific and
  still to be extracted from the ortho pass.
- **`strict=False` on the checkpoint.** SMPLest-X loads with unmatched weights
  and warns. Outputs look sane, but "looks sane" is also what a partially
  loaded model produces. Not yet chased down.

## Cross-camera correspondence: still the wrong primitive

Measured again on this calibration, not assumed. Hungarian assignment of
grounded foot points between the two cameras across all 86 doubly-verified
frames gives a median 4.5 m residual, 3.5% within 0.5 m. The sideline's rays
graze at 80–100 m, so its weak axis is exactly the one the endzone measures
best. This reproduces the earlier finding (~85% mispaired) and confirms
per-frame geometric matching cannot carry identity.

---

## Cross-camera correspondence: measured three ways, all negative

Recorded so it is not attempted a fourth time without new information.

| primitive | result |
|---|---|
| per-frame foot points, Hungarian | median 4.5 m, 3.5% within 0.5 m |
| **trajectory** distance over shared frames | median 3.31 m, 2 of 22 pairs within 1 m |
| axis decomposition | bias X −0.23 m, Y +0.63 m; spread 2.27 / 3.50 m |

The third row is the one that settles it. The **biases are small**, so the two
calibrations do agree about where the field is — this is not a calibration bug.
What defeats matching is that median nearest-neighbour disagreement is 1.06 m
along the field and 1.29 m across it, while players in a formation stand roughly
1–2 m apart. The positional precision is comparable to the player spacing, so
the nearest track in the other camera is very often the wrong man, and no
smarter assignment recovers information that was never there.

Trajectory averaging barely helped (4.5 m → 3.3 m), which is itself diagnostic:
averaging suppresses noise, so the residual is dominated by systematic
per-player error rather than jitter.

**The dependency therefore runs the other way.** Correspondence cannot produce
identity, but identity produces correspondence for free: if the sideline feed
names a track #85 and the endzone feed independently names one #85, they are the
same player, with no geometric matching involved. Only 86 of 1302 frames have
both cameras verified anyway, so geometry was never going to carry this.

---

## Identity across two cameras, and what checking it revealed

Running the identity pipeline independently on the endzone feed and merging on
jersey number takes play_001 from 10 identified players to **17 of 22**.

| feed | identified | best vote counts |
|---|---|---|
| sideline | 10 | 51, 44, 15, 14 |
| endzone | 14 | 73, 60, 53, 51, 50, 44, 43, 39 |
| union | **17** | |

The endzone reads numbers far better because it is zoomed much tighter
(f ~19000 px against ~2000), so a jersey that is a dozen pixels tall on the
sideline is legible there. The two feeds also miss *different* players, which is
why the union beats either.

### The overlap is not the free check it looks like

Seven players were named by both cameras. That reads like corroboration -- the
feeds share no pixels, no tracker state and no calibration -- and this document
initially recorded it as such. Measuring it says otherwise:

| jersey | separation | verdict |
|---|---|---|
| #20 Julian Love | 2.17 m over 86 frames | ok |
| #1 Kyler Murray | **11.04 m** over 86 frames | contradicted |
| #8 Coby Bryant | **16.13 m** over 81 frames | contradicted |
| #18, #63, #85, #91 | 0-9 shared frames | untestable |

Sixteen metres is not a precision problem; it is a different human being. The
error in the reasoning was this: each camera solves a **forced** one-to-one
assignment against the same 22 jerseys, so every number gets emitted at most
once whether or not the evidence supports it. Two feeds can therefore agree on
a label with neither being right, and agreement is a hint rather than a proof.

Vote counts say which half fails. The sideline called a track Kyler Murray on
**zero** jersey votes -- via the sole-candidate alignment rule, which bypasses
the vote thresholds when only one player on the field plays that position --
while the endzone read #1 seventy-three times on a different track 11 m away.
For #8 the sideline track's nearest endzone track of any kind is 6.06 m away,
against a measured 1.06/1.29 m nearest-neighbour scatter, so that track simply
grounds badly.

### Geometry's actual job

Cross-camera geometry cannot CREATE correspondence -- measured three ways above,
all negative -- but it can REFUTE one, and that asymmetry is what makes it
useful. A claim that two tracks are the same player is falsifiable by a single
well-grounded frame, even though no amount of position data can generate the
claim in the first place.

So the honest state of play_001 is not "17 confirmed":

| | count |
|---|---|
| named | 17 of 22 |
| geometrically verified | **1** (#20) |
| contradicted, then repaired to the better-evidenced camera | 2 (#1, #8) |
| unverifiable -- one camera only, or no shared verified frames | 14 |

Only 86 of 1302 frames are verified in both cameras, which is why so little is
testable. Raising that count is now the constraint on verification, not on
identity.

### Still missing

ARI #4 Dortch, #14 Wilson, #70 Johnson Jr.; SEA #13 Jones, #27 Woolen. Three
were read and lost on the margin test rather than for want of evidence -- #14
took 53 votes against 36 for #4, #70 took 13 against 17 -- so the next lever is
resolving those contests, not more OCR.

---

## Endzone frame coverage: 104 -> 473 verified

The cross-camera identity check above could only test 4 of 7 overlapping
players, because only 86 of 1302 frames were verified in BOTH cameras. Fixing
that turned out to require fixing a separate, unrelated breakage first.

### The endzone mosaic had stopped calibrating at all

Re-running the recorded command failed at the reference-camera solve. It was not
a code regression: no calibration commit landed after the good `cameras.npz` was
written (2026-08-19 02:24 PDT; last prior commit 02:09), and the ORIGINAL
command reproduces the failure exactly. What changed is the toolchain --
**OpenCV 4.x -> 5.0.0, Python -> 3.14, numpy -> 2.4**.

The evidence that localises it:

| check | result |
|---|---|
| line detection + labelling | **identical** to the good run: same 7 lines, same two-line hole |
| today's hash points through the **stored, validated** camera | median **25.3 px** (max 53) |
| inlier support, both orientations | 64-69% against a 70% gate |
| refine + verify on the stored camera | **104/104 frames, median 0.74 px** |

At f=19187 -- a 6 degree telephoto -- 25 px is about 15 cm on the field. So the
accumulated paint moved slightly and the reference solve failed its own gates,
while every stage after it still worked perfectly.

Two candidate causes were tested and **rejected**, rather than assumed: the
regenerated `tracks.parquet` (imgsz 640 -> 1920 changes player masks) is not
responsible -- old and new boxes fail identically; and OpenCV's inlier mask is
not an artefact -- it agrees exactly with explicit reprojection counting.
Relaxing the 70% gate would also not have helped, and would have been the wrong
fix: the `rms > 3.0 px` check below it is strictly stronger, since it runs over
ALL points rather than RANSAC's inliers, and rejects the same solve.

### The fix, and why it is not a bypass

The reference solve is a **once-per-game** step -- the tripod holds one centre
all half -- and its result was already solved, checked against the yard lines
and shipped in `cameras.npz`. `--reuse-reference` takes it instead of
re-deriving it. Everything downstream is self-checking: a frame whose camera
cannot be verified against paint it can see keeps `conf = 0`, so a stale
reference costs coverage, never correctness.

### What the coverage was actually limited by

Nothing was failing verification. At `--refine-stride 5`, four frames in five
were **never candidates** -- the refine pass zeroes every `conf` and re-sets
only frames on its grid. The tell was in the data all along: 104 verified frames
separated by 103 gaps, so no two adjacent frames were ever both verified.

| | before | after |
|---|---|---|
| endzone verified | 104 | **473** |
| verified in BOTH cameras | 86 | **364** |
| median verification offset | 0.96 px | 0.94 px |
| points landing on the field | 100% | 100% |
| nearest-sideline-player agreement | 1.44 m | 1.77 m |

Sideline is untouched at 816. Only 69 of the old 104 frames re-verify: the
bundle now spans 1020 nodes rather than 104, so nodes shift and 35 fail their
own check. Accuracy is preserved but not improved, which is expected -- extra
frames on a tripod add coverage, not parallax.

### What it bought, honestly

The two contradicted identities are now measured over **364 frames instead of
86**, which makes them far harder to dismiss: #1 Murray at 10.44 m and #8 Bryant
at 16.50 m. It did **not** make more pairs testable. Those are limited by
whether the two feeds' TRACKS span the same frames, not by calibration.

It also revealed a real limit on the endzone formation path. Pre-snap verified
frames went 27 -> 131, yet clean formations went 2 -> 0, and the cause is not
calibration: in BOTH calibrations exactly **5** pre-snap frames have >= 20
players detected at all, with a median of 18. The endzone is a tight telephoto
and players occlude each other along its axis, so the detector rarely sees all
22 while `assign_offense_roles` needs exactly 11 and 11.

The one identity lost with it (#33 Benson) was assigned on **zero jersey votes**
from alignment alone, across 2 frames -- the same sole-candidate rule that
produced the wrong Kyler Murray. Losing it is a correctness gain, not a
regression.

---

## Two views of one player: the pose is fine, the placement is not

With both feeds posed on the same 364 doubly-verified frames, and identity
supplying the correspondence, four players are seen independently by both
cameras. Nothing else in this pipeline has ever checked a POSE against anything
outside the network that produced it; two cameras 131 m apart, sharing no
pixels, no tracker and no calibration, are that check.

| | whole-body offset | pose shape |
|---|---|---|
| #20 Julian Love | 2.92 m | **0.08 m** |
| #58 Derick Hall | 2.88 m | **0.09 m** |
| #91 Byron Murphy II | 3.15 m | **0.11 m** |
| #70 Paris Johnson Jr. | 2.71 m | **0.13 m** |
| median | 2.90 m | **0.10 m** |

Separating the two was the point of reporting them separately. **The two cameras
agree on the pose to 10 cm and disagree about where the player is standing by
2.9 m.** The articulation is good; the placement is the weak link. That also
explains the cross-camera correspondence failures recorded above — the 1.4-1.8 m
nearest-player scatter was this same error seen from another angle.

### The error is depth, not calibration

Decomposing the offset along each camera's viewing ray:

| | along sideline ray | along endzone ray | across BOTH | total |
|---|---|---|---|---|
| #20 | 0.73 m | 1.95 m | **0.78 m** | 2.92 m |
| #58 | 1.27 m | 2.04 m | **0.74 m** | 2.88 m |
| #70 | 1.56 m | 0.51 m | **0.75 m** | 2.71 m |
| #91 | 1.33 m | 2.14 m | **0.64 m** | 3.15 m |

Almost all of it lies ALONG the rays -- the depth each monocular estimate had to
guess -- and only about **0.7 m** sits in the direction neither camera measures
well. That is the irreducible part; the rest is exactly what a two-view fusion
removes, because each camera's weak axis is the other's strong one.

So fused placement should land near 0.7 m rather than 2.9 m, and 0.7 m is BELOW
the 1-2 m spacing between players -- the threshold that defeated geometric
correspondence three times over. That does not bootstrap correspondence for
unidentified players, since fusing requires knowing the pairing first, but it
does mean every identified player can be placed several times more accurately
than either camera manages alone.

### Placement is not broken — the ENDZONE's placement is

Chasing the 2.9 m, measured per camera rather than between them. A player moves
smoothly, so departure from a locally smooth path is placement noise, and it can
be measured on one camera alone:

| | ground jitter | box-bottom jitter |
|---|---|---|
| sideline | **0.01 m** | 0.3 px |
| endzone | **1.08 m** | 0.6 px |

The sideline places players to a centimetre. The endzone turns 0.6 px of box
movement into 1.08 m on the ground — about 1.8 m per pixel, against the 0.03 m
per pixel a static camera at that geometry gives. The boxes are not moving; the
endzone's per-frame camera is. It zooms across f = 1,532 to 23,200, each frame
solves its own focal and rotation, and verification checks yard lines ACROSS the
view, which at an 11 degree grazing angle barely constrains depth — the one
direction placement needs.

**Box-bottom error was ruled out first.** At this geometry 20 px of error moves
the ground point 0.53 m (sideline) or 0.67 m (endzone); explaining 2.9 m would
need ~100 px, which no detector box is off by.

### Smoothing the endzone camera: helps precision, not accuracy

The camera is a tripod panning smoothly, so frame-to-frame pose jumps are
estimation noise. Zero-phase smoothing (never causal — that filter moved this
very camera by 97 px of pan lag) collapses the jitter, but the cross-camera
agreement barely moves:

| min_cutoff | ground jitter | cross-camera offset | reference frame moved |
|---|---|---|---|
| as solved | 1.08 m | 2.93 m | — |
| 6.0 | 0.44 m | 2.49 m | 13 px |
| 3.0 | 0.25 m | 2.46 m | 40 px |
| 1.5 | 0.11 m | 2.44 m | 74 px |
| 0.8 | 0.05 m | 2.41 m | 91 px |

An 18% improvement in the thing that matters, bought by displacing the reference
frame's own camera — which is exact by construction — by up to 91 px. That is
the same magnitude as the pan-lag bug, arrived at from the other direction:
zero-phase smoothing has no lag, but it still pulls an exact sample toward noisy
neighbours. **Not shipped.**

So the 2.9 m is not jitter and not smoothable: it is slowly-varying endzone
DEPTH error, which matches the earlier finding that the offset lies along the
viewing rays.

**The practical conclusion is to stop treating the two cameras as equals.** The
sideline is the placement instrument; the endzone is the identity instrument,
where its tight zoom reads jerseys five to ten times better. Fusion should weight
by measured precision rather than the assumed isotropic sigmas it ships with.
