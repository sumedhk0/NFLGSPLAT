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
