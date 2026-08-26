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
