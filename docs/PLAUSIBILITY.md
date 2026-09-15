# Plausibility: the loss, the rulers, and where a local generative repair fits

Written 2026-09-15 after the user asked for "some sort of loss parameter based on what the desired
outcome is -- no joints in weird positions, no jitter when running, no teleportation -- and where
reconstruction cannot do it, a PURELY LOCAL generative repair". This is how that works in practice
on this pipeline, what is built, and what decides when the generative step is allowed.

## The terms

Every term is a pure function in `nfl_gsplat/render/motion_rulers.py`, read from the timeline exactly
as `05k` draws it, printed by `scripts/07l_measure_plausibility.py` and written to
`$DIAG/<play>_<tag>_plausibility.json` so two versions are one diff.

| term | what it reads | a real player | what v38 read |
|---|---|---|---|
| contiguous step | root distance between consecutive drawn frames | < 0.20 m/frame (12 m/s) | 26 steps > 0.6 m, top 73 m/s |
| root jitter | second difference of the root | ~0.003 m/frame^2 | worst ids 0.15-0.25 |
| joint jitter | second difference of pelvis-relative joints, max over joints | 0.01-0.02; 0.05 is a visible twitch | p90 median 0.15 across ids |
| joint speed | first difference of the same | a sprinting hand 0.17 m/frame | hands 0.4-0.6 |
| hinge violations | knees/elbows bent backwards (< -15 deg) or sideways (> 35 deg off axis) | 0 | 3-14 % of hinge-frames |
| handover steps | a step whose two frames came from different camera sets | -- | 11 of 214 |
| census | \|KC - 11\| + \|BAL - 11\| per frame | 0 | 2.0 on the play |

Two rules for reading them, learned the hard way (see HANDOFF, 2026-09-15):

1. **State the population and the window.** Whole-clip numbers are the post-whistle crowd; the live
   window (300-460 on play 1) is what the user watches. `07l` prints both.
2. **A repair needs a second ruler.** Jitter alone crowns a frozen mannequin; the census alone is blind
   to a man drawn in the wrong place. The second ruler for pose is the limbs' reprojection onto the
   footage keypoints in BOTH cameras (the sideline's depth axis is the endzone's lateral axis); for
   placement it is the endzone reprojection of the lower joints. Every fix in this document was
   accepted or rejected on the pair, never on one.

## Normalising into one loss

The terms have different units. For gating repairs they are used as **exceedances**, not as a sum:
a frame (or an id's stretch of frames) is *implausible* when any term exceeds its threshold --
step > 0.25 m/frame, joint jitter > 0.05 m/frame^2, a hinge violation, joint speed > 0.25 m/frame.
A single scalar is only needed for ranking stretches to look at, and there the exceedance ratio
(value / threshold, max over terms) is enough. Summing weighted terms was considered and dropped:
every weight would be a knob, and the thresholds already come from physics.

## What classical repair fixed, and what it could not

Applied on play 1, each measured on its pair of rulers:

- teleports: four mechanisms, each with a cut or a gate (`08t`, `08u`, `smooth_xy` within runs,
  `08v` to keep the poses) -- 26 -> 1 step over 0.6 m whole clip; 21 remain over 0.25 on the play.
- limb jitter: a Gaussian on the axis-angles (`POSE_SMOOTH_SIGMA`) -- p90 0.23 -> 0.11.
- weird joints: `clamp_hinges` -- violations to zero for +1.7 px at the endzone p90. The collars and
  spine were NOT clamped (measured: the limit moves the arm off the keypoints); their repair is a
  pose prior in the fit.

What is left after these are stretches where the fit itself is garbage rather than noisy: the ids
that carry the joint-jitter tail (4, 9, 162, 165) with hands at 25-37 m/s and hinges every which way.
No smoother can make a good pose from a bad one.

## The local generative repair, in practice

Scope: a single id over a contiguous stretch of frames where the exceedance stays high AFTER the
classical passes, bounded by clean frames on both sides (or a track end). Everything outside the
stretch is untouched -- that is what "purely local" means here.

What is generated: the **pose**, never the root. The root comes from the cameras and is clean; the
pose is what the fit got wrong. The generator is conditioned on what is known and reliable:

- the root trajectory (speed, heading) over the stretch and its margins,
- the clean poses at both boundaries,
- the 2D keypoints where confident (so the repair still faces the footage),
- the player's own clean frames elsewhere in the play (his gait, his stance).

Order of preference, cheapest first, each accepted only if it lowers the exceedance without raising
the reprojection beyond the boundary frames' own level:

1. **Interpolate through the stretch** between the boundary poses (SLERP), when the stretch is short
   (< 0.3 s) -- a limb that flails for ten frames is more plausible held than fitted.
2. **Borrow from the same player**: the clean stretch of his own play with the nearest root speed
   and heading, time-warped to the root trajectory. Nothing is invented; he did move like this.
3. **A motion prior**: a short-window pose model (a VPoser-style latent, or a small learned
   sequence prior over SMPL-X body_pose) sampled conditioned on the boundaries and the root speed,
   with the confident keypoints as a soft term. This is the only step that generates, and it is
   allowed only when 1 and 2 fail the ruler.

Two guards make this safe to ship:

- **Provenance.** Every repaired frame carries a flag (`source = "repair:<method>"`) through the
  timeline to the render, so a viewer (or the overlay) can show which limbs are measured and which
  are inferred. A repair that cannot be pointed at is a lie.
- **The ruler decides, not the eye.** The stretch is re-scored after repair on the same terms and
  the same second ruler; a repair that lowers jitter while pulling the limbs off the keypoints is
  rejected, like every correction before it (`corrections-must-beat-what-they-correct`).

Not built yet: the stretch finder (exceedance runs per id) and steps 1-2. Step 3 needs a prior and
is deliberately last.
