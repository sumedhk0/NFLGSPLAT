# Design: photometric appearance for the avatar twin (M8)

Status: design, 2026-09-03. Not built. The last large unbuilt piece of the
pipeline; everything upstream (cameras in the field frame, fused SMPL-X bodies,
per-vertex Gaussians with footage colours) exists and runs end to end.

## What exists, and what it is not

`compositing.mesh_to_gaussians` binds one flat Gaussian per SMPL-X vertex; its
colour is the per-vertex MEDIAN of the footage pixels under vertices that face the
camera (`compositing.appearance`, over every frame the body was posed in). That is
a sampled texture, not a fit: nothing is optimised against the images, the
Gaussians' scales and opacities are fixed by mesh spacing, and a vertex never seen
by either camera keeps the team colour.

## Constraints that shape the design (measured, not assumed)

- Two fixed cameras, ~100 m away, 8-12 degree lenses. A player is 120-160 px
  tall in the sideline view, larger in the endzone view. Per-frame appearance
  detail is ~1-2 cm per pixel at best.
- The camera track is per frame and in the rule-book field frame (`08d`), with
  cross-field scale from the hash/numeral rulers; reprojection of a fused body is
  good to ~0.9 m in placement (cross-view gap) and 3-4 cm in joint rms after the
  refit. Placement error dominates: a photometric loss must tolerate a body that
  is misplaced by a few pixels, or it fits blur.
- No CUDA toolkit locally, so `gsplat`'s rasteriser does not JIT here; the CPU
  splatter (`preview_cpu`) is not differentiable. PACE is off the table unless the
  user asks. So the optimiser has to be pure torch, GPU-capable, and small.
- Occlusion between players is real and frequent at the snap; the appearance
  pass already uses a facing mask but no inter-body occlusion test.

## Approach

Per player (not per scene), per play:

1. **Parameterisation.** Keep the vertex binding (one Gaussian per vertex, in the
   body's canonical frame). Optimise per-Gaussian colour (RGB) and a per-Gaussian
   log-scale multiplier and opacity; positions stay bound to the vertex (no
   drift -- the pose stage owns geometry). Initialise colour from the median
   texture, scale from mesh spacing, opacity 0.99.
2. **Renderer.** A pure-torch differentiable splatter: project each Gaussian's
   centre and 2D covariance through the per-frame camera (flat Gaussian: the
   in-plane extent is the two tangent directions, thickness ~0), tile the image,
   alpha-composite front to back. Bodies are small (10k Gaussians into ~150x150
   px), so a per-body crop of the frame is enough and the tile count is tiny.
   The whole scene is never rendered during the fit.
3. **Loss.** L1 on the body's crop against the frame, masked to the body's
   projected silhouette dilated by 2 px, with a per-frame 2-D translation nuisance
   parameter (2 floats per frame per body) so a few pixels of placement error are
   absorbed rather than fitted. Total-variation prior on colour over mesh edges
   to keep the number readable. Other bodies' silhouettes (from their own
   projections) are excluded from the mask: no inter-body occlusion handling
   beyond that in v1.
4. **Data.** Every frame in which the body is posed (both views), stride 6 as the
   pose cache; ~60 crops per body per play. Both views in one fit, which is the
   point: the endzone sees the back, the sideline the profile.
5. **Output.** The same GaussianBatch format `05d` and `05h` already consume, so
   the render stage does not change; per-player `appearance_<pid>.npz`
   (colour, scale multiplier, opacity) loaded by `05d --fitted-appearance`.

## Acceptance (before it replaces the median texture)

Measured on play 1, the same bodies, held-out frames (every 5th posed frame):

- Held-out crop L1 must drop against the median-texture render, per body; report
  the median over bodies and the worst body.
- Jersey number legibility: the OCR read rate on the RENDERED body (endzone view
  of the back) must not fall below the median texture's -- numbers are the
  thing a viewer looks for.
- No colour bleed onto turf: the fitted opacity of vertices that never face a
  camera must not drift toward turf green (median colour of those vertices vs
  the team colour, reported).

If held-out L1 improves but legibility drops, the TV prior is too weak or the
translation nuisance is absorbing motion it should not; do not ship that.

## What NOT to do

- Do not fit Gaussian positions or add Gaussians. Geometry belongs to the pose
  stage; free positions would fit placement error into the body.
- Do not fit a whole-scene splat from two fixed cameras (measured earlier: two
  views 90 degrees apart at 100 m do not constrain a free splat; the avatar
  prior is the product decision).
- Do not start with the field: the field is procedural and correct in the
  rule-book frame; footage-fitting it buys nothing a viewer notices.

## Size

Renderer + loss + tests: about a day. Fit on play 1 (both views, ~30 bodies):
minutes on the local GPU per body. Acceptance harness: half a day.
