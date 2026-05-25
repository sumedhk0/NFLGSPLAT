# NFL Broadcast → Free-Viewpoint Gaussian Splatting Pipeline

Reconstruct NFL plays from two synchronized broadcast feeds (sideline + endzone) as free-viewpoint 3D Gaussian Splatting scenes, renderable from any virtual camera.

CPU test suite: **140 passing** (calibration, tracking, field I/O, triangulation, pose fusion, ball Kalman + football asset, avatars, PLY merge, identity registry, avatar/shape library, scene compositing, end-to-end smoke). GitHub Actions runs `pytest -m "not gpu and not slow and not real_video"` + ruff on every push; see `.github/workflows/ci.yml`.

## What this pipeline does

1. **Camera calibration** — human-annotated field landmarks + PnP solve in a metric world frame.
2. **Static field reconstruction** — 3D Gaussian Splatting of the stadium from pre-snap empty-field frames.
3. **Per-player SMPL-X pose fusion** — SMPLest-X on each view, triangulate body joints, refit SMPL-X to 3D joints, smooth with a 1€ filter.
4. **Animatable Gaussian avatars** — feed-forward LHM++ by default; 3DGS-Avatar optimization for "hero" players (QB, key WR).
5. **Ball trajectory** — football-tuned YOLO + 3D Kalman with a gravity prior.
6. **Compositing + novel-view render** — concatenate field + avatars + ball into one Gaussian batch, rasterize along a virtual camera trajectory, encode to MP4.

Output: `outputs/{game_id}/{play_id}/render.mp4`.

## Quick start

```bash
bash scripts/00_setup_environments.sh
bash scripts/01_download_models.sh

# Interactive: click landmarks on one frame per camera
python scripts/02_calibrate_cameras.py --game game_001

# User places synced feeds at data/raw/game_001/{sideline,endzone}.mp4

bash scripts/03_reconstruct_field.sh game_001
bash scripts/04_process_play.sh game_001 play_001
python scripts/05_render_novel_view.py --game game_001 --play play_001 \
    --trajectory configs/trajectories/fly_through.yaml
```

On a SLURM cluster:

```bash
sbatch scripts/slurm/full_game.sbatch game_001
```

## Prerequisites (see SETUP.md for full details)

- **SMPL-X / SMPL body models** — license-gated; manual download.
- **Human-annotated calibration landmarks** — the `annotate_gui.py` click tool.
- **Real broadcast video** — user-provided.

The pipeline fails loudly with a precise pointer to SETUP.md if any of these are missing. Nothing is stubbed with fake data.

## Known limitations

- **Broadcast pan/zoom drift.** Single-frame calibration is insufficient for live plays. We support per-keyframe re-calibration with linear interpolation of extrinsics. For rapid zoom changes, re-annotate more keyframes or expect position error at the meter scale.
- **Football pads, not everyday clothes.** SMPLest-X was trained on unpadded bodies. Poses will be plausible but not perfect. This is a quality ceiling we document rather than fight.
- **Two views is minimal.** Occluded limbs jitter; mitigated with the 1€ filter and short-gap interpolation, but not fully.
- **Crowds are blurry.** The field looks sharp at 30 000 splatfacto iters; the stands are blurry and this is acceptable.
- **LHM++ weights from Alibaba OSS.** Slow download from US networks; `01_download_models.sh` supports resume.
- **Env-gated adapters.** SMPLest-X, LHM++, and 3DGS-Avatar each pin incompatible CUDA / PyTorch versions, so their in-tree wrappers are thin *seams* that shell out to the external repo inside the right conda env. Calling them outside their env raises `NotImplementedError` pointing to SETUP.md §8. This is what keeps the four stages coexisting.
- **Season-scale reuse changes outputs slightly (on purpose).** With the avatar/shape library (SETUP.md §9), each player is reconstructed once and reused across plays/games, and `betas` are frozen to the library value. This makes appearance consistent play-to-play and the pose solve more stable, but is *not* bit-identical to re-estimating per play. Set `pose.refit.use_library_betas: false` to re-estimate betas per play (only sensible without the library, since per-play betas mismatch a cached avatar's rig).
- **Roster prior coverage + alignment.** Player recognition is far more robust with the nflverse roster/participation prior, but participation data isn't available for every season, and mapping play-by-play to our frame windows depends on `plays.yaml` `meta:`. Without a roster the pipeline still runs in `identity.source=ocr_only` mode (OCR + jersey color), with no cross-game identity guarantees.
- **Referees and the ball are generic assets, not reconstructions.** Officials render as a single shared striped-shirt avatar posed by their solved motion; non-roster non-referees are dropped. The football is one authored canonical asset oriented along the Kalman velocity (with optional spin) — broadcast footage is too small/blurry for a true per-ball reconstruction.

## Environment matrix

| Stage | Conda env | PyTorch | CUDA |
|---|---|---|---|
| Pose (SMPLest-X) | `smplx` | 2.1 | 12.1 |
| Field + render (gsplat / nerfstudio) | `gsplat` | 2.3 | 12.1 |
| LHM++ feed-forward | `lhm` | 2.1 | 12.1 |
| 3DGS-Avatar (heroes) | `avatar` | 2.0 | 11.8 |

Stages hand off through on-disk files only (NPZ, PLY, Parquet, JSON). That is what lets them coexist despite incompatible pins.

## Repository layout

```
envs/                 Conda environment YAMLs (4 envs)
configs/              OmegaConf YAML pipeline + stage configs
nfl_gsplat/
  calibration/        PnP + landmark annotation
  field/              Pre-snap frame extraction + splatfacto
  tracking/           YOLO + BoT-SORT + cross-camera re-ID
  pose/               SMPLest-X + triangulation + SMPL-X refit
  avatars/            LHM++ wrapper + 3DGS-Avatar + LBS animate
  ball/               Football YOLO + 3D Kalman
  compositing/        PLY merge + gsplat rasterization + viser viewer
  utils/              io, video, geometry, logging
scripts/              Orchestration shell + python + sbatch
tests/                pytest, including synthetic-fixture smoke test
data/                 Raw video, annotations, body models (gitignored)
outputs/              Per-game / per-play caches + renders (gitignored)
```

## Testing

```bash
pytest tests/                                    # all unit + smoke
pytest tests/test_calibration.py -v              # calibration gate
pytest tests/test_pipeline_smoke.py -v           # full pipeline, < 5 min
pytest -m "not gpu"                              # CPU-only (CI)
```

The smoke test generates a synthetic field + 3 SMPL-X bodies + 2 calibrated cameras and exercises every stage end-to-end with LHM++ mocked. It is what catches integration breakage as external repos update.

## Troubleshooting

- `SetupError: SMPL-X model not found at data/body_models/smplx/SMPLX_NEUTRAL.npz` — see SETUP.md §2.
- `CalibrationError: reprojection error 11.3 px exceeds threshold 5.0 px` — re-annotate with more landmarks (`python scripts/02_calibrate_cameras.py --game ... --camera ...`), or pick a different keyframe. See SETUP.md §3.
- `LHMVRAMError: 11.4 GB free, minimum 16 GB for LHM-1B and 8 GB for LHM-MINI` — pipeline refuses to silently pick a different model. Free VRAM, select a smaller model explicitly via `avatars.lhm.model=lhm_mini`, or move to a larger GPU.
- gsplat / nerfstudio import conflicts — always use the `gsplat` conda env for field + composite stages. Mixing them with `smplx` or `lhm` envs breaks.
