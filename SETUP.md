# Setup

Three things must be in place before the pipeline will run. None of them can be automated. All of them are detected at runtime and produce a precise pointer back to the relevant section below.

---

## §1. Conda environments

The pipeline uses four conda environments to isolate conflicting CUDA / PyTorch pins across SMPLest-X, gsplat+nerfstudio, LHM++, and 3DGS-Avatar.

```bash
bash scripts/00_setup_environments.sh
```

This creates:

| Env | Purpose | PyTorch | CUDA |
|---|---|---|---|
| `nfl_smplx` | SMPLest-X pose inference | 2.1 | 12.1 |
| `nfl_gsplat` | Field reconstruction + compositing | 2.3 | 12.1 |
| `nfl_lhm` | LHM++ feed-forward avatars | 2.1 | 12.1 |
| `nfl_avatar` | 3DGS-Avatar per-hero optimization | 2.0 | 11.8 |

Version pins are exact in `envs/*.yml`. Do not upgrade blindly — `gsplat` and `nerfstudio` periodically break each other.

---

## §2. Body models (SMPL-X and SMPL)

These are license-gated and require manual download.

1. Register at **https://smpl-x.is.tue.mpg.de/** and accept the license.
2. Download **"SMPL-X v1.1 (NPZ+PKL)"**.
3. Register at **https://smpl.is.tue.mpg.de/** and accept the license.
4. Download **"SMPL v1.1.0"** (needed for SMPLest-X compatibility).
5. Place files as:

```
data/body_models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
└── smpl/
    ├── SMPL_NEUTRAL.pkl
    ├── SMPL_MALE.pkl
    └── SMPL_FEMALE.pkl
```

The pipeline checks for `data/body_models/smplx/SMPLX_NEUTRAL.npz` at startup. Missing file → `SetupError` pointing to this section.

---

## §3. Camera calibration (human-in-the-loop)

Every camera pair must be calibrated once per keyframe. NFL broadcast cameras pan and zoom during plays, so we re-calibrate at keyframes (default: once per play start) and linearly interpolate extrinsics between them.

### First-time run

```bash
python scripts/02_calibrate_cameras.py --game game_001
```

For each camera, a clickable OpenCV window shows a reference frame overlaid with the list of expected NFL field landmarks (yard-line intersections with sidelines and hash marks, pylons, goalpost bases). Click each visible landmark. Press `s` to save, `q` to quit.

Annotations are written to `data/annotations/{game_id}/{cam}_landmarks.json`. The PnP solver runs immediately and reports the reprojection RMS. If RMS > 5 px the script refuses to save — re-click or add more landmarks.

**Minimum annotations per camera: 6.** With 10+, bundle adjustment via `cv2.calibrateCamera` refines intrinsics too.

### Adding a keyframe for mid-play zoom changes

```bash
python scripts/02_calibrate_cameras.py --game game_001 --camera sideline --keyframe-frame 4200
```

Extrinsics between keyframes are linearly interpolated. This is the documented limitation — for smoother tracking of aggressive pans, annotate more keyframes.

---

## §4. Non-gated model weights

```bash
bash scripts/01_download_models.sh
```

Downloads:

- **SMPLest-X-H32** checkpoint (from SMPLer-X release on GitHub)
- **LHM-1B** and **LHM-MINI** weights (Alibaba OSS — slow from US, `aria2c` with resume is used)
- **3DGS-Avatar** release weights
- **YOLOv8x** person detector (auto-download via ultralytics on first use)
- **Football-tuned YOLOv8** ball detector (roboflow public dataset, fine-tuned locally — weights cached at `weights/ball_yolov8.pt`)
- **ViTPose** weights via mmpose

All downloads are resumable. Re-running the script is idempotent.

---

## §5. Raw video

You supply the broadcast feeds. They must be **already synchronized** (same start frame, same FPS).

```
data/raw/{game_id}/
├── sideline.mp4
└── endzone.mp4
```

The pipeline validates on startup that both files exist and have matching frame counts (±2 frames tolerance for broadcast jitter).

---

## §6. SLURM configuration

Edit `configs/pipeline.yaml` under the `slurm:` key to match your cluster:

```yaml
slurm:
  account: your_account
  partition: your_gpu_partition
  gpu: h100:1
  time_field: "02:00:00"
  time_play: "01:00:00"
  mem: "64G"
```

The `scripts/slurm/*.sbatch` files template these values via `envsubst` on launch.

---

## §7. Verifying the setup

```bash
# Should pass on CPU:
pytest tests/test_calibration.py -v
pytest tests/test_triangulation.py -v
pytest tests/test_ply_merge.py -v

# Full synthetic pipeline, CPU-only with mocked LHM++:
pytest tests/test_pipeline_smoke.py -v
```

If all four suites pass, scaffolding and the numerical foundations are sound. Real-video runs require the GPU envs + body models + annotations above.

---

## §8. Env-gated adapters (SMPLest-X, LHM++, 3DGS-Avatar)

Three heavy external models are treated as *env-gated adapters* rather than vendored code:

| Model | Wrapper module | External repo path (default) | Runs in |
|---|---|---|---|
| SMPLest-X-H32 | `nfl_gsplat/pose/smplestx_infer.py` | `third_party/SMPLer-X/` | `nfl_smplx` |
| LHM-1B / LHM-MINI | `nfl_gsplat/avatars/lhm_wrapper.py` | `third_party/LHM/` | `nfl_lhm` |
| 3DGS-Avatar | `nfl_gsplat/avatars/gdgs_avatar_train.py` | `third_party/3dgs-avatar-release/` | `nfl_avatar` |

Each wrapper in-tree is a **stable seam**: it validates prerequisites (weights + repo checkout), sets up the call signature, and then shells out to the real adapter via `scripts/04_process_play.sh`. The body of each `infer_*` / `generate_*` / `train_*` function raises `NotImplementedError` if imported outside its conda env — this is deliberate so unit tests and CI can still import the pipeline on CPU without pulling torch, CUDA, or gated weights.

If you see:

```
NotImplementedError: SMPLest-X adapter is env-gated; run inside the nfl_smplx conda env
via scripts/04_process_play.sh. See SETUP.md §8 for the adapter wiring.
```

Either you are running a production command outside the right conda env, or `scripts/01_download_models.sh` has not finished cloning the external repo. Check:

1. `conda activate nfl_smplx` (or `nfl_lhm`, `nfl_avatar` depending on the stage).
2. `ls third_party/SMPLer-X/` (or `LHM/`, `3dgs-avatar-release/`) is non-empty.
3. The stage is being invoked through `scripts/04_process_play.sh`, not imported directly from the CI env.

**Why this design.** The adapters pin incompatible PyTorch / CUDA versions; vendoring any one of them would force the entire pipeline onto its pins and break the others. Env-gating is what lets all four coexist.
