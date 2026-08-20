# Training the field splat on PACE

Trains `field.ply` from a **prepared** `field/transforms.json` — the poses are
not re-derived on PACE, so calibration stays where it was validated.

Everything below is copy-paste. Local shell = your Windows machine, PACE shell =
`login-phoenix.pace.gatech.edu`.

---

## 0. What actually has to travel

`data/` and `logs/` are both gitignored, so **the repo carries no play data**.
Two separate transfers:

| What | How | Size |
|---|---|---|
| code | `git pull` on PACE | — |
| `field/frames/` + `field/transforms.json` | `scp` from local | ~257 MB |

Videos, `cameras.npz`, `tracks.parquet` and `meta.yaml` are **not** needed —
nothing in this job re-derives a pose.

---

## 1. Local — confirm what you are about to send

```bash
cd /c/Users/sumedh/NFLGSPLAT
git push origin main
python -c "import json; d=json.load(open(r'data/2025/week_04/SEA_at_AZ/play_001/field/transforms.json')); import collections; print(len(d['frames']),'poses',dict(collections.Counter(f['camera'] for f in d['frames'])))"
```

Expect `120 poses {'sideline': 60, 'endzone': 60}`.

---

## 2. PACE — get the code

```bash
ssh skothari67@login-phoenix.pace.gatech.edu

# first time only
cd ~/scratch && git clone https://github.com/sumedhk0/NFLGSPLAT.git

cd ~/scratch/NFLGSPLAT
git pull origin main
mkdir -p logs                     # REQUIRED: Slurm opens --output BEFORE the
                                  # job runs, so a missing logs/ dir makes the
                                  # job die with no output at all
```

---

## 3. PACE — build the conda env (first time only, ~30-60 min)

This previously failed twice with `CondaVerificationError: ... libabseil ...
appears to be corrupted`. That is a **full or inode-capped filesystem**, not a
bad package. Put conda's package cache and envs on scratch first:

```bash
pace-quota                        # check home is not full, and inodes

mkdir -p ~/scratch/conda/pkgs ~/scratch/conda/envs
export CONDA_PKGS_DIRS=~/scratch/conda/pkgs
export CONDA_ENVS_DIRS=~/scratch/conda/envs
conda clean -a -y                 # drop any half-extracted packages

bash scripts/00_setup_environments.sh --only nfl_gsplat
```

Put the two `export` lines in `~/.bashrc` so later sessions agree with this one.

Verify before burning GPU time:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda/envs/nfl_gsplat
python -c "import torch, gsplat, nerfstudio; print(torch.__version__, torch.cuda.is_available())"
command -v ns-train ns-export
```

`torch.cuda.is_available()` prints `False` on a login node — that is expected
and fine. `ns-train` and `ns-export` must both resolve.

---

## 4. Local — upload the frames and poses

New shell on your Windows machine:

```bash
cd /c/Users/sumedh/NFLGSPLAT
PLAY=data/2025/week_04/SEA_at_AZ/play_001
DEST=skothari67@login-phoenix.pace.gatech.edu:~/scratch/NFLGSPLAT/$PLAY

ssh skothari67@login-phoenix.pace.gatech.edu "mkdir -p ~/scratch/NFLGSPLAT/$PLAY/field"
scp -r "$PLAY/field/frames" "$PLAY/field/transforms.json" "$DEST/field/"
```

`rsync -avP` instead of `scp -r` if you have it — it resumes.

---

## 5. PACE — submit

```bash
cd ~/scratch/NFLGSPLAT
sbatch scripts/slurm/field_train_only.sbatch data/2025/week_04/SEA_at_AZ/play_001
```

The job checks, in order: `transforms.json` exists, the conda env resolves,
`ns-train`/`ns-export` are on PATH, and **every image the poses reference is
actually present** — all before requesting real work. Add `-A <account>` to
override the default `paceship-pso`.

```bash
squeue -u $USER                                # queued / running
tail -f logs/nfl-field-train-<jobid>.out       # live
```

Early in the log you want:

```
env: /storage/.../scratch/conda/envs/nfl_gsplat
torch 2.3.1 cuda True
transforms.json: 120 poses, all images present
```

`cuda True` here is on the compute node, and matters.

---

## 6. Collect the result

```bash
# on PACE
ls -lh data/2025/week_04/SEA_at_AZ/play_001/field.ply

# from local
scp skothari67@login-phoenix.pace.gatech.edu:~/scratch/NFLGSPLAT/data/2025/week_04/SEA_at_AZ/play_001/field.ply .
```

---

## Notes

**QOS is `embers`** — preemptible, which is why the job sets `--requeue`. A
requeued job restarts from the beginning; `splatfacto` at 30k iterations fits
inside the 2 h wall clock with room, so this is cheap insurance rather than a
real risk.

**Do not use `field_recon.sbatch` for this.** Its middle stage regenerates
`transforms.json` from `cameras.npz` — which is not even uploaded here, so it
would fail, and if it were uploaded it would rebuild the poses on PACE rather
than using the ones validated locally.

**Only 120 of the play's frames have a verified camera**, and only those were
extracted. That is the intended input, not a shortfall — see SETUP.md on why
roughly half of an endzone play's frames carry no evidence to verify against.
