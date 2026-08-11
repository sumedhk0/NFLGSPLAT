"""Run one precompute STAGE across a batch of plays, so you issue one command
per stage instead of per play. Continues on failure and prints a summary.

Stages (run in this order):
  roboflow  — sideline field keypoints (LOCAL, needs internet + ROBOFLOW_API_KEY)
  sideline  — sideline calibration -> cameras.npz  (local or PACE)
  identity  — detect+track + jersey OCR + player_uid (PACE, nfl_smplx, GPU)

Examples:
  set ROBOFLOW_API_KEY=...
  python scripts/run_precompute_batch.py --stage roboflow --plays BATCH
  python scripts/run_precompute_batch.py --stage sideline --plays BATCH
  python scripts/run_precompute_batch.py --stage identity --plays BATCH --device cuda

--plays accepts a comma list (play_002,play_004,...) or the literal BATCH (the
recommended first batch below). Paths are under data/2025/week_04/SEA_at_AZ/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = Path("data/2025/week_04/SEA_at_AZ")
BATCH = ["play_001", "play_002", "play_004", "play_005", "play_009",
         "play_011", "play_014", "play_016", "play_026", "play_029"]
MODEL_ID = "football-field-key-points-mvmjf/2"


def _cmd(stage: str, play: str, season: int, device: str) -> list[str]:
    pdir = GAME_DIR / play
    if stage == "roboflow":
        return [sys.executable, "scripts/03_roboflow_precompute.py",
                str(pdir / "sideline.mp4"),
                "--out", str(pdir / "roboflow_kps_sideline.json"),
                "--model-id", MODEL_ID]
    if stage == "sideline":
        return [sys.executable, "scripts/02_autocalibrate.py",
                "--play-dir", str(pdir), "--mode", "pretrained",
                "--cameras", "sideline"]
    if stage == "identity":
        return [sys.executable, "scripts/03c_identity_tracks.py", str(pdir),
                "--season", str(season), "--device", device]
    raise SystemExit(f"unknown stage {stage!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["roboflow", "sideline", "identity"])
    ap.add_argument("--plays", default="BATCH",
                    help="comma list of play dirs, or 'BATCH' for the default set")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--device", default="cuda", help="identity stage only")
    args = ap.parse_args()

    plays = BATCH if args.plays.strip() == "BATCH" else \
        [p.strip() for p in args.plays.split(",") if p.strip()]
    # play_001 already has roboflow + sideline; skip those stages for it.
    if args.stage in ("roboflow", "sideline"):
        plays = [p for p in plays if p != "play_001"]

    # The stage scripts import nfl_gsplat, but `python scripts/x.py` only puts
    # scripts/ on sys.path — and the repo cannot be pip-installed editable
    # everywhere (pyproject pins numpy<2 for paddle, which would break a cu128
    # torch). Hand the repo root down via PYTHONPATH instead.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    print(f"stage={args.stage}  plays={plays}\n")
    ok, failed = [], []
    for play in plays:
        if not (GAME_DIR / play).is_dir():
            print(f"[SKIP] {play}: dir not found"); failed.append(play); continue
        cmd = _cmd(args.stage, play, args.season, args.device)
        print(f"[RUN ] {play}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        (ok if rc == 0 else failed).append(play)
        print(f"[{'OK  ' if rc == 0 else 'FAIL'}] {play} (exit {rc})\n", flush=True)

    print(f"\n=== stage {args.stage}: {len(ok)} ok, {len(failed)} failed ===")
    if failed:
        print(f"failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
