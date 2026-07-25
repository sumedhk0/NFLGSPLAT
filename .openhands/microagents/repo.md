---
name: repo
type: repo
agent: CodeActAgent
---

# NFLGSPLAT — repository context (always loaded)

You are working on **NFLGSPLAT**: NFL All-22 broadcast footage → camera
calibration → free-viewpoint Gaussian Splatting.

**Read these first for full project context (they replace prior chat history):**
1. `docs/HANDOFF.md` — current state, what's done, the exact NEXT STEP.
2. `docs/agent-context/MEMORY.md` — index of durable project memory; then the
   individual `docs/agent-context/*.md` files it lists.
3. For any active feature: its spec in `docs/superpowers/specs/` and plan in
   `docs/superpowers/plans/`.

**Hard rules:**
- NEVER commit real NFL video/frames. `data/` and `kp_eval/` are gitignored;
  diagnostics go outside the repo.
- All GPU jobs run on GT PACE Phoenix `embers` partition (account
  `paceship-pso`); checkpoint for preemption.
- Fail loud (`SetupError`/`CalibrationError` + actionable pointer); no silent
  fallback that changes numerical results.
- Feature branch off `main` → test-first (TDD) → `--no-ff` merge → delete
  branch. Commit/push only when the user asks.
- End commit messages with a `Co-Authored-By:` trailer.

**Environment:** Python 3.11. Tests: `python -m pytest -m "not gpu and not slow" -q`.
Lint: `python -m ruff check nfl_gsplat tests scripts`. PaddleOCR is PACE-only
(`nfl_smplx` env); local uses easyocr as an OCR proxy.

**Where things are:** calibration in `nfl_gsplat/calibration/`, tracking/identity
in `nfl_gsplat/tracking/` + `nfl_gsplat/identity/`, entry scripts in `scripts/`.
The live task is the **jersey-identity endzone calibration** — see HANDOFF.md
"NEXT STEP" for the acceptance runbook.
