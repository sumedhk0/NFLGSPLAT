#!/bin/bash
# One All-22 play, end to end, RESUMABLE. Every stage leaves a marker in the
# play-dir (.done_<stage>) and is skipped when the marker exists; --fresh
# wipes the markers and the stage outputs first. The machine this runs on
# is switched off from time to time, so a run that dies mid-stage costs
# that stage only: re-run the same command and it continues.
#
#   bash scripts/pipeline_play.sh <play-dir> <sideline.mp4> <endzone.mp4> <los-yards> [--fresh] [--from-paint]
#
# Stages (play-dir relative):
#   paint     scripts/08 (full paint solve of the sideline + endzone)  -> recon.npz     [--from-paint only]
#   export    scripts/08b                                             -> cameras.npz, tracks.parquet
#   refine    scripts/08e (every frame's camera to the paint)     -> cameras.npz
#   shift     scripts/08d --no-rows --apply                           -> cameras.npz in the field frame
#   endzone   scripts/08 --sideline-from (mirror check)               -> recon_abs.npz, then 08b again
#   check     scripts/08d --los-yards (prints rulers, LOS)            -> field_offset.json
#   pose_s    scripts/05c sideline (resumes per frame)                -> poses_sideline.json
#   pose_e    scripts/05c endzone --match-frames                      -> poses_endzone.json
#   identity  scripts/08c --week 1                                    -> identity_resolved.pkl
#   keypoints scripts/05m (YOLOv8-pose per tracked person, both views)  -> keypoints_2d.parquet
#   tri       scripts/05n (joints triangulated with both cameras)       -> poses_tri.json
#   fuse      scripts/05e (monocular joints fused) -- OPT-IN, FUSE=1; 2.6-3.4x worse than tri
#   refit     scripts/05f                                             -> poses_refit.json
#   fit       scripts/05i (appearance fit from the footage) -- OPT-IN, FIT=1: the hi-fi render
#             wears synthetic uniforms (render.uniform); fitted textures measured no better
#   field     scripts/05l footage warped onto the ground plane        -> <play-dir>/field_texture.npz (+PNG in diag)
#   teams     scripts/08f team per id from torso colour (bimodal only) -> <play-dir>/team_by_colour.json
#   hifi      scripts/05k 1080p GPU render on the footage field        -> <play-dir>/render_hifi/
#   render    scripts/05d world mode, fitted appearance              -> <play-dir>/render_abs/
#
# Environments: nflgsplat for calibration/identity, smplx312 for pose/fuse/refit/render
# (pickles written under numpy 2 do not load under numpy 1 -- keep it that way).
set -u
# A stage is a python run piped through grep for the log; without pipefail
# the grep decided the stage's fate and a traceback that contained the
# word "shift" passed the shift stage (play 2, 2026-09-03).
set -o pipefail
export PYTHONIOENCODING=utf-8 PYTHONPATH="C:/Users/sumedh/NFLGSPLAT"
PYN="C:/venvs/nflgsplat/Scripts/python.exe"; PYS="C:/venvs/smplx312/Scripts/python.exe"
cd "C:/Users/sumedh/NFLGSPLAT" || exit 1

P="$1"; SIDE="$2"; END="$3"; LOS="$4"; shift 4
DIAG="C:/Users/sumedh/diag"; PLAY="$(basename "$P")"
RED="${RED:-KC}"; WHITE="${WHITE:-BAL}"          # the saturated and the white kit (08f, render.uniform)
SEED_FROM="${SEED_FROM:-}"                       # a solved play-dir of the same game: its sideline mount seeds 08
FRESH=0; FROM_PAINT=0
for a in "$@"; do
  case "$a" in --fresh) FRESH=1;; --from-paint) FROM_PAINT=1;; esac
done
ROOT="$(dirname "$P")"
NAME="$(basename "$P")"
mkdir -p "$P"

log()  { echo; echo "=== $(date +%H:%M:%S) [$NAME] $1"; }
done_() { [ -f "$P/.done_$1" ]; }
mark() { date +%s > "$P/.done_$1"; }
fail() { echo "FAILED at $1 -- re-run the same command to resume"; exit 1; }

if [ "$FRESH" = 1 ]; then
  log "fresh: wiping markers and stage outputs"
  rm -f "$P"/.done_* "$P/poses_sideline.json" "$P/poses_endzone.json" "$P/poses_fused.json" \
        "$P/poses_refit.json" "$P/identity_resolved.pkl" "$P/identity_unnamed.pkl" "$P/identity_fused.pkl" \
        "$P/tracks_identity.parquet" "$P/cameras_relative.npz" "$P/field_offset.json"
fi

if [ "$FROM_PAINT" = 1 ] && ! done_ paint; then
  log "paint solve (08)"
  "$PYN" scripts/08_reconstruct_all22.py --root "$ROOT" --sideline "$SIDE" --endzone "$END" --no-mirror-check ${SEED_FROM:+--seed-from "$SEED_FROM"} \
     --out "$P/recon.npz" 2>&1 | grep -v "Warning\|warn" | grep -E "candidate|rulers|pass the|gap  |reconciled  |player height|Error|Exit|refus" || fail paint
  rm -f "$P/.done_export" "$P/.done_refine" "$P/.done_shift" "$P/.done_endzone"
  mark paint
fi

if ! done_ export; then
  log "export (08b)"
  RECON="$P/recon.npz"; [ -f "$RECON" ] || RECON="C:/Users/sumedh/diag/all22_reconstruction.npz"
  "$PYN" scripts/08b_export_play_dir.py --recon "$RECON" --root "$ROOT" --sideline "$SIDE" --endzone "$END" --out "$P" \
     2>&1 | grep -v "Warning\|warn" | grep -E "cameras:|linked|tracks.parquet" || fail export
  mark export
fi

if ! done_ refine; then
  log "refine every frame's camera to the paint (08e)"
  "$PYN" scripts/08e_refine_cameras.py --play-dir "$P" 2>&1 | grep -v "Warning\|warn" | grep -E "grid|rewritten|Error" || fail refine
  mark refine
fi

if ! done_ shift; then
  log "shift (08d --no-rows --apply)"
  "$PYN" scripts/08d_field_offset.py --play-dir "$P" --no-rows --apply --los-yards "$LOS" \
     2>&1 | grep -v "Warning\|warn" | grep -E "shift|scrimmage|rewritten" || fail shift
  mark shift
fi

if ! done_ endzone; then
  log "endzone re-solve in the field frame with the mirror check (08 --sideline-from), then export again"
  "$PYN" scripts/08_reconstruct_all22.py --sideline-from "$P" --root "$ROOT" --sideline "$SIDE" --endzone "$END" \
     --out "$P/recon_abs.npz" 2>&1 | grep -v "Warning\|warn" | grep -E "mount side|mirror|gap  |reconciled  |player height|Error|Exit" || fail endzone
  "$PYN" scripts/08b_export_play_dir.py --recon "$P/recon_abs.npz" --root "$ROOT" --sideline "$SIDE" --endzone "$END" --out "$P" \
     2>&1 | grep -v "Warning\|warn" | grep -E "cameras:|linked|tracks.parquet" || fail endzone-export
  rm -f "$P"/.done_pose_s "$P"/.done_pose_e "$P"/.done_identity "$P"/.done_keypoints "$P"/.done_tri "$P"/.done_fuse "$P"/.done_refit "$P"/.done_hifi "$P"/.done_render \
        "$P/poses_sideline.json" "$P/poses_endzone.json"
  mark endzone
fi

if ! done_ check; then
  log "rulers + line of scrimmage (08d)"
  out="$("$PYN" scripts/08d_field_offset.py --play-dir "$P" --los-yards "$LOS" 2>&1)" || { echo "$out" | tail -3; fail check; }
  echo "$out" | grep -E "by ruler|agree|DISAGREE|shift|scrimmage"
  echo "$out" | grep -q "DISAGREE" && fail "check: the hash and numeral rulers disagree on this calibration"
  echo "$out" | grep -q "MISMATCH" && fail "check: the formation is not at the play description's line of scrimmage"
  mark check
fi

if ! done_ field; then
  log "field texture from the footage (05l; LOOK at the PNG: paint must land on the drawn field)"
  "$PYN" scripts/05l_field_from_footage.py --play-dir "$P" --preview "$DIAG/${PLAY}_field_texture.png"      2>&1 | grep -v "Warning\|warn\|nanmedian" | grep -E "field texture|footage turf|Error" || fail field
  mark field
fi

if ! done_ pose_s; then
  log "pose sideline (05c, resumes per frame)"
  "$PYS" scripts/05c_pose_play.py --play-dir "$P" --cam sideline --out "$P/poses_sideline.json" \
     2>&1 | grep -v Warning | tail -1 || fail pose_s
  mark pose_s
fi

if ! done_ pose_e; then
  log "pose endzone (05c --match-frames)"
  "$PYS" scripts/05c_pose_play.py --play-dir "$P" --cam endzone --match-frames "$P/poses_sideline.json" \
     --out "$P/poses_endzone.json" 2>&1 | grep -v Warning | tail -1 || fail pose_e
  mark pose_e
fi

if ! done_ identity; then
  log "identity (08c)"
  "$PYN" scripts/08c_identity_all22.py --play-dir "$P" --week 1 2>&1 | grep -v "Warning\|warn" | tail -4 || fail identity
  mark identity
fi

if ! done_ keypoints; then
  log "2-D keypoints per tracked person in both views (05m, YOLOv8-pose)"
  "$PYN" scripts/05m_keypoints_2d.py --play-dir "$P" 2>&1 | grep -v "Warning\|warn" | grep -E "keypoints:|matched|Error" || fail keypoints
  mark keypoints
fi

if ! done_ tri; then
  log "joints triangulated from the keypoints with both cameras (05n)"
  "$PYS" scripts/05n_triangulate_keypoints.py --play-dir "$P" 2>&1 | grep -v "Warning\|warn" | grep -E "offset|triangulated|Error" || fail tri
  mark tri
fi

if [ "${FUSE:-0}" = "1" ] && ! done_ fuse; then
  log "fuse views (05e)"
  "$PYS" scripts/05e_fuse_views.py --play-dir "$P" --poses "$P/poses_sideline.json" "$P/poses_endzone.json" \
     --identity "$P/identity_resolved.pkl" --out "$P/poses_fused.json" 2>&1 | grep -v Warning | grep "median across" || fail fuse
  mark fuse
fi

if ! done_ refit; then
  log "refit SMPL-X to fused joints (05f)"
  "$PYS" scripts/05f_refit_fused.py --play-dir "$P" --fused "$P/poses_tri.json" --poses "$P/poses_sideline.json" \
     --identity "$P/identity_resolved.pkl" --out "$P/poses_refit.json" 2>&1 | grep -v Warning | grep "refit" || fail refit
  mark refit
fi

if [ "${FIT:-0}" = "1" ] && ! done_ fit; then
  log "appearance fit to the footage (05i; held-out L1 against the median texture)"
  "$PYS" scripts/05i_fit_appearance.py --play-dir "$P" --poses "$P/poses_refit.json" --out-dir "$P/appearance" \
     2>&1 | grep -v "Warning\|warn\|nanmedian\|med = " | grep -E "bodies|gain|saved|Error" || fail fit
  mark fit
fi

if ! done_ teams; then
  log "team per id from torso colour where bimodal (08f; refuses otherwise, identity teams stand)"
  "$PYN" scripts/08f_team_by_colour.py --play-dir "$P" --red "$RED" --white "$WHITE" 2>&1 | grep -v "Warning\|warn" | grep -E "split|refusing|wrote|Error" || true
  mark teams
fi

if ! done_ hifi; then
  log "hi-fi render on the footage field (05k; resumable)"
  "$PYS" scripts/05k_render_hifi.py --play-dir "$P" --out-dir "$P/render_hifi" --appearance "$P/appearance"      --field-texture "$P/field_texture.npz" --uniforms --numbers --helmets --follow --eye-offset 2 -26 10 --fov 50 2>&1 | grep -v "Warning\|warn" | grep -E "timeline:|field from|wrote|Error" || fail hifi
  mark hifi
fi

if ! done_ render; then
  log "render, world mode with the fitted appearance (05d)"
  "$PYS" scripts/05d_render_play.py --play-dir "$P" --poses "$P/poses_refit.json" --identity "$P/identity_resolved.pkl" \
     --fitted-appearance "$P/appearance" \
     --out-dir "$P/render_abs" 2>&1 | grep -v "Warning\|shape fit\|lb (beta" | grep -E "camera fixed|wrote|tracks survive|fitted appearance" || fail render
  mark render
fi

echo; echo "PLAY DONE $NAME"
