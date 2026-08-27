"""Automatic per-frame field calibration → cameras.npz (headless, no display).

    python scripts/02_autocalibrate.py --play-dir data/2025/week_04/SEA_at_AZ/play_001

Detects + identifies field markings each frame and solves the camera per frame
(no manual annotation, no keyframes). Fails loud if a long run of frames can't be
registered. Replaces the manual 02_calibrate + 02b path (kept as fallback).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from nfl_gsplat.calibration.run_autocalib import (
    build_autocalib_npz, build_autocalib_npz_learned, build_autocalib_npz_pretrained,
)
from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
from nfl_gsplat.errors import SetupError
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.meta import load_meta

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


class CalibMode(str, Enum):
    hint = "hint"
    learned = "learned"
    pretrained = "pretrained"
    cross_endzone = "cross-endzone"
    identity_endzone = "identity-endzone"
    mosaic_endzone = "mosaic-endzone"



def _stored_reference_camera(cameras_npz, ref_frame: int, cam: str = "endzone"):
    """The endzone camera already stored at ``ref_frame``, as a CalibrationResult.

    Raises rather than returning a filled-from-a-neighbour pose: ``conf == 0``
    means the frame's camera was INTERPOLATED, not solved, and reusing one as
    the reference would silently anchor the whole play to a guess.
    """
    import numpy as np

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    cams = load_camera_track(cameras_npz)
    track = cams.get(cam)
    if track is None:
        raise SetupError(
            f"--reuse-reference: no {cam!r} camera in {cameras_npz}. There is "
            "nothing to reuse; run once without the flag to solve one.")
    if not 0 <= int(ref_frame) < len(track.conf):
        raise SetupError(
            f"--reuse-reference: frame {ref_frame} is outside the stored track "
            f"(0..{len(track.conf) - 1}).")
    if float(np.asarray(track.conf)[int(ref_frame)]) <= 0:
        raise SetupError(
            f"--reuse-reference: frame {ref_frame} has conf=0 in {cameras_npz}, "
            "which means its pose was filled in from a neighbour rather than "
            "solved. Pick a --ref-frame that verified.")
    intr, pose = track.at(int(ref_frame))
    return CalibrationResult(
        intrinsics=CameraIntrinsics(intr.fx, intr.fy, intr.cx, intr.cy,
                                    track.width, track.height),
        pose=CameraPose(R=pose.R, t=pose.t), rms_px=0.0,
        num_correspondences=0, refined_with_ba=False)


@app.command()
def main(play_dir: Path = typer.Option(..., "--play-dir"),
         mode: CalibMode = typer.Option(CalibMode.hint, "--mode",
                                        help="'hint' (default), 'learned' (requires --model-ckpt), "
                                             "or 'pretrained' (requires --roboflow-kps)."),
         model_ckpt: Optional[Path] = typer.Option(None, "--model-ckpt",
                                                    help="Path to LandmarkNet checkpoint (learned mode only)."),
         yard_min: float = typer.Option(-25.0, "--yard-min",
                                        help="World-X lower bound for landmark schema (learned mode)."),
         yard_max: float = typer.Option(25.0, "--yard-max",
                                        help="World-X upper bound for landmark schema (learned mode)."),
         roboflow_kps: Optional[Path] = typer.Option(None, "--roboflow-kps",
             help="Path to roboflow_kps.json (pretrained mode; from scripts/03_roboflow_precompute.py)."),
         territory: str = typer.Option("away", "--territory",
             help="Which side of the 50 the visible yard numbers belong to (pretrained mode)."),
         cameras: str = typer.Option("sideline,endzone", "--cameras",
             help="Comma-separated camera names present in the play dir (e.g. 'sideline')."),
         rotate: str = typer.Option("", "--rotate",
             help="Per-camera view rotation override, e.g. 'endzone=90' or "
                  "'endzone=90,sideline=0'. Default: endzone->90, others->0."),
         player_boxes: Optional[Path] = typer.Option(None, "--player-boxes",
             help="Path to tracks.parquet for player masking (pretrained mode; "
                  "default <play_dir>/tracks.parquet if present)."),
         play_dirs: Optional[str] = typer.Option(None, "--play-dirs",
             help="Comma-separated play dirs sharing one physically-fixed endzone "
                  "camera (identity-endzone mode); defaults to --play-dir alone."),
         stride: int = typer.Option(6, "--stride",
             help="Frame sampling stride for the accumulated mosaic (mosaic-endzone mode)."),
         ref_frame: int | None = typer.Option(None, "--ref-frame",
             help="Force the mosaic's reference frame index instead of auto-picking "
                  "the sampled frame with the largest field extent (mosaic-endzone "
                  "mode) — the operator's recovery lever if auto-pick clips yard "
                  "lines at the image border."),
         propagate_stride: int | None = typer.Option(None, "--propagate-stride",
             help="Frame stride for the OUTPUT camera track (mosaic-endzone "
                  "mode). Defaults to --stride. Set 1 for a camera on every "
                  "frame, which is what compositing needs; it adds coverage, "
                  "not accuracy, since a tripod's frames share one centre."),
         refine_stride: int | None = typer.Option(None, "--refine-stride",
             help="Enable the bundle-adjusted refinement pass (mosaic-endzone "
                  "mode) at this frame stride. Frames are re-solved jointly "
                  "against consecutive-frame homographies plus field anchors, "
                  "and every frame that cannot be VERIFIED against paint it can "
                  "see is left at conf=0 rather than shipped. Use a stride that "
                  "divides the splat's frame sampling so every exported frame "
                  "is a node. Trades coverage for correctness."),
         reuse_reference: bool = typer.Option(False, "--reuse-reference",
             help="Reuse the endzone reference camera already in cameras.npz at "
                  "--ref-frame instead of re-solving it from hash marks "
                  "(mosaic-endzone mode). The reference solve is a "
                  "once-per-game step and is the most environment-sensitive "
                  "stage in the pipeline; reusing a validated camera is the "
                  "normal path for re-running a play. Requires --ref-frame."),
         max_gap: int | None = typer.Option(None, "--max-gap",
             help="Longest run of consecutive uncalibrated frames tolerated "
                  "inside the mosaic-endzone track (default: 3x --stride). "
                  "Raise it only when the uncovered span is understood: a "
                  "fast pan can break the pure-rotation model outright."),
         diag_dir: str | None = typer.Option(None, "--diag-dir",
             help="Directory for the first-run mosaic diagnostic PNG (mosaic-endzone "
                  "mode); default <play_dir>."),
         config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
    load_cli_config(config, config_override, set_)
    rotations = {}
    for pair in (p.strip() for p in rotate.split(",") if p.strip()):
        cam_name, _, deg_s = pair.partition("=")
        if deg_s not in ("0", "90", "180", "270"):
            raise typer.BadParameter(
                f"--rotate {pair!r}: rotation must be 0/90/180/270.")
        rotations[cam_name.strip()] = int(deg_s)
    pd = PlayDir.from_dir(play_dir, cameras=tuple(c.strip() for c in cameras.split(",") if c.strip()))
    meta = load_meta(pd.meta_yaml)
    videos = {cam: pd.video(cam) for cam in pd.cameras}

    if mode is CalibMode.identity_endzone:
        from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
        from nfl_gsplat.calibration.run_autocalib import build_endzone_identity_from_plays
        ep = meta.endzone_prior
        if ep is None:
            raise SetupError(
                f"{pd.meta_yaml}: --mode identity-endzone requires an `endzone_prior:` "
                "block (x_range/y_range/z_range/focal_range). See SETUP.md §3."
            )
        prior = EndzonePrior(tuple(ep["x_range"]), tuple(ep["y_range"]),
                             tuple(ep["z_range"]), tuple(ep["focal_range"]))
        dirs = [PlayDir.from_dir(Path(p.strip()), cameras=pd.cameras).dir
                for p in play_dirs.split(",") if p.strip()] if play_dirs else [pd.dir]
        out = build_endzone_identity_from_plays(play_dirs=dirs, prior=prior, fps=meta.fps)
    elif mode is CalibMode.mosaic_endzone:
        from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
        from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
        ep = meta.endzone_prior
        if ep is None:
            raise SetupError(
                f"{pd.meta_yaml}: --mode mosaic-endzone needs an 'endzone_prior:' "
                "block (x_range/y_range/z_range/focal_range). See SETUP.md §3."
            )
        ea = meta.endzone_anchor
        anchors = tuple(((float(a["point_px"][0]), float(a["point_px"][1])),
                         float(a["world_x_m"])) for a in ea["lines"]) if ea else None
        reference_camera = None
        if reuse_reference:
            if ref_frame is None:
                raise SetupError(
                    "--reuse-reference needs --ref-frame: it names the frame "
                    "whose stored camera is reused, and the auto-picked "
                    "reference is not recorded in cameras.npz.")
            reference_camera = _stored_reference_camera(
                pd.dir / "cameras.npz", ref_frame)
        out = build_endzone_mosaic(
            play_dir=pd.dir, tracks_path=pd.dir / "tracks.parquet",
            cameras_npz=pd.dir / "cameras.npz", endzone_video=pd.video("endzone"),
            fps=meta.fps, anchors=anchors, stride=stride, ref_frame=ref_frame,
            max_gap=max_gap, propagate_stride=propagate_stride,
            refine_stride=refine_stride, diag_dir=diag_dir,
            reference_camera=reference_camera,
            prior=EndzonePrior(tuple(ep["x_range"]), tuple(ep["y_range"]),
                               tuple(ep["z_range"]), tuple(ep["focal_range"])))
    elif mode is CalibMode.cross_endzone:
        from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
        out = build_endzone_from_sideline(
            play_dir=pd.dir, tracks_path=pd.dir / "tracks.parquet",
            cameras_npz=pd.dir / "cameras.npz", endzone_video=pd.video("endzone"),
            fps=meta.fps, rotations=rotations or None)
    elif mode is CalibMode.pretrained:
        # Per-camera keypoint caches: --roboflow-kps names a single-camera
        # cache explicitly; otherwise each camera uses the convention
        # <play_dir>/roboflow_kps_{cam}.json (falling back to the legacy
        # <play_dir>/roboflow_kps.json for a single camera).
        if roboflow_kps is not None:
            if len(videos) != 1:
                raise typer.BadParameter(
                    "--roboflow-kps only applies to a single camera; with "
                    "multiple cameras place roboflow_kps_{cam}.json files "
                    "in the play dir.")
            kps_json = {next(iter(videos)): roboflow_kps}
        else:
            kps_json = {}
            for cam in videos:
                p = pd.dir / f"roboflow_kps_{cam}.json"
                if not p.exists() and len(videos) == 1:
                    p = pd.dir / "roboflow_kps.json"
                kps_json[cam] = p
        from nfl_gsplat.calibration.player_masks import boxes_provider_from_tracks
        boxes_path = player_boxes if player_boxes is not None else (pd.dir / "tracks.parquet")
        if boxes_path.exists():
            masks_provider = boxes_provider_from_tracks(boxes_path)
        else:
            _LOG.warning("no player tracks at %s — running calibration UNMASKED "
                         "(endzone likely fails; run scripts/03b_detect_players.py)",
                         boxes_path)
            masks_provider = None
        out = build_autocalib_npz_pretrained(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            kps_json=kps_json, territory=territory,
            rotations=rotations or None, masks_provider=masks_provider,
        )
    elif mode is CalibMode.learned:
        if model_ckpt is None:
            raise typer.BadParameter("--model-ckpt is required in learned mode.")
        # TODO(bring-up): per-game model_ckpt + yard window in meta.yaml
        out = build_autocalib_npz_learned(
            play_dir=pd.dir, videos=videos, fps=meta.fps,
            model_ckpt=model_ckpt, yard_min=yard_min, yard_max=yard_max,
        )
    else:
        # TODO(bring-up): wire tracks.parquet player boxes via masks_provider to de-clutter lines
        out = build_autocalib_npz(
            play_dir=pd.dir, videos=videos, fps=meta.fps, hints=meta.calib_hints,
        )
    _LOG.info(f"wrote automatic per-frame calibration → {out}")


if __name__ == "__main__":
    app()
