"""Pose stage CLI: per-instance SMPL-X motion → ``poses/{instance_id}.npz``.

For each renderable instance in ``entities.json`` this:

1. crops the player in both cameras across the play window and runs SMPLest-X
   (the env-gated seam — needs ``nfl_smplx`` + weights);
2. triangulates the body joints to 3D (:mod:`pose.triangulate`);
3. refits SMPL-X per frame against those 3D joints (:func:`fuse_smplx.fuse_sequence`),
   using a forward-kinematics forward so the fit matches what the renderer animates;
4. fills short gaps + 1€-smooths the parameter streams;
5. converts to per-joint LBS transforms ``joint_tfms[T, J, 4, 4]`` and writes them.

Betas are frozen to the library when present (so the cached avatar's rig and the
per-play skeleton share bone lengths); the player's best reference crop + betas
are written to ``data/{season}/_library/_refs/{uid}.npz`` for the avatar-build stage.

Steps 2-5 are pure numpy/scipy and unit-tested in ``tests/test_run_pose.py``;
step 1 is the GPU seam, exercised on PACE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from nfl_gsplat.pose.forward_kinematics import (
    SMPLX_BODY_PARENTS,
    fk_forward,
    joint_tfms_sequence,
)
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, fuse_sequence
from nfl_gsplat.pose.temporal_smooth import (
    OneEuroConfig,
    interpolate_short_gaps,
    smooth_param_sequence,
)
from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.pose.triangulate import TriangulationConfig, triangulate_joints_two_view
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

NUM_BODY_JOINTS = 22


def _fill_remaining(arr: np.ndarray) -> np.ndarray:
    """Forward/back-fill NaN frames so long gaps don't poison the FK transforms.

    ``interpolate_short_gaps`` leaves long gaps as NaN; carry the nearest valid
    frame across them (and zero-fill a fully-empty channel) so every emitted
    ``joint_tfms`` frame is finite.
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    T = out.shape[0]
    last: np.ndarray | None = None
    for t in range(T):
        if np.isfinite(out[t]).all():
            last = out[t].copy()
        elif last is not None:
            out[t] = last
    # Back-fill any leading NaNs from the first valid frame; else zeros.
    nxt: np.ndarray | None = None
    for t in range(T - 1, -1, -1):
        if np.isfinite(out[t]).all():
            nxt = out[t].copy()
        elif nxt is not None:
            out[t] = nxt
    return np.nan_to_num(out, nan=0.0)


def solve_joint_tfms(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    cameras: Mapping[str, CameraTrack],
    rest_joints: np.ndarray,                # [J, 3] (encodes frozen betas)
    parents: tuple[int, ...] = SMPLX_BODY_PARENTS,
    *,
    tri_cfg: TriangulationConfig | None = None,
    fit_cfg: SMPLXFitConfig | None = None,
    euro_cfg: OneEuroConfig | None = None,
    gap_frames: int = 5,
) -> np.ndarray:
    """Triangulate → refit → smooth → FK. Returns ``joint_tfms[T, J, 4, 4]``.

    Pure (no video / torch); the SMPLest-X ``observations`` are produced by the
    env-gated seam. ``observations`` maps ``cam -> {"uv":[T,J,2], "conf":[T,J]}``
    over the body joints.
    """
    tri_cfg = tri_cfg or TriangulationConfig()
    fit_cfg = fit_cfg or SMPLXFitConfig()
    euro_cfg = euro_cfg or OneEuroConfig()

    tri = triangulate_joints_two_view(observations, cameras, tri_cfg)
    joints3d, valid = tri.joints3d, tri.valid

    forward = fk_forward(
        rest_joints, parents,
        body_pose_dim=fit_cfg.body_pose_dim,
        global_orient_dim=fit_cfg.global_orient_dim,
        transl_dim=fit_cfg.transl_dim,
    )
    dim = fit_cfg.body_pose_dim + fit_cfg.global_orient_dim + fit_cfg.transl_dim
    init = np.zeros(dim, dtype=np.float64)
    # Warm-start translation from the first frame with a valid pelvis (joint 0).
    pelvis_ok = np.where(valid[:, 0])[0]
    if pelvis_ok.size:
        init[fit_cfg.body_pose_dim + fit_cfg.global_orient_dim:] = joints3d[pelvis_ok[0], 0]

    fit = fuse_sequence(joints3d, valid, init, forward, fit_cfg)

    streams = []
    for raw in (fit.body_pose, fit.global_orient, fit.transl):
        filled, _ = interpolate_short_gaps(raw, fit.valid_frames, max_gap=gap_frames)
        smoothed = smooth_param_sequence(filled, euro_cfg)
        streams.append(_fill_remaining(smoothed))
    body_pose_s, global_orient_s, transl_s = streams

    return joint_tfms_sequence(global_orient_s, body_pose_s, transl_s, rest_joints, parents)


# --- env-gated extraction seam (video + SMPLest-X) -------------------------

def extract_observations(
    instance_id: int,
    tracks_df,
    cameras: Mapping[str, CameraTrack],
    video_paths: Mapping[str, Path | str],
    window_start: int,
    window_end: int,
    smplestx_cfg,
    *,
    id_col: str = "global_player_id",
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray | None, np.ndarray | None]:
    """Crop the instance in each camera over the window, run SMPLest-X, and
    assemble ``{cam: {"uv":[T,J,2], "conf":[T,J]}}`` over the body joints.

    Also returns ``(best_crop, betas)`` for the player's library reference (the
    largest-bbox crop and its estimated shape). Env-gated: reads video frames
    and calls the SMPLest-X adapter (``nfl_smplx``).
    """
    from nfl_gsplat.pose.smplestx_infer import infer_crops
    from nfl_gsplat.utils.video import iter_frames

    T = window_end - window_start + 1
    obs: dict[str, dict[str, np.ndarray]] = {}
    best_crop: np.ndarray | None = None
    best_area = -1.0
    best_betas: np.ndarray | None = None

    for cam in cameras:
        uv = np.zeros((T, NUM_BODY_JOINTS, 2), dtype=np.float64)
        conf = np.zeros((T, NUM_BODY_JOINTS), dtype=np.float64)
        rows = tracks_df[(tracks_df["cam"] == cam) & (tracks_df[id_col] == instance_id)]
        by_frame = {int(r["frame"]): r for _, r in rows.iterrows()}
        video = video_paths[cam]
        crops, boxes, frame_slots = [], [], []
        for fidx, frame in iter_frames(video, start_frame=window_start):
            if fidx > window_end:
                break
            r = by_frame.get(fidx)
            if r is None:
                continue
            x1, y1, x2, y2 = (int(r["bbox_x1"]), int(r["bbox_y1"]),
                              int(r["bbox_x2"]), int(r["bbox_y2"]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append(frame[y1:y2, x1:x2])
            boxes.append([x1, y1, x2, y2])
            frame_slots.append(fidx - window_start)
        if not crops:
            obs[cam] = {"uv": uv, "conf": conf}
            continue
        sizes = [(c.shape[0], c.shape[1]) for c in crops]
        h = max(s[0] for s in sizes)
        w = max(s[1] for s in sizes)
        batch = np.zeros((len(crops), h, w, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            batch[i, : c.shape[0], : c.shape[1]] = c[..., :3]
        out = infer_crops(batch, np.asarray(boxes, dtype=np.float64), smplestx_cfg)
        for i, slot in enumerate(frame_slots):
            uv[slot] = out["joints2d"][i, :NUM_BODY_JOINTS]
            conf[slot] = out["confidence"][i, :NUM_BODY_JOINTS]
            area = float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
            if area > best_area:
                best_area = area
                best_crop = crops[i]
                best_betas = np.asarray(out["betas"][i], dtype=np.float32)
        obs[cam] = {"uv": uv, "conf": conf}

    return obs, best_crop, best_betas


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    from pathlib import Path

    import pandas as pd
    import typer

    from nfl_gsplat.avatars.build_one import reference_path
    from nfl_gsplat.avatars.library import AvatarLibrary
    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.identity.registry import REFEREE_UID
    from nfl_gsplat.paths import PlayDir
    from nfl_gsplat.pose.forward_kinematics import load_smplx_skeleton
    from nfl_gsplat.pose.fuse_smplx import resolve_betas
    from nfl_gsplat.pose.smplestx_infer import SMPLestXConfig
    from nfl_gsplat.utils.io import read_json, write_npz
    from nfl_gsplat.utils.meta import load_meta
    from nfl_gsplat.utils.video import ffprobe_meta

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(play_dir: Path = typer.Option(..., "--play-dir"),
             config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pdir = PlayDir.from_dir(play_dir)
        meta = load_meta(pdir.meta_yaml)
        cameras = load_camera_track(pdir.cameras_npz)
        first_cam = next(iter(cameras))
        n_frames = ffprobe_meta(pdir.video(first_cam)).num_frames
        tracks = pd.read_parquet(pdir.tracks)
        entities = read_json(pdir.entities)
        video_paths = {cam: pdir.video(cam) for cam in cameras}

        body_dir = Path(str(cfg.paths.body_models))
        gender = str(cfg.pose.smplx_gender)
        tri_cfg = TriangulationConfig(reproj_px_max=float(cfg.pose.reproj_reject_px),
                                      conf_min=float(cfg.pose.smplestx_conf_reject))
        fit_cfg = SMPLXFitConfig(use_library_betas=bool(cfg.avatars.library.enabled))
        euro = OneEuroConfig(min_cutoff=float(cfg.pose.smoother.min_cutoff),
                             beta=float(cfg.pose.smoother.beta),
                             d_cutoff=float(cfg.pose.smoother.d_cutoff), fps=meta.fps)
        smplestx_cfg = SMPLestXConfig(device=str(cfg.pose.get("device", "cuda:0")))
        library = AvatarLibrary(root=pdir.library_root, season="")

        pdir.poses_dir.mkdir(parents=True, exist_ok=True)
        for ent in entities:
            iid = int(ent["instance_id"])
            uid = ent["player_uid"]
            obs, crop, est_betas = extract_observations(
                iid, tracks, cameras, video_paths,
                0, n_frames - 1, smplestx_cfg, id_col="global_player_id",
            )
            betas, _ = resolve_betas(
                library.get_betas(uid),
                lambda eb=est_betas: eb if eb is not None else np.zeros(10, np.float32),
                fit_cfg,
            )
            rest_joints, parents = load_smplx_skeleton(body_dir, gender, betas)
            tfms = solve_joint_tfms(obs, cameras, rest_joints, parents,
                                    tri_cfg=tri_cfg, fit_cfg=fit_cfg, euro_cfg=euro,
                                    gap_frames=int(cfg.pose.gap_interpolation_frames))
            write_npz(pdir.pose(str(iid)), joint_tfms=tfms.astype(np.float32))
            # Stash the player's best reference for the avatar-build stage.
            if uid != REFEREE_UID and crop is not None:
                write_npz(reference_path(pdir.library_root, "", uid),
                          crop=crop.astype(np.uint8), betas=np.asarray(betas, np.float32))
            _LOG.info(f"pose: instance {iid} ({uid}) → {pdir.pose(str(iid))}")

    app()


if __name__ == "__main__":
    _main()
