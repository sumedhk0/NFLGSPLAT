"""Cache SMPL-X parameters for every tracked player across a range of frames.

Inference is the expensive stage -- a 0.69B network over thousands of crops --
so it is done ONCE and written to disk. Everything after it (shape fitting,
placement, compositing, rendering, smoothing) is cheap and can be re-run against
the cache without touching the GPU again.

Only frames whose camera VERIFIED are used. A frame with ``conf == 0`` has a
pose filled in from a neighbour, so placing a body through it would inherit a
guess that nothing downstream could detect.

Runs in the SMPL-X env (.venv-smplx).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

CROP_W, CROP_H = 192, 256
MIN_BOX_W, MIN_BOX_H = 16, 32
# Frames per inference batch. Crops from several frames are pushed through
# together: the model's fixed overhead per call dominates at 20 crops.
FRAMES_PER_BATCH = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--stride", type=int, default=6,
                    help="sample every Nth VERIFIED frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--match-frames", type=Path, default=None,
                    help="restrict to frames already present in another pose "
                         "cache, so the two cameras can be fused")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import cv2
    import pandas as pd

    from nfl_gsplat.pose.smplestx_infer import SMPLestXConfig, infer_crops
    from nfl_gsplat.utils.video import iter_frames

    play = args.play_dir
    track = load_camera_track(play / "cameras.npz")[args.cam]
    conf = np.asarray(track.conf)
    verified = [int(f) for f in np.flatnonzero(conf > 0)
                if args.start <= f < args.end]
    if args.match_frames is not None:
        # Fusing two views needs the SAME instants in both, and the two cameras
        # verify on different frames -- 816 sideline against 473 endzone, with
        # 364 in common. Striding each feed independently would land on almost
        # no shared frames, so the second camera follows the first's list.
        other = pickle.load(open(args.match_frames, "rb"))["frames"]
        wanted = set(verified) & set(int(f) for f in other)
        _LOG.info("%d verified frames; %d also posed in %s",
                  len(verified), len(wanted), args.match_frames.name)
    else:
        wanted = set(verified[::max(1, args.stride)])
        _LOG.info("%d verified frames in range; posing %d of them at stride %d",
                  len(verified), len(wanted), args.stride)
    if not wanted:
        raise SetupError(
            f"no {args.cam!r} frames to pose in [{args.start}, {args.end}). "
            f"Verified frames run "
            f"{int(np.flatnonzero(conf > 0).min())}..{int(np.flatnonzero(conf > 0).max())}"
            + (f", and none of them appear in {args.match_frames}."
               if args.match_frames is not None
               else f" and stride is {args.stride}."))

    df = pd.read_parquet(play / "tracks.parquet")
    df = df[(df["cam"] == args.cam) & (df["track_id"] >= 0)
            & (df["frame"].isin(wanted))]

    cfg = SMPLestXConfig()
    cache: dict[int, dict[int, dict]] = {}
    pending_crops, pending_meta = [], []
    t0 = time.time()
    n_done = 0

    def _flush():
        nonlocal pending_crops, pending_meta
        if not pending_crops:
            return
        out = infer_crops(np.asarray(pending_crops, np.uint8),
                          np.asarray([m["bbox"] for m in pending_meta], float),
                          cfg)
        for i, meta in enumerate(pending_meta):
            cache.setdefault(meta["frame"], {})[meta["track_id"]] = {
                "betas": np.asarray(out["betas"][i], np.float32),
                "body_pose": np.asarray(out["body_pose"][i], np.float32),
                "global_orient": np.asarray(out["global_orient"][i], np.float32),
                "transl": np.asarray(out["transl"][i], np.float32),
                "bbox": np.asarray(meta["bbox"], np.float32),
                # The 2D joints, in ORIGINAL-frame pixels. Cached because
                # placement from a single pixel -- the box's bottom-centre --
                # throws away ~22 observations per view, and it is depth that
                # a foot point cannot carry: measured on play_001, 2.2 m of the
                # 2.9 m cross-camera error lies along the viewing rays, which
                # is exactly what triangulating two views constrains.
                "joints2d": np.asarray(out["joints2d"][i], np.float32),
                "joints3d_cam": np.asarray(out["joints3d_cam"][i], np.float32),
            }
        pending_crops, pending_meta = [], []

    frames_in_batch = 0
    for idx, rgb in iter_frames(play / f"{args.cam}.mp4", stride=1):
        if idx not in wanted:
            continue
        h_img, w_img = rgb.shape[:2]
        for r in df[df["frame"] == idx].itertuples():
            x1, y1 = max(0, int(r.bbox_x1)), max(0, int(r.bbox_y1))
            x2, y2 = min(w_img, int(r.bbox_x2)), min(h_img, int(r.bbox_y2))
            if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                continue
            pending_crops.append(cv2.resize(rgb[y1:y2, x1:x2], (CROP_W, CROP_H)))
            pending_meta.append({"frame": idx, "track_id": int(r.track_id),
                                 "bbox": [x1, y1, x2, y2]})
        frames_in_batch += 1
        n_done += 1
        if frames_in_batch >= FRAMES_PER_BATCH:
            _flush()
            frames_in_batch = 0
            rate = n_done / max(1e-6, time.time() - t0)
            _LOG.info("posed %d/%d frames (%.2f frames/s, %.0f s left)",
                      n_done, len(wanted), rate,
                      (len(wanted) - n_done) / max(1e-6, rate))
            # Checkpoint as we go: this run is long, and losing an hour of GPU
            # time to an interruption is avoidable.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "wb") as fh:
                pickle.dump({"cam": args.cam, "stride": args.stride,
                             "frames": cache}, fh)
    _flush()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"cam": args.cam, "stride": args.stride, "frames": cache}, fh)
    n_poses = sum(len(v) for v in cache.values())
    _LOG.info("wrote %s: %d frames, %d poses (%.0f s)", args.out, len(cache),
              n_poses, time.time() - t0)


if __name__ == "__main__":
    main()
