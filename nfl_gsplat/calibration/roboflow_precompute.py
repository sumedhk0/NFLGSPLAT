"""Core of the roboflow precompute step (script-independent, testable).

The CLI (scripts/03_roboflow_precompute.py) wires the video reader and the
hosted-inference HTTP client; this module only filters and caches.
"""
from __future__ import annotations

from nfl_gsplat.calibration.roboflow_kps import ModelKeypoint, write_kps_json


def run_precompute(frames_iter, *, infer_fn, model_id: str, video_name: str,
                   num_frames: int, out_json, kp_conf: float = 0.5) -> int:
    """Run ``infer_fn`` over frames, filter, write the JSON cache.

    Drops sideline classes ('*-sl', hallucinated) and keypoints below
    ``kp_conf``. Returns the number of frames with >=1 kept keypoint.
    """
    frames: dict[int, list[ModelKeypoint]] = {}
    n_hit = 0
    for idx, bgr in frames_iter:
        kept = [(n, u, v, c) for (n, u, v, c) in infer_fn(bgr)
                if c >= kp_conf and not n.endswith("-sl")]
        frames[idx] = kept
        if kept:
            n_hit += 1
    write_kps_json(out_json, model_id=model_id, video_name=video_name,
                   num_frames=num_frames, kp_conf=kp_conf, frames=frames)
    return n_hit
