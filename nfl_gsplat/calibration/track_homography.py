"""Track the field ground-plane homography across a play (per-frame calibration).

Between consecutive PnP keyframe anchors we track the field->image homography
frame-to-frame on static field features (players masked), forward from the left
anchor and backward from the right anchor, then blend by normalized distance so
the chain snaps to both anchors and drift stays bounded. Each blended homography
is decomposed to (K, R, t). Confidence (RANSAC inlier ratio etc.) is gated:
a low-confidence run fails loud, asking for another keyframe.

Pure logic (blend / gap check / assemble) is unit-tested; the OpenCV optical-flow
+ RANSAC estimation is the `_estimate_interframe_homographies` seam.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.calibration.decompose_homography import homography_to_krt
from nfl_gsplat.errors import CalibrationError


@dataclass(frozen=True)
class TrackConfig:
    min_conf: float = 0.35
    max_gap: int = 0
    max_features: int = 800
    ransac_thresh_px: float = 3.0


def _norm(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2]


def blend_chains(forward: list[np.ndarray], backward: list[np.ndarray]) -> list[np.ndarray]:
    """Blend a forward-tracked and backward-tracked homography chain.

    ``forward[i]`` is tracked from the left anchor (i=0 is the left anchor),
    ``backward[i]`` from the right anchor (last index is the right anchor).
    Weight w = i/(n-1): w=0 -> all forward (left anchor), w=1 -> all backward.
    """
    n = len(forward)
    assert n == len(backward)
    out: list[np.ndarray] = []
    for i in range(n):
        w = 0.0 if n == 1 else i / (n - 1)
        H = (1.0 - w) * _norm(forward[i]) + w * _norm(backward[i])
        out.append(_norm(H))
    return out


def check_confidence(conf: np.ndarray, *, min_conf: float, max_gap: int) -> None:
    """Raise CalibrationError naming the first run of > max_gap frames below floor."""
    below = np.asarray(conf) < min_conf
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) > max_gap:
                raise CalibrationError(
                    f"camera tracking lost lock on frames {i}-{j - 1} "
                    f"(confidence < {min_conf}). Add a keyframe in that range and "
                    f"re-run scripts/02b_track_calibration.py. See SETUP.md §3."
                )
            i = j
        else:
            i += 1


def assemble_track(
    homographies: list[np.ndarray], conf: np.ndarray, *, width: int, height: int,
) -> CameraTrack:
    """Decompose each frame's homography into (K, R, t) -> CameraTrack."""
    T = len(homographies)
    K = np.zeros((T, 3, 3)); R = np.zeros((T, 3, 3)); t = np.zeros((T, 3))
    for i, H in enumerate(homographies):
        Ki, Ri, ti = homography_to_krt(H, width=width, height=height)
        K[i], R[i], t[i] = Ki, Ri, ti
    return CameraTrack(K=K, R=R, t=t, conf=np.asarray(conf, float),
                       width=width, height=height)


def _estimate_interframe_homographies(
    video_path, frame_a: int, frame_b: int, masks_provider, cfg: TrackConfig,
) -> tuple[list[np.ndarray], np.ndarray]:
    """OpenCV seam: relative homography mapping frame_a -> each frame in (a, b].

    Returns (rel list of 3x3 mapping the frame_a image to frame_k image, conf
    array). Implemented against cv2 (goodFeaturesToTrack on the field region
    outside masks_provider(k) boxes, calcOpticalFlowPyrLK, findHomography RANSAC)
    at real-video bring-up. Monkeypatched in tests.
    """
    raise NotImplementedError(
        "optical-flow homography estimation is finalized against real video at "
        "single-play bring-up; tests monkeypatch this seam."
    )


def track_camera_sequence(
    video_path,
    anchors: dict[int, np.ndarray],
    *,
    num_frames: int,
    width: int,
    height: int,
    masks_provider=lambda frame: [],
    cfg: TrackConfig = TrackConfig(),
) -> CameraTrack:
    """Build a per-frame CameraTrack from keyframe-anchor field->image homographies."""
    anchor_frames = sorted(anchors)
    if not anchor_frames:
        raise CalibrationError("no keyframe anchors provided")
    Hs: list[np.ndarray | None] = [None] * num_frames
    conf = np.zeros(num_frames)

    for f in range(0, anchor_frames[0]):
        Hs[f] = _norm(anchors[anchor_frames[0]]); conf[f] = 1.0
    for f in range(anchor_frames[-1], num_frames):
        Hs[f] = _norm(anchors[anchor_frames[-1]]); conf[f] = 1.0

    for a, b in zip(anchor_frames[:-1], anchor_frames[1:]):
        rel_fwd, c_fwd = _estimate_interframe_homographies(video_path, a, b, masks_provider, cfg)
        rel_bwd, c_bwd = _estimate_interframe_homographies(video_path, b, a, masks_provider, cfg)
        forward = [anchors[a]] + [Hrel @ anchors[a] for Hrel in rel_fwd]
        backward = list(reversed([anchors[b]] + [Hrel @ anchors[b] for Hrel in rel_bwd]))
        seg = blend_chains(forward, backward)
        seg_conf = np.minimum(np.r_[1.0, c_fwd], np.r_[list(reversed(c_bwd)), 1.0])
        for k, f in enumerate(range(a, b + 1)):
            Hs[f] = seg[k]; conf[f] = seg_conf[k]

    check_confidence(conf, min_conf=cfg.min_conf, max_gap=cfg.max_gap)
    return assemble_track([h for h in Hs], conf, width=width, height=height)
