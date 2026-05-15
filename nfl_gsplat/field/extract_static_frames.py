"""Sample frames from pre-snap (empty field) time ranges, per camera.

Used by the field reconstruction stage: nerfstudio ``splatfacto`` trains on a
small bag of empty-field frames so the static stadium geometry is not
corrupted by players. This module is the lightweight pre-step that takes a
long broadcast clip and writes the subset of frames that splatfacto will
consume.

Output layout::

    {out_dir}/frames/{cam}/{range_idx:02d}_{frame:06d}.png

The per-range prefix prevents filename collisions when multiple pre-snap
windows are sampled from the same camera.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.video import ffprobe_meta

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class StaticFrameConfig:
    fps_sample: float = 2.0
    max_frames_per_cam: int = 60


@dataclass(frozen=True)
class PreSnapRange:
    start_sec: float
    duration_sec: float


def _extract_range(
    video: Path,
    out_dir: Path,
    start_sec: float,
    duration_sec: float,
    fps_sample: float,
    name_prefix: str,
) -> list[Path]:
    """Run ffmpeg to extract frames in ``[start_sec, start_sec+duration_sec]``
    at ``fps_sample`` fps. Writes ``{out_dir}/{name_prefix}_{i:06d}.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{name_prefix}_%06d.png")
    # ``-ss`` before ``-i`` is fast seek (keyframe-accurate-enough for 2 fps sampling).
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec}",
        "-i", str(video),
        "-t", f"{duration_sec}",
        "-vf", f"fps={fps_sample}",
        "-start_number", "0",
        pattern,
    ]
    subprocess.check_call(cmd)
    return sorted(out_dir.glob(f"{name_prefix}_*.png"))


def extract_static_frames(
    videos: Mapping[str, Path | str],
    pre_snap_ranges: Iterable[PreSnapRange],
    out_dir: Path | str,
    cfg: StaticFrameConfig,
) -> dict[str, list[Path]]:
    """Extract pre-snap frames for each camera into ``out_dir/frames/{cam}``.

    ``videos`` maps camera name to source video path. The same ``pre_snap_ranges``
    are applied to every camera — broadcast feeds are synchronized, so the
    time windows are shared.

    Returns ``{cam: [frame_paths...]}`` sorted by filename.
    """
    out_dir = Path(out_dir)
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    ranges = list(pre_snap_ranges)
    if not ranges:
        raise ValueError("pre_snap_ranges is empty — nothing to sample")

    per_cam: dict[str, list[Path]] = {}
    for cam, video in videos.items():
        video = Path(video)
        if not video.exists():
            raise SetupError(
                f"video file missing for camera '{cam}': {video}. "
                "Place the broadcast clip at that path (see SETUP.md §5)."
            )
        meta = ffprobe_meta(video)
        clip_duration = meta.num_frames / meta.fps
        cam_dir = frames_root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for i, rng in enumerate(ranges):
            if rng.start_sec < 0 or rng.start_sec >= clip_duration:
                raise ValueError(
                    f"pre_snap_range[{i}] start={rng.start_sec}s is outside "
                    f"clip duration ({clip_duration:.2f}s) for {cam}"
                )
            duration = min(rng.duration_sec, clip_duration - rng.start_sec)
            name_prefix = f"r{i:02d}"
            paths = _extract_range(
                video, cam_dir, rng.start_sec, duration, cfg.fps_sample, name_prefix
            )
            written.extend(paths)

        if len(written) > cfg.max_frames_per_cam:
            # Even spacing — keep the first ``max_frames_per_cam`` at regular stride.
            stride = max(1, len(written) // cfg.max_frames_per_cam)
            keep = written[::stride][: cfg.max_frames_per_cam]
            drop = [p for p in written if p not in set(keep)]
            for p in drop:
                p.unlink()
            written = keep

        _LOG.info(f"extract_static_frames({cam}): {len(written)} frames "
                  f"across {len(ranges)} pre-snap window(s)")
        per_cam[cam] = written

    return per_cam
