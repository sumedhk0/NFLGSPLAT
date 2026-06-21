"""ffprobe metadata, frame extraction, mp4 encode — all via imageio/ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    num_frames: int


def ffprobe_meta(path: Path | str) -> VideoMeta:
    """Return width/height/fps/num_frames for a video file.

    Uses ``ffprobe`` from PATH. Raises FileNotFoundError if ffprobe is absent.

    Parsed by KEY, not column position: ffprobe emits the requested ``stream``
    entries in the stream's natural order, not the order they were asked for
    (e.g. ``duration`` before ``nb_frames``), so positional parsing silently
    swaps fields. ``num_frames`` prefers the container's ``nb_frames`` when it's
    a positive integer and otherwise falls back to ``round(duration * fps)``.
    """
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError(
            "ffprobe not found on PATH — install ffmpeg (conda-forge::ffmpeg). "
            "See SETUP.md §1."
        )
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "default=noprint_wrappers=1", str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            fields[key.strip()] = val.strip()

    width = int(fields["width"])
    height = int(fields["height"])
    num, den = fields["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    nb_frames = fields.get("nb_frames", "")
    if nb_frames.isdigit() and int(nb_frames) > 0:
        num_frames = int(nb_frames)
    else:  # nb_frames missing/"N/A" — derive from duration
        num_frames = int(round(float(fields["duration"]) * fps))
    return VideoMeta(width=width, height=height, fps=fps, num_frames=num_frames)


def extract_frames(
    video: Path | str,
    out_dir: Path | str,
    *,
    fps: float | None = None,
    start_sec: float = 0.0,
    num_sec: float | None = None,
) -> list[Path]:
    """Extract frames as PNGs. Returns sorted list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec}", "-i", str(video)]
    if num_sec is not None:
        cmd += ["-t", f"{num_sec}"]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-start_number", "0", str(out_dir / "%06d.png")]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(out_dir.glob("*.png"))


def encode_mp4(
    frames_dir: Path | str,
    out_path: Path | str,
    *,
    fps: float = 30.0,
    codec: str = "libx264",
    crf: int = 18,
) -> Path:
    """Encode a directory of zero-padded PNGs to an MP4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", f"{fps}",
        "-i", str(Path(frames_dir) / "%06d.png"),
        "-c:v", codec, "-crf", f"{crf}", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def iter_frames(video: Path | str, start_frame: int = 0, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, HWC uint8 RGB) using imageio. Lazy and memory-bounded."""
    import imageio.v3 as iio

    for i, frame in enumerate(iio.imiter(str(video), plugin="pyav")):
        if i < start_frame:
            continue
        if (i - start_frame) % stride != 0:
            continue
        yield i, np.asarray(frame)
