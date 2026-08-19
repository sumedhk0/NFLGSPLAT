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

from dataclasses import dataclass
from pathlib import Path

import cv2
from typing import Iterable, Mapping

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.video import ffprobe_meta, iter_frames

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
    keep_frames: "set[int] | None" = None,
) -> list[Path]:
    """Extract frames in ``[start_sec, start_sec+duration_sec]`` at
    ``fps_sample`` fps, named by their TRUE SOURCE FRAME INDEX as
    ``{out_dir}/{name_prefix}_{source_frame:06d}.png``.

    The name carries the source index because build_transforms pairs each image
    with a camera pose by parsing that number and calling ``track.at()`` on it.
    This used to shell out to ffmpeg with ``-vf fps=`` and ``-start_number 0``,
    which numbers the OUTPUT sequentially: on a 59.94 fps clip sampled at 2 fps,
    ``..._000010.png`` is source frame ~314, but the pose lookup used 10.
    Every image in transforms.json was therefore paired with a camera from a
    completely different moment of the play (verified: that file matches source
    frame 314 with mean abs difference 0.00, and frame 10 with 22.29).

    Decoding here rather than in ffmpeg also sidesteps ``-ss`` fast seek, whose
    keyframe rounding is what made the true index 314 instead of the nominal
    300 -- there is no arithmetic that recovers it, so the index has to come
    from the decoder that actually produced the frame.

    With ``keep_frames``, only those source indices are written, and the fps
    step is ignored. Sampling on a clock keeps whatever frames the clock lands
    on, which is unrelated to which frames have a camera worth training against:
    on play_001 a 2 fps clock kept 44 endzone frames of which 16 were verified,
    while 103 verified cameras existed in the same play."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = ffprobe_meta(video)
    step = max(1, int(round(meta.fps / float(fps_sample))))
    first = int(round(start_sec * meta.fps))
    last = min(meta.num_frames - 1,
               int(round((start_sec + duration_sec) * meta.fps)))

    written: list[Path] = []
    for idx, rgb in iter_frames(video, stride=1):
        if idx < first:
            continue
        if idx > last:
            break
        if keep_frames is not None:
            if idx not in keep_frames:
                continue
        elif (idx - first) % step:
            continue
        path = out_dir / f"{name_prefix}_{idx:06d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        written.append(path)
    return sorted(written)


def extract_static_frames(
    videos: Mapping[str, Path | str],
    pre_snap_ranges: Iterable[PreSnapRange],
    out_dir: Path | str,
    cfg: StaticFrameConfig,
    keep_frames: "Mapping[str, set[int]] | None" = None,
) -> dict[str, list[Path]]:
    """Extract pre-snap frames for each camera into ``out_dir/frames/{cam}``.

    ``videos`` maps camera name to source video path. The same ``pre_snap_ranges``
    are applied to every camera — broadcast feeds are synchronized, so the
    time windows are shared.

    ``keep_frames[cam]`` restricts extraction to those source frame indices --
    normally the frames whose camera VERIFIED, so the per-camera budget is spent
    on views that can actually be trained against instead of on whatever a fixed
    sampling clock happened to land on.

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
        # Clear this camera's previous extraction. Frames are named by SOURCE
        # INDEX, so a re-run with different sampling leaves the old images in
        # place under different names and build_transforms picks up the union of
        # two samplings -- including frames the current calibration never
        # verified. The directory is this function's own output, nothing else
        # writes here.
        stale = sorted(cam_dir.glob("*.png"))
        for old_png in stale:
            old_png.unlink()
        if stale:
            _LOG.info("extract_static_frames(%s): cleared %d frame(s) from a "
                      "previous extraction", cam, len(stale))

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
                video, cam_dir, rng.start_sec, duration, cfg.fps_sample,
                name_prefix,
                keep_frames=None if keep_frames is None else set(
                    keep_frames.get(cam, ())),
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

        if keep_frames is not None and not written:
            raise SetupError(
                f"extract_static_frames({cam}): none of the "
                f"{len(keep_frames.get(cam, ()))} verified frame(s) fall inside "
                "the pre-snap window(s). Widen the windows in meta.yaml, or "
                "re-run the calibration so it verifies frames in them.")
        _LOG.info("extract_static_frames(%s): %d frames across %d pre-snap "
                  "window(s)%s", cam, len(written), len(ranges),
                  "" if keep_frames is None else " (verified only)")
        per_cam[cam] = written

    return per_cam


def _main() -> None:  # pragma: no cover - thin CLI wiring, exercised on PACE
    import typer

    from nfl_gsplat.cli import CONFIG_OPT, CONFIG_OVERRIDE_OPT, SET_OPT, load_cli_config
    from nfl_gsplat.paths import PlayDir

    app = typer.Typer(add_completion=False)

    @app.command()
    def main(
        play_dir: Path = typer.Option(..., "--play-dir"),
        verified_only: bool = typer.Option(
            True, "--verified-only/--all-frames",
            help="Extract only frames whose camera VERIFIED in cameras.npz "
                 "(conf > 0). A fixed sampling clock lands on frames unrelated "
                 "to which ones have a trustworthy camera; on play_001 it kept "
                 "16 usable endzone frames while 103 verified cameras existed. "
                 "--all-frames restores clock sampling."),
        config=CONFIG_OPT, config_override=CONFIG_OVERRIDE_OPT, set_=SET_OPT,
    ) -> None:
        cfg = load_cli_config(config, config_override, set_)
        pdir = PlayDir.from_dir(play_dir)

        # Determine clip duration from the first camera video; use the full clip
        # as the single pre-snap sampling window (per-play clips are already
        # trimmed to the pre-snap period at ingest).
        first_video = pdir.video(pdir.cameras[0])
        meta = ffprobe_meta(first_video)
        clip_duration = meta.num_frames / meta.fps
        ranges = [PreSnapRange(start_sec=0.0, duration_sec=clip_duration)]

        videos = {cam: pdir.video(cam) for cam in pdir.cameras}
        out_dir = pdir.dir / "field"
        frame_cfg = StaticFrameConfig(
            fps_sample=float(cfg.field.fps_sample),
            max_frames_per_cam=int(cfg.field.pre_snap_frames_per_cam),
        )
        keep = None
        if verified_only:
            import numpy as np

            from nfl_gsplat.calibration.cameras_io import load_camera_track
            tracks = load_camera_track(pdir.cameras_npz)
            keep = {cam: set(np.flatnonzero(tr.conf > 0).tolist())
                    for cam, tr in tracks.items()}
            _LOG.info("extract_static_frames: verified cameras per feed: %s",
                      {c: len(v) for c, v in sorted(keep.items())})
        per_cam = extract_static_frames(videos, ranges, out_dir, frame_cfg,
                                        keep_frames=keep)
        total = sum(len(v) for v in per_cam.values())
        _LOG.info(f"extract_static_frames: {total} frames total → {out_dir / 'frames'}")

    app()


if __name__ == "__main__":
    _main()
