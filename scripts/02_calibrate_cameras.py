"""Annotate landmark keyframes for each camera of a play.

Usage::

    # Annotate frame 0 only (default single keyframe):
    python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001

    # Annotate multiple keyframe anchors:
    python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001 \\
        --keyframe 0 --keyframe 300 --keyframe 600

    # Annotate only one camera:
    python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001 \\
        --camera sideline --keyframe 0 --keyframe 150

Writes ``{play_dir}/{cam}_keyframes.json`` for each camera. The per-frame PnP
solve is performed by ``scripts/02b_track_calibration.py``.

Respects the pipeline's fail-loud error philosophy: missing videos raise
:class:`SetupError` with the exact next command to run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import typer

from nfl_gsplat.calibration.annotate_gui import annotate as run_annotator
from nfl_gsplat.calibration.keyframes import Keyframe, save_keyframes
from nfl_gsplat.errors import SetupError
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.io import read_json
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    play_dir: Path = typer.Option(..., help="path to the play folder, e.g. data/2024/week_01/NO_at_ATL/play_001"),
    camera: list[str] = typer.Option(["sideline", "endzone"], "-c", "--camera"),
    keyframe: list[int] = typer.Option([0], "--keyframe", help="frame index(es) to annotate as anchors"),
) -> None:
    pd = PlayDir.from_dir(play_dir)
    pd.dir.mkdir(parents=True, exist_ok=True)

    for cam in camera:
        video = pd.video(cam)

        if not video.exists():
            raise SetupError(
                f"missing video {video}. Place the broadcast clip at that path "
                "(see SETUP.md §2)."
            )

        keyframes: list[Keyframe] = []
        for k in keyframe:
            _LOG.info(f"launching annotation GUI for {cam} frame {k}")
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tmp_json = Path(tf.name)
            try:
                run_annotator(video, tmp_json, frame_index=k, window_title=f"{cam} — frame {k}")
                entries = read_json(tmp_json)
            finally:
                if tmp_json.exists():
                    tmp_json.unlink()

            lms = {d["name"]: (float(d["uv"][0]), float(d["uv"][1])) for d in entries}
            keyframes.append(Keyframe(frame=k, landmarks=lms))

        out_path = save_keyframes(pd.keyframes_json(cam), keyframes)
        _LOG.info(f"wrote {out_path}")
        print(f"[02_calibrate] {cam}: wrote {len(keyframes)} keyframe(s) → {out_path}")

    print()
    print("Next step: run  scripts/02b_track_calibration.py --play-dir", play_dir)


if __name__ == "__main__":
    app()
