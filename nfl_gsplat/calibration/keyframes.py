"""Per-camera keyframe annotations for per-frame calibration.

A keyframe is a frame index + the landmark pixel clicks on that frame. Stored as
``{cam}_keyframes.json`` so the batch tracker (02b) re-uses them without
re-annotating. Anchors are solved by solve_pnp; the tracker fills the frames
between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.io import read_json, write_json


@dataclass(frozen=True)
class Keyframe:
    frame: int
    landmarks: dict[str, tuple[float, float]]   # name -> (u, v)


def save_keyframes(path: Path | str, keyframes: list[Keyframe]) -> Path:
    payload = [
        {"frame": int(k.frame),
         "landmarks": [{"name": n, "uv": [float(u), float(v)]}
                       for n, (u, v) in k.landmarks.items()]}
        for k in keyframes
    ]
    write_json(path, payload)
    return Path(path)


def load_keyframes(path: Path | str) -> list[Keyframe]:
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"keyframes not found at {path}. Annotate first: "
            "scripts/02_calibrate_cameras.py --play-dir <dir> --annotate. See SETUP.md §3."
        )
    out: list[Keyframe] = []
    for entry in read_json(path):
        lms = {d["name"]: (float(d["uv"][0]), float(d["uv"][1])) for d in entry["landmarks"]}
        out.append(Keyframe(frame=int(entry["frame"]), landmarks=lms))
    return sorted(out, key=lambda k: k.frame)
