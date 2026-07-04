"""Pretrained Roboflow field-keypoint cache: class-name mapping + JSON I/O.

The hosted model (football-field-key-points-mvmjf) emits classes like
'30', '30-bottom', '30-top-hash', '30-bottom-sl'. Mapping to NFL_LANDMARKS
names (validated on real All-22 eval, 2026-07-03):

    '30'              -> {terr}_30_left_number    (bare = TOP number; left=+Y=image-top)
    '30-bottom'       -> {terr}_30_right_number
    '30-top-hash'     -> {terr}_30_left_hash
    '30-bottom-hash'  -> {terr}_30_right_hash
    '*-sl'            -> None  (sidelines hallucinated off-frame; dropped)
    '50*'             -> mid_50_* (territory-less)

``territory`` ('home'|'away') resolves which side of the 50 the yard numbers
belong to — the model doesn't know; the play's meta does.
"""
from __future__ import annotations

import json
from pathlib import Path

from nfl_gsplat.errors import SetupError

ModelKeypoint = tuple[str, float, float, float]   # (model_class, u, v, conf)

_VARIANT_TABLE = {
    (): ("left", "number"),
    ("bottom",): ("right", "number"),
    ("top", "hash"): ("left", "hash"),
    ("bottom", "hash"): ("right", "hash"),
    # sideline variants intentionally absent -> None (hallucinated)
}


def yard_base(model_class: str, territory: str) -> str | None:
    """'30-top-hash' -> 'away_30' (or None for non-yard classes)."""
    if territory not in ("home", "away"):
        raise ValueError(f"territory must be home/away, got {territory!r}")
    yard = model_class.split("-")[0]
    if not yard.isdigit():
        return None
    yd = int(yard)
    if yd == 50:
        return "mid_50"
    if yd < 5 or yd > 45 or yd % 5 != 0:
        return None
    return f"{territory}_{yd}"


def to_nfl_name(model_class: str, territory: str) -> str | None:
    """Roboflow class -> NFL_LANDMARKS name, or None if unmappable."""
    base = yard_base(model_class, territory)
    if base is None:
        return None
    variant = tuple(model_class.split("-")[1:])
    if variant not in _VARIANT_TABLE:
        return None
    lr, kind = _VARIANT_TABLE[variant]
    # Gate number landmarks to yards where they're painted: 10/20/30/40 + mid_50
    if kind == "number" and base != "mid_50":
        yard = int(model_class.split("-")[0])
        if yard not in {10, 20, 30, 40}:
            return None
    return f"{base}_{lr}_{kind}"


def write_kps_json(path, *, model_id: str, video_name: str, num_frames: int,
                   kp_conf: float, frames: dict[int, list[ModelKeypoint]]) -> None:
    doc = {
        "model_id": model_id, "video": video_name, "num_frames": num_frames,
        "kp_conf": kp_conf,
        "frames": {str(i): [{"name": n, "u": u, "v": v, "conf": c}
                            for (n, u, v, c) in kps]
                   for i, kps in frames.items()},
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def load_kps_json(path, *, expect_num_frames: int | None = None
                  ) -> dict[int, list[ModelKeypoint]]:
    p = Path(path)
    if not p.exists():
        raise SetupError(
            f"roboflow keypoint cache not found: {p} — run "
            "scripts/03_roboflow_precompute.py on the play video first."
        )
    doc = json.loads(p.read_text())
    if expect_num_frames is not None and doc.get("num_frames") != expect_num_frames:
        raise SetupError(
            f"stale roboflow_kps.json: cached for {doc.get('num_frames')} frames, "
            f"video has {expect_num_frames} — re-run scripts/03_roboflow_precompute.py."
        )
    return {int(i): [(k["name"], float(k["u"]), float(k["v"]), float(k["conf"]))
                     for k in kps]
            for i, kps in doc["frames"].items()}
