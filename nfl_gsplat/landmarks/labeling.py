"""Pure helpers for the landmark labeling tool (frame sampling + record build)."""
from __future__ import annotations


def sample_frame_indices(num_frames: int, count: int) -> list[int]:
    """Evenly spread ``count`` frame indices over [0, num_frames-1] inclusive."""
    if count <= 1 or num_frames <= 1:
        return [0]
    count = min(count, num_frames)
    step = (num_frames - 1) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def build_label_record(file: str, points) -> dict:
    return {"file": file,
            "points": [{"name": n, "uv": [float(u), float(v)]} for n, (u, v) in points]}
