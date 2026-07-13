"""Turn a per-play tracks.parquet into the calibration masks_provider seam.

Player boxes (original-frame pixels) are zeroed out of the white mask before
line/hash detection, so players' bright uniforms don't create false field
markings (measured: unmasked players make fit_hash_rows fit diagonal-garbage
hash rows on the endzone view). Boxes are indexed by (cam, frame); calibration
rotates them per-camera before masking the rotated working frame.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from nfl_gsplat.errors import SetupError

Box = tuple[float, float, float, float]


def boxes_provider_from_tracks(
    tracks_path,
) -> Callable[[str], Callable[[int], list[Box]]]:
    """Load tracks.parquet once; return masks_provider(cam) -> boxes_for(frame)."""
    import pandas as pd

    p = Path(tracks_path)
    if not p.exists():
        raise SetupError(
            f"player tracks not found: {p} — run scripts/03b_detect_players.py "
            "on the play first (writes tracks.parquet)."
        )
    df = pd.read_parquet(p)
    by_cam: dict[str, dict[int, list[Box]]] = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples(index=False):
        by_cam[str(r.cam)][int(r.frame)].append(
            (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)))

    def masks_provider(cam: str) -> Callable[[int], list[Box]]:
        per_frame = by_cam.get(cam, {})
        return lambda fidx: per_frame.get(int(fidx), [])

    return masks_provider
