"""Interactive Gaussian-splat preview with viser.

Serves a local web-based viewer of a merged :class:`GaussianBatch`. The heavy
import is lazy so the rest of the pipeline stays import-safe on CPU boxes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from nfl_gsplat.compositing.merge_ply import GaussianBatch, load_gaussian_ply
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def _lazy_viser():
    try:
        import viser  # type: ignore
    except ImportError as e:
        raise SetupError(
            "viser not installed — activate the `nfl_gsplat` conda env. See SETUP.md §1."
        ) from e
    return viser


def serve(batch: GaussianBatch, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start a blocking viser server serving ``batch`` as a point cloud with
    color from the SH DC coefficient. Ctrl-C to stop.

    Viser's GS viewer is not part of its public API yet; we approximate by
    rendering the DC color as a simple point cloud which is perfectly adequate
    for "does my scene look broken?" sanity checks.
    """
    viser = _lazy_viser()
    server = viser.ViserServer(host=host, port=port)
    # SH DC → approximate RGB in [0, 1]. SH_C0 = 1/sqrt(4π) ≈ 0.28209479.
    rgb = np.clip(0.5 + 0.28209479 * batch.sh[:, :, 0], 0.0, 1.0)
    server.scene.add_point_cloud(
        "/gs", points=batch.xyz, colors=(rgb * 255).astype(np.uint8), point_size=0.02,
    )
    _LOG.info(f"viser viewer at http://{host}:{port}")
    while True:
        time.sleep(1.0)


def serve_ply(path: Path | str, **kwargs) -> None:
    batch = load_gaussian_ply(path)
    serve(batch, **kwargs)
