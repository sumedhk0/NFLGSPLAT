"""Merge Gaussian primitives from multiple sources (field + players + ball).

All primitives from the pipeline's four sources land in a single Gaussian
batch whose fields map 1:1 to the on-disk 3DGS PLY format::

    xyz        [N, 3]     world-space means
    rot        [N, 4]     wxyz quaternion
    scale      [N, 3]     log-scale
    opacity    [N]        logit-opacity
    sh         [N, 3, K]  SH coefficients (K = (deg+1)²); first column = DC
    sh_degree  int

SH-degree normalization: if sources disagree, everything is trimmed down to
the minimum degree present. This is a lossy-but-necessary step — gsplat's
rasterizer needs one SH degree per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class GaussianBatch:
    xyz: np.ndarray                     # [N, 3] float32
    rot: np.ndarray                     # [N, 4] float32, wxyz
    scale: np.ndarray                   # [N, 3] float32 (log-space)
    opacity: np.ndarray                 # [N] float32 (logit-space)
    sh: np.ndarray                      # [N, 3, K] float32
    sh_degree: int

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    def assert_no_nans(self) -> None:
        for name, arr in (("xyz", self.xyz), ("rot", self.rot),
                          ("scale", self.scale), ("opacity", self.opacity),
                          ("sh", self.sh)):
            if not np.isfinite(arr).all():
                bad = int(np.sum(~np.isfinite(arr)))
                raise ValueError(f"GaussianBatch.{name}: {bad} non-finite entries")


def _sh_degree_from_k(k: int) -> int:
    deg = int(round(np.sqrt(k) - 1))
    if (deg + 1) ** 2 != k:
        raise ValueError(f"K={k} is not a perfect square for any SH degree")
    return deg


def merge_batches(batches: Sequence[GaussianBatch]) -> GaussianBatch:
    """Concatenate batches along the ``N`` axis after trimming to min SH degree."""
    if not batches:
        raise ValueError("merge_batches got no inputs")
    min_deg = min(b.sh_degree for b in batches)
    min_k = (min_deg + 1) ** 2

    def _trim_sh(b: GaussianBatch) -> np.ndarray:
        return b.sh[:, :, :min_k]

    out = GaussianBatch(
        xyz=np.concatenate([b.xyz.astype(np.float32) for b in batches], axis=0),
        rot=np.concatenate([b.rot.astype(np.float32) for b in batches], axis=0),
        scale=np.concatenate([b.scale.astype(np.float32) for b in batches], axis=0),
        opacity=np.concatenate([b.opacity.astype(np.float32) for b in batches], axis=0),
        sh=np.concatenate([_trim_sh(b).astype(np.float32) for b in batches], axis=0),
        sh_degree=min_deg,
    )
    out.assert_no_nans()
    return out


# --- PLY I/O ---------------------------------------------------------------

def _ply_property_names(sh_degree: int) -> list[str]:
    """Canonical property ordering for 3DGS PLY."""
    k_total = (sh_degree + 1) ** 2
    k_rest = k_total - 1
    names = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    # f_rest layout: rest coefficients for all 3 channels, R then G then B.
    for c in range(3 * k_rest):
        names.append(f"f_rest_{c}")
    names += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def save_gaussian_ply(path: Path | str, batch: GaussianBatch) -> Path:
    """Write a standard 3DGS PLY (binary little-endian)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch.assert_no_nans()
    N = batch.num_gaussians
    sh_deg = batch.sh_degree
    names = _ply_property_names(sh_deg)

    f_dc = batch.sh[:, :, 0]                        # [N, 3]
    f_rest = batch.sh[:, :, 1:]                     # [N, 3, K-1]
    k_rest = f_rest.shape[-1]
    # Reorder to [N, 3 * k_rest] in R0..R{k-1} G0..G{k-1} B0..B{k-1} order.
    f_rest_flat = f_rest.reshape(N, 3 * k_rest)

    normals = np.zeros((N, 3), dtype=np.float32)
    row = np.concatenate([
        batch.xyz.astype(np.float32),
        normals,
        f_dc.astype(np.float32),
        f_rest_flat.astype(np.float32),
        batch.opacity.reshape(N, 1).astype(np.float32),
        batch.scale.astype(np.float32),
        batch.rot.astype(np.float32),
    ], axis=1)
    assert row.shape[1] == len(names), (
        f"row width {row.shape[1]} != header property count {len(names)}"
    )

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "".join(f"property float {n}\n" for n in names)
        + "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(row.astype(np.float32).tobytes(order="C"))
    return path


def load_gaussian_ply(path: Path | str) -> GaussianBatch:
    """Parse a standard 3DGS PLY into a :class:`GaussianBatch`."""
    path = Path(path)
    with open(path, "rb") as fh:
        line = fh.readline().strip()
        if line != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = fh.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"{path}: unsupported format {fmt!r}")
        count: int | None = None
        props: list[tuple[str, str]] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            line = line.strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            elif line.startswith(b"property"):
                _p, dtype, name = line.split()
                props.append((name.decode("ascii"), dtype.decode("ascii")))
            elif line == b"end_header":
                break
        assert count is not None
        if any(dt != "float" for _, dt in props):
            raise ValueError(f"{path}: only float properties supported")
        names = [n for n, _ in props]
        row_bytes = count * len(names) * 4
        raw = fh.read(row_bytes)
        if len(raw) < row_bytes:
            raise ValueError(f"{path}: truncated payload")
        arr = np.frombuffer(raw, dtype=np.float32).reshape(count, len(names))

    # Pick columns by name. If some are missing (e.g. no f_rest_*), we still
    # succeed with SH degree 0.
    def _col(n: str) -> np.ndarray:
        return arr[:, names.index(n)]

    xyz = np.stack([_col("x"), _col("y"), _col("z")], axis=1)
    rot = np.stack([_col("rot_0"), _col("rot_1"), _col("rot_2"), _col("rot_3")], axis=1)
    scale = np.stack([_col("scale_0"), _col("scale_1"), _col("scale_2")], axis=1)
    opacity = _col("opacity")
    f_dc = np.stack([_col("f_dc_0"), _col("f_dc_1"), _col("f_dc_2")], axis=1)  # [N, 3]

    rest_cols = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_cols:
        rest_arr = np.stack([arr[:, names.index(n)] for n in rest_cols], axis=1)
        k_rest = rest_arr.shape[1] // 3
        if rest_arr.shape[1] != 3 * k_rest:
            raise ValueError(f"{path}: f_rest count {rest_arr.shape[1]} not divisible by 3")
        rest_arr = rest_arr.reshape(count, 3, k_rest)
    else:
        rest_arr = np.zeros((count, 3, 0), dtype=np.float32)

    sh = np.concatenate([f_dc[:, :, None], rest_arr], axis=2)
    sh_degree = _sh_degree_from_k(sh.shape[-1])
    return GaussianBatch(xyz=xyz, rot=rot, scale=scale, opacity=opacity,
                         sh=sh, sh_degree=sh_degree)


def batch_from_arrays(
    xyz: np.ndarray, rot: np.ndarray, scale: np.ndarray,
    opacity: np.ndarray, sh: np.ndarray,
) -> GaussianBatch:
    sh_degree = _sh_degree_from_k(sh.shape[-1])
    return GaussianBatch(xyz=xyz, rot=rot, scale=scale, opacity=opacity,
                         sh=sh, sh_degree=sh_degree)
