"""Pipeline-wide exception types.

Every error should name the file/resource missing and point to the relevant
SETUP.md section. Never degrade silently; raise with an actionable message.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class SetupError(PipelineError):
    """A prerequisite (weights, annotations, video) is missing or malformed.

    The message MUST include the exact expected file path and the SETUP.md
    section the user should read.
    """


class CalibrationError(PipelineError):
    """Calibration failed numerically (e.g., reprojection > threshold)."""


class PoseFusionError(PipelineError):
    """SMPL-X 2-view fusion failed for a player (too many invalid frames)."""


class LHMVRAMError(PipelineError):
    """Insufficient VRAM for any LHM++ variant. We do not silently downgrade."""


class CacheHashMismatch(PipelineError):
    """Upstream inputs changed and a stage's cache is stale (use --force)."""


class IdentityError(PipelineError):
    """Player identity could not be resolved (missing/misaligned roster data).

    The message MUST name the missing resource (roster parquet, participation
    alignment) and point to the relevant SETUP.md section. We never guess a
    ``player_uid`` silently when the roster prior is requested but unavailable.
    """
