"""Every module that asks for a logger must actually get one.

Regression guard. ``get_logger`` used a single global ``_CONFIGURED`` boolean,
so the FIRST module to call it configured its own logger and set the flag; every
module afterwards got a bare logger with no handler and no level. WARNING and
above still appeared, because logging.lastResort prints those to stderr, so the
breakage looked like "some logs show up" rather than "logging is broken" -- and
INFO diagnostics vanished repo-wide. The endzone mosaic's yard-line reprojection
error, the one number that says whether the calibration is trustworthy, was
among the messages being dropped.
"""
from __future__ import annotations

import logging


def test_second_module_to_ask_still_logs():
    """A logger requested AFTER another one must still emit INFO.

    Captured with a handler on the package logger rather than caplog: the
    package logger sets ``propagate = False`` on purpose, so nothing reaches
    pytest's root-level capture, and a caplog-based assertion here would fail
    against a perfectly healthy configuration.
    """
    import nfl_gsplat.utils.logging as ul

    ul._CONFIGURED.clear()
    ul._HANDLERS = None
    first = ul.get_logger("nfl_gsplat.first_caller")
    second = ul.get_logger("nfl_gsplat.second_caller")

    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    package = logging.getLogger("nfl_gsplat")
    handler = _Capture(level=logging.INFO)
    package.addHandler(handler)
    try:
        first.info("from the first module")
        second.info("from the second module")
    finally:
        package.removeHandler(handler)

    assert "from the first module" in seen
    assert "from the second module" in seen, (
        "the second module's INFO log was dropped — get_logger configured only "
        "the first caller")


def test_module_loggers_inherit_an_effective_level():
    """A module logger must resolve to INFO, not the root default of WARNING."""
    import nfl_gsplat.utils.logging as ul

    ul._CONFIGURED.clear()
    ul._HANDLERS = None
    ul.get_logger("nfl_gsplat.calibration.endzone_refine")
    log = logging.getLogger("nfl_gsplat.calibration.endzone_refine")
    assert log.getEffectiveLevel() <= logging.INFO
    assert log.isEnabledFor(logging.INFO)
