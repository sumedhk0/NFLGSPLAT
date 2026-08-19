"""No module may define the same top-level name twice.

This is a guard against a specific, repeated accident, not a style rule. Several
modules here were extended by running a throwaway script that appended a
function to the file. Re-running that script -- or writing a second one that
"fixed" the same function -- appended the definition again instead of replacing
it. Python accepts that silently: the LAST definition wins, so the file reads as
if the earlier copy were live while the interpreter uses the later one.

It bit this package six times during endzone bring-up. Twice the two copies had
DRIFTED, so the version being read while debugging was not the version running,
and the visible code explained behaviour that could not happen. Both surviving
cases (``bundle_adjust``, ``register_dense_to_reference``) were byte-identical
duplicates, which is the lucky outcome.

Checked with ast rather than text so it sees only real top-level definitions --
a name rebound inside ``if``/``try`` (an import fallback) is legitimate and does
not trip this.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "nfl_gsplat"
_MODULES = sorted(_ROOT.rglob("*.py"))


def _top_level_names(tree):
    """(name, lineno) for every def/class/assignment at module top level."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    yield tgt.id, node.lineno


@pytest.mark.parametrize("path", _MODULES, ids=lambda p: p.name)
def test_module_defines_each_name_once(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    dupes = []
    for name, lineno in _top_level_names(tree):
        if name in seen:
            dupes.append(f"{name!r} at line {seen[name]} and again at {lineno}")
        seen[name] = lineno
    assert not dupes, (
        f"{path.relative_to(_ROOT.parent)} defines the same top-level name "
        "more than once, so the earlier definition is dead code that still "
        "reads as live:\n  " + "\n  ".join(dupes))
