"""Structure of the loader that no fixture exercises: where a rule's block sits decides whether it runs."""
import ast
from pathlib import Path


def _enclosing_ifs(tree, target_name):
    """For every call to ``target_name`` in ``tree``, the source of the test of each enclosing ``if`` (outermost first)."""
    out = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk_if(child, stack)
            else:
                if isinstance(child, ast.Call):
                    f = child.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                    if name == target_name:
                        out.append(list(stack))
                walk(child, stack)

    def walk_if(node, stack):
        test = ast.unparse(node.test)
        walk(node.test, stack)
        for b in node.body:
            walk(ast.Module(body=[b], type_ignores=[]), stack + [test])
        for b in node.orelse:
            walk(ast.Module(body=[b], type_ignores=[]), stack + ["not (" + test + ")"])

    walk(tree, [])
    return out


def test_presnap_fill_runs_with_the_line_vouch_not_under_another_vouch():
    """5779113 (2026-09-20) inserted the short-team vouch above the pre-snap fill and the fill became the body of that
    switched-off vouch's ``if``: dead code for five days (play 1's pre-snap census 0.43 -> 0.51 unnoticed)."""
    src = (Path(__file__).resolve().parents[1] / "nfl_gsplat" / "render" / "play_timeline.py").read_text(encoding="utf-8")
    calls = _enclosing_ifs(ast.parse(src), "fill_presnap_holes")
    assert calls, "fill_presnap_holes is not called by the loader"
    for stack in calls:
        assert any("line_vouch_m" in t for t in stack), stack
        assert not any("SHORT_TEAM_VOUCH" in t or "POCKET_VOUCH" in t for t in stack), stack
