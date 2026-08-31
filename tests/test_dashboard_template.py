"""The page shell lives in dashboard.html; this checks it and its renderer still agree.

`fill` raises on a slot the renderer does not provide, so a missing value fails loudly at
render time. The other direction is silent: a value passed under a name the template no
longer contains simply disappears, and the page loses a section without anybody's test
going red. That is the case worth a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

from evalkit.analysis import dashboard
from evalkit.analysis.dashboard import _SLOT, TEMPLATE, fill

DASHBOARD = Path(dashboard.__file__)


def rendered_slots() -> set[str]:
    """The keyword names handed to the `fill(TEMPLATE, ...)` call in build_dashboard."""
    tree = ast.parse(DASHBOARD.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fill"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "TEMPLATE"
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError("build_dashboard no longer renders the template with fill(TEMPLATE, ...)")


def test_every_template_slot_is_filled() -> None:
    assert set(_SLOT.findall(TEMPLATE)) <= rendered_slots()


def test_every_rendered_value_has_a_slot() -> None:
    unused = rendered_slots() - set(_SLOT.findall(TEMPLATE))
    assert not unused, f"build_dashboard passes {sorted(unused)}, which dashboard.html no longer has a slot for"


def test_a_filled_value_is_never_itself_filled() -> None:
    """A sample's text can contain anything, braces included, and must land as content."""
    page = fill("<p>{{a}}{{b}}</p>", a="{{b}}", b="ok")
    assert page == "<p>{{b}}ok</p>"
