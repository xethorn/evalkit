"""Every keyword the CLI passes to build_dashboard must exist on build_dashboard.

`run` builds the dashboard automatically and `dashboard` builds it on demand, from two
separate call sites. One of them passed ``title=`` for months while the parameter was
named ``subject=``, so every run ended with "run recorded, but the dashboard failed to
build" — and because the failure was caught and printed as a hint, nothing ever went red.

A static check rather than an invocation: it needs no database, no runs and no browser,
and it fails on the argument that drifted rather than on whatever the page renders.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from evalkit.analysis.dashboard import build_dashboard

CLI = Path(inspect.getfile(__import__("evalkit.cli", fromlist=["cli"])))


def _keywords_passed_to(source: str, func_name: str) -> list[tuple[int, set[str]]]:
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
            out.append((node.lineno, {kw.arg for kw in node.keywords if kw.arg}))
    return out


def test_every_build_dashboard_call_matches_its_signature():
    accepted = set(inspect.signature(build_dashboard).parameters)
    calls = _keywords_passed_to(CLI.read_text(), "build_dashboard")
    assert calls, "no build_dashboard calls found in cli.py — did it move?"
    for lineno, passed in calls:
        unknown = passed - accepted
        assert not unknown, (
            f"cli.py:{lineno} passes {sorted(unknown)} to build_dashboard, "
            f"which accepts {sorted(accepted)}"
        )
