"""Grade the agent's *process*: which tools it called, in what order, how often.

Output-only grading hides the failures that matter most. An agent can narrate a posted
journal entry it never posted, or reach the right number by scanning ten thousand rows
instead of calling the aggregate. Both look fine to a judge reading prose and are caught
here.

Tool names are matched loosely (case-insensitive, and a suite name matches any observed
tool that contains it) because backend display names drift; the exact observed names are
always recorded so a rename shows up as a diff rather than a silent regression.
"""

from __future__ import annotations

from functools import lru_cache

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..solver import trace_from_state
from ..trace import AgenticTrace
from .common import EXCLUDED, EXCLUSION_REASON, QUALITY_METRICS, clamp


def _squash(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _display_aliases() -> dict[str, str]:
    """The target's machine-name -> display-name table.

    A product usually reports *display* names on the wire ("Database Query") while suites
    are written against the real one ("execute_sql_query"). The target supplies the
    mapping so a suite never has to know which spelling it will see, and the framework
    never has to know the product's tool list. Resolved per call, and never fatal: a
    target that offers no table just gets exact matching. Cached — call
    ``_display_aliases.cache_clear()`` after switching targets in a test.
    """
    try:
        from ..config import settings

        return settings().target.tool_aliases()
    except Exception:
        return {}


def _aliases(name: str, table: dict[str, str] | None = None) -> set[str]:
    """Every spelling a tool might appear under."""
    table = _display_aliases() if table is None else table
    squashed = _squash(name)
    out = {squashed}
    for machine, display in table.items():
        if squashed in (_squash(machine), _squash(display)):
            out |= {_squash(machine), _squash(display)}
    return out


def _matches(expected: str, observed: str) -> bool:
    """Loose match, so a suite can name a step by its kind or its exact identity.

    A target may report a step under a structured identity — ``link:datalab``,
    ``endpoint:<path>``, ``connector:<slug>`` — rather than a plain tool name. A suite may
    write either the full identity or just the value — ``datalab`` matches
    ``link:datalab`` — because the useful assertion is usually "it ran a query", not
    "it ran a query and the link type string was exactly this".
    """
    expected, observed = expected.strip(), observed.strip()
    for prefix in ("link:", "endpoint:", "connector:"):
        if (
            observed.lower().startswith(prefix)
            and not expected.lower().startswith(prefix)
            and _squash(expected)
            and _squash(expected) in _squash(observed)
        ):
            return True
    table = _display_aliases()
    exp, obs = _aliases(expected, table), _aliases(observed, table)
    if exp & obs:
        return True
    # Fall back to containment so a suite can match a family of endpoint-dispatched
    # tools ("write_endpoint" matching "Write Operation: Post Journal Entry").
    return any(e in o or o in e for e in exp for o in obs)


def _found(expected: str, trace: AgenticTrace) -> bool:
    return any(_matches(expected, name) for name in trace.tool_name_set)


def _count(expected: str, trace: AgenticTrace) -> int:
    return sum(1 for c in trace.tool_calls if _matches(expected, c.key))


def _order_ok(order: list[str], trace: AgenticTrace) -> bool:
    """Is ``order`` a subsequence of the observed call sequence?"""
    remaining = list(order)
    for name in trace.tool_names:
        if remaining and _matches(remaining[0], name):
            remaining.pop(0)
    return not remaining


@scorer(metrics=QUALITY_METRICS)
def tool_calls():
    """One score covering required / required-any / forbidden / order / call-budget.

    Every declared expectation is one check; the score is the fraction satisfied, so a
    partial regression is visible instead of collapsing to a binary fail.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        spec = ((state.metadata or {}).get("expect") or {}).get("tools") or {}
        required = spec.get("required") or []
        required_any = spec.get("required_any") or []
        forbidden = spec.get("forbidden") or []
        order = spec.get("order") or []
        max_calls = spec.get("max_calls") or {}

        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        if not any([required, required_any, forbidden, order, max_calls]):
            return Score(
                value=0.0,
                metadata={
                    EXCLUDED: True,
                    EXCLUSION_REASON: ["no_tool_expectations"],
                    "observed_tools": trace.tool_names,
                },
            )

        checks: list[tuple[str, bool, str]] = []
        for name in required:
            ok = _found(name, trace)
            checks.append((f"required:{name}", ok, "called" if ok else "never called"))
        if required_any:
            ok = any(_found(name, trace) for name in required_any)
            checks.append((f"required_any:{'|'.join(required_any)}", ok, "satisfied" if ok else "none called"))
        for name in forbidden:
            hits = _count(name, trace)
            checks.append((f"forbidden:{name}", hits == 0, f"{hits} call(s)"))
        if order:
            ok = _order_ok(order, trace)
            checks.append((f"order:{'->'.join(order)}", ok, "in order" if ok else "out of order"))
        for name, budget in max_calls.items():
            hits = _count(name, trace)
            checks.append((f"max_calls:{name}<={budget}", hits <= int(budget), f"{hits} call(s)"))

        passed = [c for c in checks if c[1]]
        failed = [c for c in checks if not c[1]]
        return Score(
            value=clamp(len(passed) / len(checks)),
            answer=f"{len(passed)}/{len(checks)}",
            explanation="; ".join(f"{name} — {detail}" for name, _ok, detail in failed) or "all tool checks passed",
            metadata={
                "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks],
                "observed_tools": trace.tool_names,
                "observed_tool_detail": [
                    {"name": c.name, "subagent": c.subagent, "status": c.status, "turn": c.turn, "source": c.source}
                    for c in trace.tool_calls
                ],
            },
        )

    return score


@scorer(metrics=[mean()])
def tool_call_count():
    """Raw number of tool calls. Not pass/fail — the series is the signal.

    Recorded on every run because a change that keeps quality flat while halving the
    number of calls is exactly the kind of win a judge score will never show you.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        by_source: dict[str, int] = {}
        for call in trace.tool_calls:
            by_source[call.source] = by_source.get(call.source, 0) + 1
        failed = [c.name for c in trace.tool_calls if c.status == "failed"]
        return Score(
            value=float(len(trace.tool_calls)),
            answer=str(len(trace.tool_calls)),
            metadata={
                "by_source": by_source,
                "distinct": len(trace.tool_name_set),
                "failed_calls": failed,
                "subagents": sorted({c.subagent for c in trace.tool_calls if c.subagent}),
            },
        )

    return score


@scorer(metrics=[mean()])
def failed_tool_calls():
    """Fraction of observed tool calls that ended in failure — a leading indicator.

    Agents recover from failed calls often enough that output quality stays flat while
    the failure rate climbs. That climb is usually the first sign of a broken tool.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        if not trace.tool_calls:
            return Score(value=0.0, answer="0/0", metadata={"no_calls": True})
        failed = [c for c in trace.tool_calls if c.status == "failed"]
        return Score(
            value=len(failed) / len(trace.tool_calls),
            answer=f"{len(failed)}/{len(trace.tool_calls)}",
            metadata={"failed": [c.name for c in failed]},
        )

    return score


@scorer(metrics=QUALITY_METRICS)
def subagents():
    """Did the orchestrator delegate to the right sub-agent?

    Routing is the one part of the agent's internal structure the stream names outright
    ("Accounts Payable Agent"). It is also where silent regressions hide: an expense
    question that starts going to the reporting agent instead of AP can still produce
    plausible prose, so nothing about the answer tells you the routing broke.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        spec = ((state.metadata or {}).get("expect") or {}).get("agents") or {}
        required = spec.get("required") or []
        forbidden = spec.get("forbidden") or []

        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        if not (required or forbidden):
            return Score(
                value=0.0,
                metadata={
                    EXCLUDED: True,
                    EXCLUSION_REASON: ["no_agent_expectations"],
                    "observed_subagents": trace.subagent_names,
                },
            )

        observed = trace.subagent_names
        checks: list[tuple[str, bool, str]] = []
        for name in required:
            hit = any(_matches(name, seen) for seen in observed)
            checks.append((f"required:{name}", hit, "delegated" if hit else f"not among {observed}"))
        for name in forbidden:
            hit = any(_matches(name, seen) for seen in observed)
            checks.append((f"forbidden:{name}", not hit, "absent" if not hit else "was delegated to"))

        passed = [c for c in checks if c[1]]
        return Score(
            value=clamp(len(passed) / len(checks)),
            answer=f"{len(passed)}/{len(checks)}",
            explanation="; ".join(f"{n} — {d}" for n, ok, d in checks if not ok) or "routing as expected",
            metadata={
                "checks": [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks],
                "observed_subagents": observed,
            },
        )

    return score
