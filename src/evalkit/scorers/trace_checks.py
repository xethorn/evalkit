"""Scorers derived from the shape of the trace rather than its content.

These are the cheap, model-free signals that catch the regressions judges miss:
budget overruns, loops, agents that never converge, and agents that guess when they
should have asked.
"""

from __future__ import annotations

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..solver import trace_from_state
from .common import EXCLUDED, EXCLUSION_REASON, QUALITY_METRICS, clamp


@scorer(metrics=QUALITY_METRICS)
def within_budget():
    """Did the run stay inside its declared step and latency budgets?

    Budgets are per-sample because "slow" is task-specific: a month-end close may take
    three minutes; answering a balance question should not.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        expect = (state.metadata or {}).get("expect") or {}
        max_steps = expect.get("max_steps")
        max_latency = expect.get("max_latency_ms")

        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        if max_steps is None and max_latency is None:
            return Score(
                value=0.0,
                metadata={
                    EXCLUDED: True,
                    EXCLUSION_REASON: ["no_budget"],
                    "steps": len(trace.tool_calls),
                    "total_ms": trace.total_ms,
                },
            )

        checks = []
        if max_steps is not None:
            checks.append(("max_steps", len(trace.tool_calls) <= int(max_steps), len(trace.tool_calls)))
        if max_latency is not None:
            checks.append(("max_latency_ms", trace.total_ms <= int(max_latency), trace.total_ms))
        passed = sum(1 for _n, ok, _v in checks if ok)
        return Score(
            value=clamp(passed / len(checks)),
            answer=f"{passed}/{len(checks)}",
            explanation="; ".join(f"{n}={v} ({'ok' if ok else 'over budget'})" for n, ok, v in checks),
            metadata={"steps": len(trace.tool_calls), "total_ms": trace.total_ms},
        )

    return score


@scorer(metrics=[mean()])
def latency_ms():
    """Wall-clock time for the whole conversation, plus per-turn detail.

    Kept as its own metric: the point of hill-climbing is a better answer *and* a usable
    one, and a change that adds 40 seconds per turn is a regression users will feel.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        turns = [t for t in trace.assistant_turns if t.latency_ms]
        return Score(
            value=float(trace.total_ms),
            answer=f"{trace.total_ms / 1000:.1f}s",
            metadata={
                "per_turn_ms": [t.latency_ms for t in turns],
                "slowest_turn_ms": max((t.latency_ms or 0 for t in turns), default=0),
                "turns": len(trace.assistant_turns),
            },
        )

    return score


@scorer(metrics=QUALITY_METRICS)
def converges():
    """Did the conversation reach an answer within the allowed user turns?

    Separate from correctness on purpose: "asked three clarifying questions and then got
    it right" and "answered immediately and got it right" are different products.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        user_turns = sum(1 for t in trace.turns if t.role == "user")
        open_question = trace.unconverged
        return Score(
            value=0.0 if open_question else 1.0,
            answer=f"{user_turns} user turn(s)",
            explanation=(
                f"still asking after {user_turns} user turn(s)" if open_question else "conversation converged"
            ),
            metadata={"user_turns": user_turns, "notes": trace.notes},
        )

    return score


@scorer(metrics=QUALITY_METRICS)
def asks_when_underspecified():
    """For prompts that are deliberately ambiguous, asking is the correct answer.

    Without this, every eval rewards confident guessing — the single most expensive
    failure mode for an agent that takes actions.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        expect = (state.metadata or {}).get("expect") or {}
        must_ask = expect.get("must_ask")

        if must_ask is None:
            return Score(value=0.0, metadata={EXCLUDED: True, EXCLUSION_REASON: ["must_ask_unset"]})
        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )

        first = next((t for t in trace.assistant_turns), None)
        asked = bool(trace.interrupts) or bool(first and "?" in (first.text or ""))
        ok = asked if must_ask else not asked
        return Score(
            value=1.0 if ok else 0.0,
            answer="asked" if asked else "proceeded",
            explanation=(
                "asked before acting" if asked and must_ask
                else "guessed instead of asking" if must_ask
                else "asked when the prompt was already complete" if asked
                else "proceeded without asking, as expected"
            ),
            metadata={"must_ask": must_ask, "asked": asked, "interrupts": len(trace.interrupts)},
        )

    return score
