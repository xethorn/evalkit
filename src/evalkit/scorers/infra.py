"""Scorers about the harness and the app, not the agent."""

from __future__ import annotations

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..solver import trace_from_state


@scorer(metrics=[mean()])
def infra_ok():
    """1.0 when the sample ran cleanly; 0.0 when the harness or app failed.

    Tracked as its own metric so a broken environment is visible immediately instead of
    being mistaken for an agent regression.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        kinds = [e.kind for e in trace.infra_errors]
        return Score(
            value=0.0 if kinds else 1.0,
            answer=",".join(kinds) or "clean",
            explanation=(
                "; ".join(f"{e.kind}: {e.message}" for e in trace.infra_errors) or "no infra errors"
            ),
            metadata={
                "infra_error_kinds": kinds,
                "artifacts": trace.artifacts,
                "truncated_events": trace.truncated_events,
                "notes": trace.notes,
            },
        )

    return score


@scorer(metrics=[mean()])
def agent_error_rate():
    """1.0 when the agent reported its own failure ("I was unable to generate a response").

    Deliberately separate from `infra_ok`: the app was healthy and the agent replied, so
    this is a quality failure and stays in the quality metrics. But it needs its own rate,
    because an agent that starts erroring on a fifth of turns otherwise reads as a mild
    dip in the judge's average rather than as the outage it is.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        return Score(
            value=1.0 if trace.agent_errors else 0.0,
            answer="agent error" if trace.agent_errors else "ok",
            explanation="; ".join(trace.agent_errors) or "the agent answered every turn",
            metadata={"agent_errors": trace.agent_errors},
        )

    return score
