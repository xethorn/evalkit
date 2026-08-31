"""Scorers, grouped by what they grade.

Nothing here is a single "is it good" number on purpose. A hill climb needs to say
*which* axis moved: the answer, the process, the cost, or the environment.
"""

from .common import EXCLUDED, QUALITY_METRICS, coverage, q_mean, q_pass_rate, q_stderr
from .figures import figures
from .infra import agent_error_rate, infra_ok
from .judge import assertions, rubric_judge
from .tools import failed_tool_calls, subagents, tool_call_count, tool_calls
from .trace_checks import (
    asks_when_underspecified,
    converges,
    latency_ms,
    within_budget,
)

# Sentinel judge model: no provider, so `rubric_judge` excludes itself cleanly.
NO_JUDGE = "none/offline"


# The default battery: output quality, process quality, cost, and environment health.
def default_scorers(judge_model: str | None = None) -> list:
    """``judge_model=NO_JUDGE`` disables grading — used for offline mock runs, whose
    transcripts are synthetic, so judging them measures nothing and costs money."""
    return [
        rubric_judge(model=judge_model),
        assertions(),
        figures(model=judge_model),
        tool_calls(),
        subagents(),
        within_budget(),
        converges(),
        asks_when_underspecified(),
        tool_call_count(),
        failed_tool_calls(),
        latency_ms(),
        infra_ok(),
        agent_error_rate(),
    ]


__all__ = [
    "EXCLUDED",
    "agent_error_rate",
    "NO_JUDGE",
    "QUALITY_METRICS",
    "asks_when_underspecified",
    "assertions",
    "figures",
    "converges",
    "coverage",
    "default_scorers",
    "failed_tool_calls",
    "infra_ok",
    "latency_ms",
    "q_mean",
    "q_pass_rate",
    "q_stderr",
    "rubric_judge",
    "subagents",
    "tool_call_count",
    "tool_calls",
    "within_budget",
]
