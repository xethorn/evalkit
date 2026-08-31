"""Shared scoring conventions and metrics.

Every scorer here returns a float in ``[0, 1]`` plus rich metadata, so scores are
directly comparable between runs and can be averaged, differenced and bootstrapped.

The one subtlety is **exclusion**. When a sample failed for infrastructure reasons
(expired session, app 500, Playwright timeout) the agent never got a fair chance, so
quality scorers mark the score ``excluded`` and the metrics below skip it. Counting
those as zeros is the classic way to make a suite untrustworthy: infra flakiness starts
looking like a model regression, and a good change gets reverted.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Metric, SampleScore, Value, metric

EXCLUDED = "excluded"
EXCLUSION_REASON = "exclusion_reason"


def excluded(score_metadata: dict[str, Any] | None) -> bool:
    return bool((score_metadata or {}).get(EXCLUDED))


def _values(scores: list[SampleScore]) -> list[float]:
    out: list[float] = []
    for s in scores:
        if excluded(s.score.metadata):
            continue
        value = s.score.value
        if isinstance(value, (int, float, bool)):
            out.append(float(value))
    return out


@metric
def q_mean() -> Metric:
    """Mean over gradeable samples only."""

    def compute(scores: list[SampleScore]) -> Value:
        vals = _values(scores)
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@metric
def q_stderr() -> Metric:
    """Standard error over gradeable samples — the bar for "is this a real improvement?"."""

    def compute(scores: list[SampleScore]) -> Value:
        vals = _values(scores)
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return (var / len(vals)) ** 0.5

    return compute


@metric
def q_pass_rate(threshold: float = 0.999) -> Metric:
    """Fraction of gradeable samples at or above ``threshold`` (default: fully passing)."""

    def compute(scores: list[SampleScore]) -> Value:
        vals = _values(scores)
        if not vals:
            return 0.0
        return sum(1 for v in vals if v >= threshold) / len(vals)

    return compute


@metric
def coverage() -> Metric:
    """Fraction of samples that were actually gradeable. A drop here invalidates a trend."""

    def compute(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        return sum(0 if excluded(s.score.metadata) else 1 for s in scores) / len(scores)

    return compute


QUALITY_METRICS = [q_mean(), q_stderr(), q_pass_rate(), coverage()]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
