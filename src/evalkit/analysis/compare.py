"""Compare two runs the way a hill climb requires: paired, per sample, with uncertainty.

Comparing two headline averages is how teams talk themselves into regressions. An
agent's score on a sample varies between epochs, suites are small, and the two runs
often cover slightly different samples. So:

* Comparison is **paired** on ``sample_id`` — only samples present in both runs count.
* Each sample contributes its **mean across epochs**, which is its best point estimate.
* Uncertainty comes from a **paired bootstrap** over samples, giving a CI on the delta.
* Pass/fail movement is counted separately (**McNemar-style** flip counts), because
  "three samples started passing and one started failing" is what a reviewer wants to
  read, not "+0.04".
* Suite-hash and judge-model mismatches are reported as **warnings**, not silently
  averaged over — those runs are not measuring the same thing.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .store import resolve_runs, sample_values, scorers_for

PASS_THRESHOLD = 0.999
# Scorers where a lower number is the improvement. `agent_error_rate` belongs here: it is
# the fraction of turns where the agent reported its own failure, so 0 is the good end.
LOWER_IS_BETTER = {"tool_call_count", "failed_tool_calls", "latency_ms", "agent_error_rate"}
# Scorers that describe the environment rather than the agent's quality.
DIAGNOSTIC = {"infra_ok", "tool_call_count", "failed_tool_calls", "latency_ms"}
# Scorers whose value is a measurement, not a 0–1 score. "Passed" and "regressed to fail"
# are meaningless for these — 7.8 tool calls crossing a threshold is not a flip — so only
# their deltas are reported.
CONTINUOUS = {"tool_call_count", "latency_ms", "failed_tool_calls", "agent_error_rate"}


@dataclass
class SampleDelta:
    sample_id: str
    baseline: float
    candidate: float
    delta: float
    baseline_epochs: int
    candidate_epochs: int

    @property
    def flipped(self) -> str:
        b_pass = self.baseline >= PASS_THRESHOLD
        c_pass = self.candidate >= PASS_THRESHOLD
        if b_pass and not c_pass:
            return "regressed"
        if c_pass and not b_pass:
            return "fixed"
        if self.delta < -0.01:
            return "worse"
        if self.delta > 0.01:
            return "better"
        return "unchanged"


@dataclass
class ScorerComparison:
    scorer: str
    n_paired: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    fixed: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)
    samples: list[SampleDelta] = field(default_factory=list)
    baseline_only: list[str] = field(default_factory=list)
    candidate_only: list[str] = field(default_factory=list)

    @property
    def lower_is_better(self) -> bool:
        return self.scorer in LOWER_IS_BETTER

    @property
    def improved(self) -> bool:
        signed = -self.delta if self.lower_is_better else self.delta
        return signed > 0

    @property
    def significant(self) -> bool:
        """Does the CI on the delta exclude zero?"""
        return (self.ci_low > 0 and self.ci_high > 0) or (self.ci_low < 0 and self.ci_high < 0)

    @property
    def verdict(self) -> str:
        if self.n_paired == 0:
            return "no paired samples"
        if not self.significant:
            return "no significant change"
        return "improved" if self.improved else "regressed"


@dataclass
class RunComparison:
    baseline_id: str
    candidate_id: str
    suite: str
    scorers: list[ScorerComparison]
    warnings: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> ScorerComparison | None:
        return next((s for s in self.scorers if s.scorer == name), None)

    @property
    def quality_scorers(self) -> list[ScorerComparison]:
        return [s for s in self.scorers if s.scorer not in DIAGNOSTIC]

    @property
    def headline(self) -> str:
        improved = [s.scorer for s in self.quality_scorers if s.significant and s.improved]
        regressed = [s.scorer for s in self.quality_scorers if s.significant and not s.improved]
        if regressed:
            return f"REGRESSION in {', '.join(regressed)}" + (f"; gains in {', '.join(improved)}" if improved else "")
        if improved:
            return f"IMPROVED: {', '.join(improved)}"
        return "no significant change"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(
    deltas: list[float], iterations: int = 10000, seed: int = 12345, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Percentile CI on the mean delta, plus a two-sided bootstrap p-value.

    Resamples *samples* (not epochs): the sample is the unit of independent evidence, and
    treating each epoch as independent would shrink the interval to a lie.
    """
    if not deltas:
        return 0.0, 0.0, 1.0
    rng = random.Random(seed)
    n = len(deltas)
    observed = _mean(deltas)
    means = []
    for _ in range(iterations):
        means.append(_mean([deltas[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    low = means[int((alpha / 2) * iterations)]
    high = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    # Fraction of resamples on the far side of zero, doubled for a two-sided test.
    tail = sum(1 for m in means if (m <= 0 if observed > 0 else m >= 0)) / iterations
    return low, high, min(1.0, 2 * tail)


def compare_scorer(
    conn: sqlite3.Connection,
    baseline_id: str,
    candidate_id: str,
    suite: str,
    scorer: str,
    seed: int = 12345,
) -> ScorerComparison:
    base = sample_values(conn, baseline_id, suite, scorer)
    cand = sample_values(conn, candidate_id, suite, scorer)
    shared = sorted(set(base) & set(cand))

    samples = [
        SampleDelta(
            sample_id=sid,
            baseline=_mean(base[sid]),
            candidate=_mean(cand[sid]),
            delta=_mean(cand[sid]) - _mean(base[sid]),
            baseline_epochs=len(base[sid]),
            candidate_epochs=len(cand[sid]),
        )
        for sid in shared
    ]
    deltas = [s.delta for s in samples]
    low, high, p = paired_bootstrap(deltas, seed=seed)

    return ScorerComparison(
        scorer=scorer,
        n_paired=len(samples),
        baseline_mean=_mean([s.baseline for s in samples]),
        candidate_mean=_mean([s.candidate for s in samples]),
        delta=_mean(deltas),
        ci_low=low,
        ci_high=high,
        p_two_sided=p,
        # Flip labels only where a 0–1 pass/fail reading is meaningful.
        fixed=[s.sample_id for s in samples if scorer not in CONTINUOUS and s.flipped == "fixed"],
        regressed=[s.sample_id for s in samples if scorer not in CONTINUOUS and s.flipped == "regressed"],
        samples=sorted(samples, key=lambda s: s.delta),
        baseline_only=sorted(set(base) - set(cand)),
        candidate_only=sorted(set(cand) - set(base)),
    )


def compare_runs(
    conn: sqlite3.Connection, baseline: Any, candidate: Any, seed: int = 12345
) -> RunComparison:
    baseline_id, candidate_id = baseline["run_id"], candidate["run_id"]
    suite = baseline["suite"]
    warnings: list[str] = []
    if candidate["suite"] != suite:
        raise ValueError(
            f"cannot compare across suites ({suite} vs {candidate['suite']}) — "
            "different suites have different samples, so a combined delta is meaningless"
        )

    if baseline["suite_sha"] and baseline["suite_sha"] != candidate["suite_sha"]:
        warnings.append(
            f"suite changed between runs ({baseline['suite_sha']} -> {candidate['suite_sha']}): "
            "prompts or rubrics differ, so the comparison is not apples-to-apples"
        )
    if baseline["judge_model"] != candidate["judge_model"]:
        warnings.append(
            f"judge model changed ({baseline['judge_model']} -> {candidate['judge_model']}): "
            "judge-graded deltas measure the judge as much as the agent"
        )
    if baseline["base_seed"] != candidate["base_seed"]:
        warnings.append(
            f"seed changed ({baseline['base_seed']} -> {candidate['base_seed']}): "
            "samples were asked different randomized prompts"
        )
    if baseline["change_id"] and baseline["change_id"] == candidate["change_id"]:
        warnings.append(
            "both runs tested the same code (identical change_id) — any delta here is pure run-to-run noise, "
            "which makes this a useful measurement of how much noise your suite has"
        )
    for row, name in ((baseline, "baseline"), (candidate, "candidate")):
        if row["epochs"] and row["epochs"] < 3:
            warnings.append(f"{name} ran only {row['epochs']} epoch(s); deltas will be dominated by noise")

    names = sorted(set(scorers_for(conn, baseline_id, suite)) | set(scorers_for(conn, candidate_id, suite)))
    comparisons = [compare_scorer(conn, baseline_id, candidate_id, suite, name, seed=seed) for name in names]
    return RunComparison(baseline_id, candidate_id, suite, comparisons, warnings)


def compare_refs(
    conn: sqlite3.Connection, baseline_ref: str, candidate_ref: str, seed: int = 12345
) -> tuple[list[RunComparison], list[str]]:
    """Compare two run references, one comparison per suite they share.

    Suites are never merged: they have different samples and different difficulty, so a
    pooled average tells you a number moved without telling you where.
    """
    base_rows = {r["suite"]: r for r in resolve_runs(conn, baseline_ref)}
    cand_rows = {r["suite"]: r for r in resolve_runs(conn, candidate_ref)}
    notes: list[str] = []
    if not base_rows:
        notes.append(f"no run matching {baseline_ref!r}")
    if not cand_rows:
        notes.append(f"no run matching {candidate_ref!r}")

    shared = sorted(set(base_rows) & set(cand_rows))
    for suite in sorted(set(base_rows) ^ set(cand_rows)):
        side = "baseline" if suite in base_rows else "candidate"
        notes.append(f"suite {suite!r} ran only in the {side} — not compared")
    return [compare_runs(conn, base_rows[s], cand_rows[s], seed=seed) for s in shared], notes


def gate(
    comparison: RunComparison, max_regressed_samples: int = 0, min_infra_ok: float = 0.9
) -> tuple[bool, list[str]]:
    """CI gate: should this change be allowed through?

    Deliberately strict about *per-sample* regressions rather than about the average. A
    change that fixes four samples and breaks one is usually still a discussion, and the
    gate's job is to force that discussion instead of hiding it inside a mean.
    """
    reasons: list[str] = []
    for scorer in comparison.quality_scorers:
        if len(scorer.regressed) > max_regressed_samples:
            reasons.append(
                f"{scorer.scorer}: {len(scorer.regressed)} sample(s) regressed from pass to fail "
                f"({', '.join(scorer.regressed[:5])})"
            )
        if scorer.significant and not scorer.improved:
            reasons.append(
                f"{scorer.scorer}: significant decline {scorer.delta:+.3f} "
                f"(95% CI {scorer.ci_low:+.3f}..{scorer.ci_high:+.3f})"
            )
    infra = comparison.by_name("infra_ok")
    if infra and infra.candidate_mean < min_infra_ok:
        reasons.append(
            f"infra_ok={infra.candidate_mean:.2f} below {min_infra_ok:.2f}: the environment was unhealthy, "
            "so this run's quality numbers should not be trusted either way"
        )
    return (not reasons), reasons
