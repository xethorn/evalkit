"""Human-readable reports: one run, a comparison, and a trend over time.

Markdown, because these get pasted into pull requests and Slack. Every report leads with
the decision (did this help?) and only then shows the numbers, and every number that
cannot be trusted is labelled as such rather than omitted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .compare import DIAGNOSTIC, RunComparison, gate
from .store import latest_runs, metric_value, sample_values, scorers_for


def _fmt(value: float, scorer: str) -> str:
    if scorer == "latency_ms":
        return f"{value / 1000:.1f}s"
    if scorer == "tool_call_count":
        return f"{value:.1f}"
    return f"{value:.3f}"


def run_report(conn: sqlite3.Connection, run_id: str, suite: str | None = None) -> str:
    """Report on one run. A run covering several suites gets one section per suite."""
    rows = list(
        conn.execute(
            "SELECT * FROM runs WHERE run_id = ?" + (" AND suite = ?" if suite else "") + " ORDER BY suite",
            (run_id, suite) if suite else (run_id,),
        )
    )
    if not rows:
        return f"no such run: {run_id}" + (f" (suite {suite})" if suite else "")
    if len(rows) > 1:
        return "\n\n---\n\n".join(run_report(conn, run_id, r["suite"]) for r in rows)
    row = rows[0]

    suite = row["suite"]
    lines = [
        f"# Run {row['run_id']} — {suite}",
        "",
        f"- **label**: {row['label'] or '(none)'}",
        f"- **suite**: {row['suite']} v{row['suite_version']} (`{row['suite_sha']}`)",
        f"- **code under test**: change `{row['change_id']}`{' — BASELINE' if row['is_baseline'] else ''}",
        f"- **judge**: {row['judge_model']}   **seed**: {row['base_seed']}   **epochs**: {row['epochs']}",
        f"- **status**: {row['status']}   **samples**: {row['samples']}",
        "",
        "## Metrics",
        "",
        "| scorer | metric | value |",
        "| --- | --- | --- |",
    ]
    for metric in conn.execute(
        "SELECT scorer, metric, value FROM run_metrics WHERE run_id = ? AND suite = ? ORDER BY scorer, metric",
        (run_id, suite),
    ):
        lines.append(f"| {metric['scorer']} | {metric['metric']} | {_fmt(metric['value'] or 0.0, metric['scorer'])} |")

    scorer_names = scorers_for(conn, run_id, suite)
    lines += ["", "## Per-sample scores", "", "| sample | " + " | ".join(scorer_names) + " |"]
    lines.append("| --- |" + " --- |" * len(scorer_names))
    per_scorer = {name: sample_values(conn, run_id, suite, name) for name in scorer_names}
    sample_ids = sorted({sid for values in per_scorer.values() for sid in values})
    for sid in sample_ids:
        cells = []
        for name in scorer_names:
            values = per_scorer[name].get(sid)
            cells.append(_fmt(sum(values) / len(values), name) if values else "–")
        lines.append(f"| {sid} | " + " | ".join(cells) + " |")

    failures = list(
        conn.execute(
            """SELECT sample_id, epoch, scorer, value, explanation FROM sample_scores
               WHERE run_id = ? AND suite = ? AND excluded = 0 AND value IS NOT NULL AND value < 1.0
                 AND scorer NOT IN ('tool_call_count','latency_ms','failed_tool_calls')
               ORDER BY value ASC LIMIT 25""",
            (run_id, suite),
        )
    )
    if failures:
        lines += ["", "## What failed", ""]
        for f in failures:
            lines.append(
                f"- `{f['sample_id']}` epoch {f['epoch']} — **{f['scorer']}** {f['value']:.2f}: "
                f"{(f['explanation'] or '').strip()[:300]}"
            )
    return "\n".join(lines) + "\n"


def comparison_report(comparison: RunComparison, max_samples: int = 12) -> str:
    passed, reasons = gate(comparison)
    lines = [
        f"# {comparison.suite}: {comparison.candidate_id} vs {comparison.baseline_id}",
        "",
        f"**{comparison.headline}**",
        "",
        f"Gate: {'PASS' if passed else 'FAIL'}",
    ]
    if reasons:
        lines += [""] + [f"- {r}" for r in reasons]
    if comparison.warnings:
        lines += ["", "## Read this first", ""] + [f"- ⚠️ {w}" for w in comparison.warnings]

    lines += [
        "",
        "## Deltas",
        "",
        "| scorer | baseline | candidate | delta | 95% CI | p | n | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in comparison.scorers:
        tag = " (diagnostic)" if s.scorer in DIAGNOSTIC else ""
        lines.append(
            f"| {s.scorer}{tag} | {_fmt(s.baseline_mean, s.scorer)} | {_fmt(s.candidate_mean, s.scorer)} | "
            f"{s.delta:+.3f} | {s.ci_low:+.3f}..{s.ci_high:+.3f} | {s.p_two_sided:.3f} | {s.n_paired} | {s.verdict} |"
        )

    for s in comparison.scorers:
        if not (s.fixed or s.regressed):
            continue
        lines += ["", f"### {s.scorer}", ""]
        if s.fixed:
            lines.append(f"- **fixed** ({len(s.fixed)}): {', '.join(f'`{x}`' for x in s.fixed)}")
        if s.regressed:
            lines.append(f"- **regressed** ({len(s.regressed)}): {', '.join(f'`{x}`' for x in s.regressed)}")
        movers = [d for d in s.samples if abs(d.delta) > 0.01][:max_samples]
        if movers:
            lines += ["", "| sample | baseline | candidate | delta |", "| --- | --- | --- | --- |"]
            for d in movers:
                lines.append(
                    f"| `{d.sample_id}` | {_fmt(d.baseline, s.scorer)} | "
                    f"{_fmt(d.candidate, s.scorer)} | {d.delta:+.3f} |"
                )
        if s.baseline_only or s.candidate_only:
            lines.append(
                f"\n_unpaired: {len(s.baseline_only)} baseline-only, {len(s.candidate_only)} candidate-only "
                "(excluded from the delta)_"
            )
    return "\n".join(lines) + "\n"


def trend_report(
    conn: sqlite3.Connection,
    suite: str | None = None,
    scorer: str = "rubric_judge",
    limit: int = 20,
) -> str:
    """Score per run over time — the hill, as a table you can read at a glance.

    One row per (run, suite): suites are never pooled, because a gain in one and a loss
    in the other would average to "no change".
    """
    runs = list(reversed(latest_runs(conn, suite=suite, limit=limit)))
    lines = [
        f"# Trend: {scorer}" + (f" ({suite})" if suite else ""),
        "",
        "| date | run | suite | label | change | q_mean | pass_rate | coverage | infra_ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "–"

    for row in runs:
        run_id, run_suite = row["run_id"], row["suite"]
        created = (row["created_at"] or "")[:16]
        lines.append(
            f"| {created} | `{run_id}` | {run_suite} | {row['label'] or ''} | "
            f"`{(row['change_id'] or '')[:10]}`{'*' if row['is_baseline'] else ''} | "
            f"{fmt(metric_value(conn, run_id, run_suite, scorer, 'q_mean'))} | "
            f"{fmt(metric_value(conn, run_id, run_suite, scorer, 'q_pass_rate'))} | "
            f"{fmt(metric_value(conn, run_id, run_suite, scorer, 'coverage'))} | "
            f"{fmt(metric_value(conn, run_id, run_suite, 'infra_ok', 'mean'))} |"
        )
    lines += ["", "_`*` marks a run against the declared baseline code._"]
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
