"""Building a demo history: the machinery, not the story.

What the demo agent *is* — its suites, samples, prompts, answers and rubric criteria —
lives in a :class:`~evalkit.scenario.DemoScenario`; the framework ships one about a
blog-writing assistant and a target may supply its own. Everything here is the part that
does not care: writing the database, the provenance, the traces, and a throwaway git
repository with real commits.

Synthetic evaluations, for building the UI without spending real runs.

A browser-driven evaluation costs minutes and real judge tokens, so iterating on the
dashboard against live data is slow and wasteful. This module fabricates a history that
exercises every state the UI has to render: a variation that improves, one that only
changes the model, a repeat of an existing variation (which is what makes the noise floor
measurable), a regression, and samples that are flaky rather than fixed.

Two safety rules, because fake numbers that look real are worse than no numbers:

* Demo data lives in its own database (``runs/demo.db``) and its own provenance directory.
  It is never written into the real store.
* Every evaluation is labelled ``DEMO`` and the page carries a banner. If you cannot tell
  at a glance whether you are looking at real results, the tool is dangerous.

The git history is *real* — a throwaway repository with actual commits — so the diff
rendering path is exercised properly rather than against a hand-written patch string.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..scenario import DemoScenario
from .demo_blog import BLOG_SCENARIO
from .store import SCHEMA

DEMO_MARKER = "DEMO"

# Non-judge scorers, derived so the page's other panels have something honest to show.
DERIVED = {
    "tool_calls": lambda judge, rng: 1.0 if judge >= 0.5 else rng.choice([0.0, 1.0]),
    "within_budget": lambda judge, rng: 1.0,
    "converges": lambda judge, rng: 1.0,
    "assertions": lambda judge, rng: min(1.0, judge + 0.25),
    "infra_ok": lambda judge, rng: 1.0,
    "agent_error_rate": lambda judge, rng: 1.0 if judge == 0.0 else 0.0,
}
CONTINUOUS_DERIVED = {
    "latency_ms": lambda judge, rng: float(rng.randint(9000, 26000)),
    "tool_call_count": lambda judge, rng: float(rng.randint(1, 6)),
    "failed_tool_calls": lambda judge, rng: 0.0,
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _build_repo(root: Path, scenario: DemoScenario) -> dict[str, str]:
    """A throwaway repo with one commit per variation, so diffs are real."""
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=False)
    root.mkdir(parents=True)
    _git(root.parent, "init", "-q", root.name)
    _git(root, "config", "user.email", "demo@example.com")
    _git(root, "config", "user.name", "Demo")
    (root / "prompt.py").write_text("ASK_BEFORE_PUBLISH = False\nPROMPT = 'go ahead'\n")
    (root / "router.py").write_text("ROUTE = 'multi-agent'\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "demo baseline")

    shas: dict[str, str] = {}
    head = _git(root, "rev-parse", "HEAD")
    for entry in scenario.plan:
        commit = entry["commit"]
        if commit is not None:
            filename, content, message = commit
            (root / filename).write_text(content)
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", message)
            head = _git(root, "rev-parse", "HEAD")
        shas[entry["variation"]] = head
    return shas


def _write_provenance(
    runs_dir: Path, run_id: str, entry: dict, sha: str, repo: Path, created: datetime,
    scenario: DemoScenario,
) -> None:
    directory = runs_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": created.isoformat(),
                "label": f"{DEMO_MARKER} {entry['label']}",
                "seed": 20260828,
                "epochs": 3,
                "suite": scenario.suite,
                "judge_model": scenario.judge_model,
                # Obviously fake, and deliberately so: a demo record carrying a real
                # tenant id is a demo that can be mistaken for a measurement.
                **scenario.target_config,
                "target_config": scenario.target_config,
                "change_id": entry["variation"],
                "is_baseline": entry["variation"] == scenario.plan[0]["variation"],
                "demo": True,
                "variation": {
                    "id": entry["variation"],
                    "model": entry["model"],
                    "agents": [],
                },
                "repos": [
                    {
                        "name": repo.name,
                        "path": str(repo),
                        "exists": True,
                        "head_sha": sha,
                        "dirty": False,
                        "branch": "main",
                        "commits_ahead": [],
                        "files_changed": [],
                    }
                ],
                "env": {"harness_fingerprint": "demo-harness", "harness_sha": "demo"},
                "feature_toggles": {},
            },
            indent=2,
        )
    )


def build(runs_dir: Path, db_path: Path, scenario: DemoScenario | None = None) -> tuple[Path, int]:
    """Create the demo database and provenance. Returns (db_path, evaluation count)."""
    scenario = scenario or BLOG_SCENARIO
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Clear the previous demo's run directories first. The dashboard reads whatever trace
    # files it finds beside a run, so leftovers from an earlier scenario show up in the new
    # page as samples that are in no database — which is how a page about one agent ends up
    # quoting another. Only `demo-<n>-*` is touched; real runs are never named that.
    for stale in sorted(runs_dir.glob("demo-[0-9]*")):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

    repo = runs_dir / "_demo" / "demo-agent"
    shas = _build_repo(repo, scenario)

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    start = datetime.now(UTC) - timedelta(days=len(scenario.plan))
    for index, entry in enumerate(scenario.plan):
        created = start + timedelta(days=index)
        run_id = f"demo-{index + 1:02d}-{entry['variation']}"
        rng = random.Random(f"{run_id}")
        _write_provenance(runs_dir, run_id, entry, shas[entry["variation"]], repo, created, scenario)

        _insert_suite(conn, run_id, scenario.suite, entry, created, entry["scores"], rng, scenario)
        _write_traces(runs_dir, run_id, entry["scores"], rng, scenario)
        first = scenario.plan[0]["variation"]
        scores_2 = scenario.scores_2.get(entry["variation"], scenario.scores_2[first])
        _insert_suite(conn, run_id, scenario.suite_2, entry, created, scores_2, rng, scenario)
        _write_traces(runs_dir, run_id, scores_2, rng, scenario)
    conn.commit()
    conn.close()
    return db_path, len(scenario.plan)


def _insert_suite(conn, run_id, suite, entry, created, scores, rng, scenario) -> None:
    """Write one suite's rows for one evaluation."""
    if True:
        conn.execute(
            """INSERT INTO runs (run_id, suite, log_path, created_at, label, suite_version, suite_sha,
                                 change_id, is_baseline, judge_model, base_seed, epochs, status,
                                 samples, provenance)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, suite, f"logs/{run_id}.eval", created.isoformat(),
                f"{DEMO_MARKER} {entry['label']}", 4, scenario.suite_sha, entry["variation"],
                1 if entry["variation"] == scenario.plan[0]["variation"] else 0,
                scenario.judge_model, 20260828, 3, "success", len(scores) * 3,
                json.dumps({"demo": True, "variation": entry["variation"], "model": entry["model"]}),
            ),
        )

        totals: dict[str, list[float]] = {}
        for sample, judge_values in scores.items():
            for epoch, judge in enumerate(judge_values, start=1):
                row_scores = {"rubric_judge": judge}
                row_scores.update({name: fn(judge, rng) for name, fn in DERIVED.items()})
                row_scores.update({name: fn(judge, rng) for name, fn in CONTINUOUS_DERIVED.items()})
                for scorer, value in row_scores.items():
                    answer, explanation, meta = _reading_metadata(sample, scorer, value, rng, scenario)
                    conn.execute(
                        """INSERT INTO sample_scores (run_id, suite, sample_id, epoch, scorer, value,
                                                      answer, explanation, excluded, metadata)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            run_id, suite, sample, epoch, scorer, value, answer, explanation, 0,
                            json.dumps({"demo": True, **meta}),
                        ),
                    )
                    totals.setdefault(scorer, []).append(value)

        for scorer, values in totals.items():
            mean = sum(values) / len(values)
            metrics = (
                {"mean": mean}
                if scorer in CONTINUOUS_DERIVED or scorer in {"infra_ok", "agent_error_rate"}
                else {
                    "q_mean": mean,
                    "q_pass_rate": sum(1 for v in values if v >= 0.999) / len(values),
                    "coverage": 1.0,
                    "q_stderr": 0.0,
                }
            )
            for metric, value in metrics.items():
                conn.execute(
                    "INSERT INTO run_metrics (run_id, suite, scorer, metric, value) VALUES (?,?,?,?,?)",
                    (run_id, suite, scorer, metric, value),
                )


# --- synthetic traces ------------------------------------------------------
# The dashboard's result / chat / trace views read the trace files the solver writes, so
# the demo has to write them too — otherwise the very views this data exists to exercise
# are the ones that stay empty.
def _write_traces(
    runs_dir: Path, run_id: str, samples: dict[str, list[float]], rng: random.Random,
    scenario: DemoScenario,
) -> None:
    """One trace file per sample per run, shaped exactly like the solver's own output."""
    directory = runs_dir / run_id / "traces"
    directory.mkdir(parents=True, exist_ok=True)
    for sample, judge_values in samples.items():
        for epoch, judge in enumerate(judge_values, start=1):
            good = judge >= 0.7
            turns = [
                {
                    "index": 0,
                    "role": "user",
                    "text": scenario.prompts.get(sample, sample),
                    "origin": "prompt",
                }
            ]
            asked = sample in scenario.clarify and good
            if asked:
                turns.append(
                    {"index": 0, "role": "assistant", "text": scenario.clarify[sample],
                     "latency_ms": rng.randint(3000, 7000)}
                )
                turns.append(
                    {"index": 1, "role": "user", "text": scenario.user_replies[sample], "origin": "scripted"}
                )
            answer = (
                scenario.good_answers.get(sample, "Here is the result.")
                if good
                else scenario.weak_answers.get(sample, scenario.good_answers.get(sample, "Here is the result."))
            )
            turns.append(
                {"index": 1 if asked else 0, "role": "assistant", "text": answer,
                 "latency_ms": rng.randint(6000, 24000)}
            )

            subagent = scenario.subagent_by_sample.get(sample)
            names = list(scenario.steps_by_sample.get(sample, []))
            if not good and names:
                names = names + names[:1]  # a weak run re-queries
            steps = [
                {
                    "name": name,
                    "subagent": subagent,
                    "status": "completed",
                    "source": "dag",
                    "detail": f"step {i + 1} of the {subagent or 'orchestrator'}'s plan",
                }
                for i, name in enumerate(names)
            ]
            interrupts = []
            if sample == "expense-report-no-silent-writes" and good:
                interrupts.append(
                    {
                        "turn": 0,
                        "tool": "call_write_endpoint",
                        "title": "Post journal entry 7,086.64 USD to accrued liabilities",
                        "fields": ["account", "amount", "period"],
                        "decision": "approved",
                    }
                )
            failed = sample == "expense-report-no-silent-writes" and not good
            trace = {
                "sample_id": sample,
                "prompt": scenario.prompts.get(sample, sample),
                "turns": turns,
                "tool_calls": steps,
                "subagents": [{"name": subagent, "turn": 0, "source": "progress"}] if subagent else [],
                "interrupts": interrupts,
                "total_ms": rng.randint(9000, 26000),
                "infra_errors": [],
                "notes": ["agent reported its own failure"] if failed else [],
            }
            (directory / f"{sample}.epoch{epoch}.json").write_text(json.dumps(trace, indent=2))


# Score metadata, so the Result view has the same shape it has for a real run: a verdict,
# a sentence, and the judge's per-criterion evidence.
EVIDENCE = {
    True: '"…" (quoted from the reply)',
    False: "none",
}


def _reading_metadata(
    sample: str, scorer: str, value: float, rng: random.Random, scenario: DemoScenario
) -> tuple[str, str, dict]:
    """(answer, explanation, metadata) for one score, shaped like the real scorers'."""
    if scorer == "rubric_judge":
        names = scenario.criteria.get(sample, ["criterion_a", "criterion_b"])
        met_count = max(0, min(len(names), round(value * len(names))))
        criteria = [
            {"id": name, "met": i < met_count, "evidence": EVIDENCE[i < met_count]}
            for i, name in enumerate(names)
        ]
        verdict = "pass" if value >= 0.999 else ("fail" if value <= 0.001 else "partial")
        explanation = {
            "pass": "Every rubric criterion is satisfied, each supported by a quote from the reply.",
            "partial": f"{met_count} of {len(names)} criteria are met; the rest are not evidenced in the reply.",
            "fail": "The reply does not satisfy any rubric criterion.",
        }[verdict]
        return verdict, explanation, {"criteria": criteria, "judge_model": scenario.judge_model}
    if scorer == "tool_calls":
        checks = [
            {
                "check": f"required_any:{scenario.primary_step}|endpoint",
                "ok": value >= 0.5,
                "detail": "called" if value >= 0.5 else "never called",
            },
            {
                "check": f"max_calls:{scenario.primary_step}<=8",
                "ok": True,
                "detail": f"{rng.randint(1, 4)} call(s)",
            },
        ]
        passed = sum(1 for c in checks if c["ok"])
        note = "all tool checks passed" if passed == len(checks) else "a tool check failed"
        return f"{passed}/{len(checks)}", note, {"checks": checks}
    if scorer == "assertions":
        missing = [] if value >= 0.999 else [scenario.missing_assertion]
        return (
            "2/2" if not missing else "1/2",
            "all content assertions passed" if not missing else "",
            {"missing": missing} if missing else {},
        )
    if scorer == "within_budget":
        return "2/2", "max_steps and max_latency_ms both within budget", {}
    if scorer == "converges":
        return "1 user turn(s)", "conversation converged", {}
    if scorer == "infra_ok":
        return "clean", "no infra errors", {}
    if scorer == "agent_error_rate":
        errored = value >= 0.5
        return (
            "agent error" if errored else "ok",
            "the agent reported its own failure" if errored else "the agent answered every turn",
            {},
        )
    return "", "", {}
