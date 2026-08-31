"""The command surface.

The workflow this is shaped around:

    evalkit doctor                      # is the environment fit to measure anything?
    evalkit baseline set                # pin today's code as the reference point
    evalkit run --label baseline        # record how we do today
    ... make a change to the agent ...
    evalkit run --label candidate --compare-to baseline
    evalkit trend                       # the hill, over time

Messages that tell you what to type use whatever this was actually invoked as, since a
distribution names its own console script — see ``PROGRAM``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import provenance as prov_mod
from .config import BORROWED_FROM, BORROWED_SECRETS, ENV_PREFIXES, LEGACY_ENV_USED, REPO_ROOT, settings
from .datasets import discover, load_suite
from .registry import target_ref
from .target import Check

app = typer.Typer(add_completion=False, help="Local, record-keeping evals for an AI agent.")
console = Console()

# What to call this program when telling someone what to run next. Taken from how it was
# actually invoked, because the console script belongs to whichever distribution ships the
# framework; `evalkit` is the fallback when that cannot be read (a test runner, `python -m`).
_INVOKED = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
PROGRAM = _INVOKED if _INVOKED and not _INVOKED.startswith(("python", "pytest", "-c")) else "evalkit"

DEFAULT_SUITE_DIR = REPO_ROOT / "evals" / "suites"


def _mount_target_commands() -> None:
    """Add whatever the configured target contributes (logging in, probing it).

    Failure is not fatal and not silent: the framework's own commands — `compare`,
    `dashboard`, `report` — read recorded results and stay useful even when the target
    package is missing or misconfigured.
    """
    try:
        extra = settings().target.cli()
    except Exception as exc:
        console.print(f"[yellow]target commands unavailable: {exc}[/yellow]")
        return
    if extra is None:
        return
    for group in extra.registered_groups:
        app.add_typer(group.typer_instance, name=group.name)
    for command in extra.registered_commands:
        app.registered_commands.append(command)


_mount_target_commands()


def _run_id(label: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in label)[:24].strip("-")
    return f"{stamp}-{slug}" if slug else stamp


# --------------------------------------------------------------------------- run
@app.command()
def run(
    suite: list[Path] = typer.Option(None, "--suite", "-s", help="Suite YAML (repeatable). Default: all."),
    label: str = typer.Option("", "--label", "-l", help="Name for this run, e.g. 'baseline' or 'sharper-prompt'."),
    tags: str = typer.Option("", "--tags", help="Only cases with these tags (comma-separated)."),
    ids: str = typer.Option("", "--ids", help="Only these case ids (comma-separated)."),
    severity: str = typer.Option("", "--severity", help="Only cases of this severity."),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Repeats per case. Overrides EVAL_EPOCHS."),
    seed: int = typer.Option(None, "--seed", help="Base seed for prompt randomization."),
    limit: int = typer.Option(None, "--limit", help="Cap the number of cases (smoke runs)."),
    max_samples: int = typer.Option(None, "--max-samples", help="Concurrent browser sessions."),
    simulated_user: str = typer.Option(
        "", "--simulated-user", help="Model to play the user when no scripted answer matches."
    ),
    compare_to: str = typer.Option("", "--compare-to", help="Run id / label / 'baseline' to diff against."),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit non-zero if the gate fails."),
    max_tasks: int = typer.Option(
        0,
        "--max-tasks",
        help="Suites to run in parallel (0 = the framework default of 1, i.e. sequential).",
    ),
    build_dashboard_after: bool = typer.Option(
        True,
        "--dashboard/--no-dashboard",
        help="Build the dashboard when the run finishes (default: yes).",
    ),
    headed: bool = typer.Option(False, "--headed", help="Watch the browser."),
    no_browser_trace: bool = typer.Option(False, "--no-browser-trace", help="Skip Playwright traces/video."),
    today: str = typer.Option("", "--today", help="Pin 'today' (YYYY-MM-DD) for date templating."),
    mock: str = typer.Option(
        "",
        "--mock",
        help="Run offline against the mock driver ('good' or 'poor'). Self-tests the harness.",
    ),
) -> None:
    """Run one or more suites against the live app and record everything."""
    from inspect_ai import eval as inspect_eval

    from .analysis import compare_refs, comparison_report, connect, gate, ingest_log
    from .tasks import suite_eval

    cfg = settings()
    if headed:
        # A request to the target, not an instruction: one that drives a browser shows it,
        # and one that does not ignores it.
        os.environ["EVAL_HEADED"] = "1"
    suites = [Path(p) for p in (suite or discover(DEFAULT_SUITE_DIR))]
    if not suites:
        console.print(f"[red]no suites found in {DEFAULT_SUITE_DIR}[/red]")
        raise typer.Exit(2)

    if mock and mock not in {"good", "poor"}:
        console.print("[red]--mock must be 'good' or 'poor'[/red]")
        raise typer.Exit(2)
    run_label = label or ("mock-" + mock if mock else "adhoc")
    run_id = _run_id(run_label)
    prov = prov_mod.capture(
        run_id=run_id,
        suite=",".join(s.stem for s in suites),
        label=run_label,
        cfg=cfg,
        root=REPO_ROOT,
    )
    console.print(prov_mod.summarize(prov))
    if not prov.is_baseline:
        console.print(
            "[yellow]This run tests uncommitted or post-baseline code — patches saved under "
            f"runs/{run_id}/provenance/[/yellow]"
        )

    os.environ["EVAL_RUN_ID"] = run_id
    os.environ["EVAL_CHANGE_ID"] = prov.change_id
    os.environ["EVAL_LABEL"] = run_label
    os.environ["EVAL_IS_BASELINE"] = "1" if prov.is_baseline else "0"

    # A filter that matches nothing in *one* suite is not an error: `--ids ar-foo` over a
    # repo with five suites legitimately matches one of them. Only a filter that matches
    # nothing *anywhere* is a mistake worth stopping for, and it gets a message naming the
    # filter rather than whichever suite happened to be alphabetically first.
    tasks = []
    skipped: list[str] = []
    for s in suites:
        try:
            tasks.append(
                suite_eval(
                    suite=str(s),
                    tags=tags or None,
                    ids=ids or None,
                    severity=severity or None,
                    epochs=epochs,
                    seed=seed,
                    simulated_user_model=simulated_user or None,
                    today=today or None,
                    mock=mock,
                    record_browser_trace=not no_browser_trace,
                )
            )
        except ValueError:
            if not (tags or ids or severity):
                raise
            skipped.append(Path(s).stem)
    if skipped:
        console.print(f"[dim]no matching cases in: {', '.join(skipped)}[/dim]")
    if not tasks:
        selectors = ", ".join(
            f"{k}={v!r}" for k, v in (("ids", ids), ("tags", tags), ("severity", severity)) if v
        )
        console.print(f"[red]no cases matched {selectors} in any of {len(suites)} suite(s)[/red]")
        raise typer.Exit(2)

    logs = inspect_eval(
        tasks,
        # The agent lives in the browser; Inspect must not try to call a model itself.
        model="mockllm/model",
        log_dir=str(cfg.run.log_dir),
        limit=limit,
        max_samples=max_samples or cfg.run.max_samples,
        # Inspect's max_samples is per TASK and max_tasks defaults to 1, so a run with one
        # task per suite executes them strictly in series. That is invisible and expensive:
        # a suite holding only a few cases x epochs cannot fill its own concurrency, its
        # last batch runs half-empty, and the next suite waits on its slowest sample.
        # Measured across nine suites: 129 minutes of agent time took 39 minutes of wall
        # clock at six-way concurrency — 55% efficiency, and the missing 45% was these
        # per-suite ramp-downs, not the agent.
        #
        # Total in-flight samples is max_tasks * max_samples, so raising this raises load
        # on the system under test; set both together.
        max_tasks=max_tasks or None,
        tags=[run_label, prov.change_id],
        metadata={
            "run_id": run_id,
            "label": run_label,
            "change_id": prov.change_id,
            "is_baseline": "1" if prov.is_baseline else "0",
            "provenance_path": str(cfg.run.runs_dir / run_id / "provenance.json"),
        },
        score_display=True,
    )

    conn = connect(cfg.run.db_path)
    for log in logs:
        result = ingest_log(conn, log, log.location or "")
        console.print(f"  ingested {result.suite}: {result.samples} sample(s), {result.scores} score(s)")
    console.print(f"\n[green]recorded run {run_id}[/green] -> {cfg.run.db_path}")
    console.print(f"transcripts: [cyan]inspect view --log-dir {cfg.run.log_dir}[/cyan]")

    failed: list[str] = []
    if compare_to:
        comparisons, notes = compare_refs(conn, compare_to, run_id)
        for note in notes:
            console.print(f"[yellow]{note}[/yellow]")
        if not comparisons:
            console.print(f"[red]nothing comparable between {compare_to!r} and this run[/red]")
            raise typer.Exit(2)

        report_text = "\n\n---\n\n".join(comparison_report(c) for c in comparisons)
        out = cfg.run.runs_dir / run_id / "comparison.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_text)
        console.print(report_text)
        console.print(f"[green]comparison written to {out}[/green]")

        failed[:] = [c.suite for c in comparisons if not gate(c)[0]]
        if failed:
            console.print(f"[red]gate failed for: {', '.join(failed)}[/red]")

    # Building the page is part of finishing a run, not a separate chore. The terminal
    # summary gives a pooled delta; the page is where per-sample flips, the noise band and
    # the trace links live, so a run whose dashboard was never built is a run nobody
    # inspected — and it was forgotten often enough to be worth doing by default.
    if build_dashboard_after:
        try:
            _build_dashboard_now(cfg, offline=True)
        except Exception as exc:
            # Never let a rendering problem discard a run that already cost minutes.
            console.print(f"[yellow]run recorded, but the dashboard failed to build: {exc}[/yellow]")
            console.print(f"[yellow]build it yourself with `{PROGRAM} dashboard --offline`[/yellow]")

    if fail_on_regression and failed:
        raise typer.Exit(1)



def _build_dashboard_now(cfg, offline: bool = True, out: Path | None = None) -> Path | None:
    """Render the dashboard for every suite with runs. Returns the path, or None if empty.

    Used by `run` for its automatic build. `dashboard` renders its own, because it also
    handles --demo and --include-mock; keep the two producing the same page by hand.
    """
    from .analysis import connect
    from .analysis.dashboard import build_dashboard, run_columns, suites_with_runs

    conn = connect(cfg.run.db_path)
    suites: dict[str, list] = {}
    for name in suites_with_runs(conn):
        runs = run_columns(conn, name, cfg.run.runs_dir, limit=12)
        if runs:
            suites[name] = runs
    if not suites:
        return None
    # `subject`, matching the `dashboard` command — the page heads itself with the tool
    # name and puts what is being evaluated beside it.
    try:
        subject = cfg.target.display_name
    except Exception:
        subject = "agent"
    html = build_dashboard(
        conn, cfg.run.runs_dir, suites, local_fonts=offline, demo=False, subject=subject, program=PROGRAM
    )
    target = out or cfg.run.runs_dir / "dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    for name, runs in suites.items():
        console.print(f"  [green]{name}[/green]: {len(runs)} evaluation(s)")
    console.print(f"-> {target}")
    return target


# --------------------------------------------------------------------- baseline
baseline_app = typer.Typer(help="The reference point for hill climbing.")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("set")
def baseline_set(note: str = typer.Option("", "--note", "-n")) -> None:
    """Pin each repo's current HEAD as the baseline to diff future runs against."""
    cfg = settings()
    repos: dict[str, str] = {}
    for repo in cfg.repos_under_test:
        sha = prov_mod._git(repo, "rev-parse", "HEAD").strip()
        if sha:
            repos[repo.name] = sha
    path = prov_mod.save_baseline(REPO_ROOT, repos, note=note)
    console.print(f"[green]baseline written to {path}[/green]")
    for name, sha in repos.items():
        console.print(f"  {name} @ {sha[:12]}")


@baseline_app.command("show")
def baseline_show() -> None:
    data = prov_mod.load_baseline(REPO_ROOT)
    if not data:
        console.print(f"[yellow]no baseline pinned — run `{PROGRAM} baseline set`[/yellow]")
        return
    for name, sha in data.items():
        console.print(f"{name} @ {sha[:12]}")


@app.command()
def diff(run_ref: str = typer.Argument("latest", help="Run id, or 'latest'.")) -> None:
    """Show the code diff a run was testing (baseline -> that run's code)."""
    cfg = settings()
    runs_dir = cfg.run.runs_dir
    if run_ref == "latest":
        candidates = sorted((p for p in runs_dir.glob("*/provenance.json")), key=lambda p: p.stat().st_mtime)
        if not candidates:
            console.print("[yellow]no runs recorded yet[/yellow]")
            raise typer.Exit(1)
        path = candidates[-1]
    else:
        path = runs_dir / run_ref / "provenance.json"
    if not path.exists():
        console.print(f"[red]no provenance at {path}[/red]")
        raise typer.Exit(1)
    data = json.loads(path.read_text())
    console.print(f"run {data['run_id']}  change {data['change_id']}")
    for repo in data["repos"]:
        console.print(f"\n[bold]{repo['name']}[/bold] {repo.get('branch')}@{(repo.get('head_sha') or '')[:12]}")
        if repo.get("diff_stat"):
            console.print(repo["diff_stat"])
        for key in ("committed_patch_file", "worktree_patch_file"):
            if repo.get(key):
                console.print(f"  patch: {path.parent / repo[key]}")
        for commit in repo.get("commits_ahead", []):
            console.print(f"  + {commit}")


# ------------------------------------------------------------------- inspection
@app.command()
def suites(suite_dir: Path = typer.Option(DEFAULT_SUITE_DIR, "--dir")) -> None:
    """List suites and their cases."""
    cfg = settings()
    table = Table("suite", "v", "case", "tags", "severity", "expectations")
    for path in discover(suite_dir):
        loaded = load_suite(path, base_seed=cfg.run.seed, months_back=cfg.run.data_months_back)
        for case in loaded.cases:
            expect = case.expect
            checks = []
            if expect.rubric:
                checks.append("rubric")
            if expect.must_contain or expect.must_not_contain or expect.must_approx:
                checks.append("assertions")
            if expect.tools.specified:
                checks.append("tools")
            if expect.max_steps or expect.max_latency_ms:
                checks.append("budget")
            if expect.must_ask is not None:
                checks.append("must_ask")
            table.add_row(
                loaded.name, str(loaded.version), case.id, ",".join(case.tags), case.severity, ",".join(checks)
            )
    console.print(table)


@app.command()
def render(
    suite: Path = typer.Argument(..., help="Suite YAML to render."),
    seed: int = typer.Option(None, "--seed"),
    today: str = typer.Option("", "--today"),
) -> None:
    """Show the exact prompts a run would send. The way to sanity-check templating."""
    from datetime import date

    cfg = settings()
    loaded = load_suite(
        suite,
        base_seed=seed if seed is not None else cfg.run.seed,
        today=date.fromisoformat(today) if today else None,
        months_back=cfg.run.data_months_back,
    )
    for case in loaded.cases:
        console.print(f"\n[bold cyan]{case.id}[/bold cyan]  seed={case.seed}")
        console.print(f"  template: [dim]{case.prompt_template}[/dim]")
        console.print(f"  prompt:   {case.prompt}")
        if case.bindings:
            console.print(f"  bindings: {case.bindings}")


# ---------------------------------------------------------------------- results
@app.command()
def ingest(log_dir: Path = typer.Option(None, "--log-dir")) -> None:
    """(Re)build the results database from the eval logs on disk."""
    from .analysis import connect, ingest_dir

    cfg = settings()
    conn = connect(cfg.run.db_path)
    results = ingest_dir(conn, log_dir or cfg.run.log_dir)
    for r in results:
        console.print(f"{r.run_id} [{r.suite}]: {r.samples} sample(s), {r.scores} score(s)")
    console.print(f"[green]{len(results)} log(s) ingested into {cfg.run.db_path}[/green]")


@app.command()
def compare(
    baseline: str = typer.Argument("baseline", help="Run id, label, 'baseline' or 'latest'."),
    candidate: str = typer.Argument("latest", help="Run id, label, 'baseline' or 'latest'."),
    out: Path = typer.Option(None, "--out", help="Write the markdown report here."),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression"),
) -> None:
    """Diff two runs, paired per sample, with confidence intervals — one section per suite."""
    from .analysis import compare_refs, comparison_report, connect, gate

    cfg = settings()
    conn = connect(cfg.run.db_path)
    comparisons, notes = compare_refs(conn, baseline, candidate)
    for note in notes:
        console.print(f"[yellow]{note}[/yellow]")
    if not comparisons:
        console.print(
            f"[red]nothing comparable between {baseline!r} and {candidate!r} — try `{PROGRAM} runs`[/red]"
        )
        raise typer.Exit(2)

    report_text = "\n\n---\n\n".join(comparison_report(c) for c in comparisons)
    console.print(report_text)
    if out:
        out.write_text(report_text)
        console.print(f"[green]written to {out}[/green]")
    failed = [c.suite for c in comparisons if not gate(c)[0]]
    if fail_on_regression and failed:
        raise typer.Exit(1)


@app.command()
def runs(limit: int = typer.Option(20, "--limit"), suite: str = typer.Option("", "--suite")) -> None:
    """List recorded runs, newest first."""
    from .analysis import connect, latest_runs

    cfg = settings()
    conn = connect(cfg.run.db_path)
    table = Table("run_id", "suite", "label", "change", "epochs", "samples", "status", "when")
    for row in latest_runs(conn, suite=suite or None, limit=limit):
        table.add_row(
            row["run_id"],
            row["suite"],
            row["label"] or "",
            (row["change_id"] or "")[:10] + ("*" if row["is_baseline"] else ""),
            str(row["epochs"]),
            str(row["samples"]),
            row["status"],
            (row["created_at"] or "")[:16],
        )
    console.print(table)
    console.print("[dim]* = run against the pinned baseline code[/dim]")


@app.command()
def report(
    run_ref: str = typer.Argument("latest"),
    suite: str = typer.Option("", "--suite", help="Limit to one suite."),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Full report for a single run — one section per suite it covered."""
    from .analysis import connect, resolve_run, run_report

    cfg = settings()
    conn = connect(cfg.run.db_path)
    row = resolve_run(conn, run_ref, suite=suite or None)
    if row is None:
        console.print(f"[red]no run matching {run_ref!r}[/red]")
        raise typer.Exit(2)
    text = run_report(conn, row["run_id"], suite=suite or None)
    console.print(text)
    if out:
        out.write_text(text)


@app.command()
def trend(
    scorer: str = typer.Option("rubric_judge", "--scorer"),
    suite: str = typer.Option("", "--suite"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Score per run over time — are we climbing?"""
    from .analysis import connect, trend_report

    cfg = settings()
    conn = connect(cfg.run.db_path)
    console.print(trend_report(conn, suite=suite or None, scorer=scorer, limit=limit))


@app.command()
def view(log_dir: Path = typer.Option(None, "--log-dir")) -> None:
    """Open the Inspect log viewer (full agentic transcripts, per sample)."""
    import subprocess

    cfg = settings()
    subprocess.run([sys.executable, "-m", "inspect_ai._view.view", "--log-dir", str(log_dir or cfg.run.log_dir)])



# ----------------------------------------------------------------------- doctor
@app.command()
def doctor() -> None:
    """Preflight the environment. Run this before believing any numbers.

    Two halves: what the framework needs to measure anything (a pinned judge, credentials,
    enough epochs, the repos it diffs) and what the *target* needs to be reachable at all
    (its app, its session, its tenant). The target's rows come from the target, so a new
    one gets a real preflight without touching this file.
    """
    import importlib.util
    import shutil

    cfg = settings()
    checks: list[Check] = []

    try:
        target = cfg.target
        checks += target.doctor_checks()
    except Exception as exc:
        target = None
        checks.append(Check("target loads", False, f"EVAL_TARGET={target_ref()!r}: {exc}"))

    if BORROWED_SECRETS:
        checks.append(
            Check("borrowed judge credentials", True, f"{', '.join(BORROWED_SECRETS)} from {BORROWED_FROM}")
        )

    judge_provider = cfg.judge.model.split("/")[0]
    key_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(judge_provider, "")
    if "-latest" in cfg.judge.model or cfg.judge.model.count("-") < 2:
        checks.append(
            Check(
                "judge model is pinned",
                False,
                f"{cfg.judge.model} looks like a moving alias — pin a dated snapshot so the "
                "metric does not change under you",
            )
        )
    checks.append(
        Check(
            f"judge credentials ({cfg.judge.model})",
            bool(os.environ.get(key_env)) if key_env else True,
            key_env or "no key needed",
        )
    )

    # The provider client is an install extra, because which one you need depends on the
    # judge you pin. A missing one otherwise shows up only mid-run, as every rubric score
    # excluded for "judge unreachable" — an expensive way to learn about a pip install.
    provider_package = {"anthropic": "anthropic", "openai": "openai"}.get(judge_provider, "")
    if provider_package:
        checks.append(
            Check(
                f"judge provider package ({provider_package})",
                importlib.util.find_spec(provider_package) is not None,
                'installed' if importlib.util.find_spec(provider_package) else 'pip install ".[judges]"',
            )
        )

    for repo in cfg.repos_under_test:
        checks.append(Check(f"repo {repo.name}", (repo / ".git").exists(), str(repo)))
    pinned = prov_mod.load_baseline(REPO_ROOT)
    checks.append(
        Check(
            "baseline pinned",
            bool(pinned),
            f"run `{PROGRAM} baseline set`" if not pinned else ", ".join(pinned),
        )
    )
    inspect_bin = Path(sys.executable).parent / "inspect"
    checks.append(
        Check(
            "inspect view available",
            inspect_bin.exists() or shutil.which("inspect") is not None,
            str(inspect_bin) if inspect_bin.exists() else "pip install inspect-ai",
        )
    )

    if cfg.run.epochs < 3:
        checks.append(
            Check("epochs >= 3", False, f"epochs={cfg.run.epochs}: too few repeats to separate signal from noise")
        )

    table = Table("check", "status", "detail")
    for check in checks:
        table.add_row(check.name, "[green]ok[/green]" if check.ok else "[red]FAIL[/red]", check.detail)
    console.print(table)

    # Not a failure: the fallback names still work. But a name that is quietly deprecated
    # in everyone's .env is a name nobody migrates, so say it once, here.
    if LEGACY_ENV_USED:
        console.print(
            f"[yellow]read under a fallback prefix: {', '.join(sorted(LEGACY_ENV_USED))} — "
            f"the framework's own knobs are {ENV_PREFIXES[0]}_*[/yellow]"
        )
    if target is not None:
        console.print(f"[dim]target: {target.display_name} ({target_ref()})[/dim]")

    if not all(c.ok for c in checks):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


def _open_in_browser(path: Path) -> bool:
    """Open a local file, reporting whether it worked.

    `webbrowser.open` returns True on some platforms without having done anything, so the
    platform opener is tried first and its exit status is what we trust.
    """
    uri = path.resolve().as_uri()
    opener = {"darwin": "open", "linux": "xdg-open"}.get(sys.platform)
    if opener and shutil.which(opener):
        return subprocess.run([opener, str(path.resolve())], check=False).returncode == 0
    return webbrowser.open(uri)


@app.command()
def dashboard(
    suite: str = typer.Option("", "--suite", help="One suite (default: every suite with runs)."),
    limit: int = typer.Option(12, "--limit", help="How many runs to show, newest kept."),
    out: Path = typer.Option(None, "--out", help="Where to write the HTML (default: runs/dashboard-<suite>.html)."),
    include_mock: bool = typer.Option(False, "--include-mock", help="Also show offline mock runs."),
    offline: bool = typer.Option(
        False, "--offline", help="Drop the webfont link so the page makes no network requests."
    ),
    open_page: bool = typer.Option(False, "--open", help="Open the page in your browser afterwards."),
    demo: bool = typer.Option(False, "--demo", help="Render the synthetic demo database instead."),
) -> None:
    """Build the hill-climb console: a static page per suite, read from the results database.

    No model is involved — this is a deterministic render of stored rows, the saved
    provenance patches, and `git diff` between each pair of runs. Regenerate it as often
    as you like; the same database produces the same page.

    The output is a local file and is meant to stay local: it carries eval results,
    internal identifiers and source patches. `--offline` also strips the webfont link so
    the page makes no network requests when opened.
    """
    from .analysis import connect
    from .analysis.dashboard import build_dashboard, run_columns, suites_with_runs

    cfg = settings()
    conn = connect(cfg.run.runs_dir / "demo.db" if demo else cfg.run.db_path)
    targets = [suite] if suite else suites_with_runs(conn)
    if not targets:
        console.print(f"[yellow]no evaluations recorded yet — run `{PROGRAM} run` first[/yellow]")
        raise typer.Exit(1)

    # One page, one tab per suite. Suites are never merged — their samples differ, so a
    # pooled number would hide a gain in one behind a loss in the other.
    suites: dict[str, list] = {}
    for name in targets:
        runs = run_columns(conn, name, cfg.run.runs_dir, limit=limit, include_mock=include_mock)
        if runs:
            suites[name] = runs
        else:
            console.print(f"[yellow]no evaluations for suite {name!r}[/yellow]")
    if not suites:
        raise typer.Exit(1)

    # The page is headed by the tool; beside it goes whatever the page is actually
    # showing. A demo page carrying the real agent's name is the one mistake this whole
    # demo is built to avoid, so a demo says what it is instead.
    try:
        subject = cfg.target.display_name
    except Exception:
        subject = "agent"
    if demo:
        from .analysis.demo_blog import BLOG_SCENARIO

        try:
            scenario = cfg.target.demo_scenario() or BLOG_SCENARIO
        except Exception:
            scenario = BLOG_SCENARIO
        subject = f"demo — {scenario.description}"
    html = build_dashboard(
        conn, cfg.run.runs_dir, suites, local_fonts=offline, demo=demo, subject=subject, program=PROGRAM
    )
    target = out or cfg.run.runs_dir / ("demo-dashboard.html" if demo else "dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    for name, runs in suites.items():
        console.print(f"  [green]{name}[/green]: {len(runs)} evaluation(s)")
    console.print(f"-> {target}")
    if open_page and not _open_in_browser(target):
        console.print(f"[yellow]could not open a browser — open it yourself:[/yellow] {target}")


@app.command("demo-data")
def demo_data() -> None:
    """Fabricate a history of evaluations so the dashboard can be built without real runs.

    Written to its own database (`runs/demo.db`) and never into the real store: fake
    numbers that could be mistaken for real ones are worse than no numbers. For the same
    reason the built-in history is about a blog-writing assistant rather than about the
    agent you are actually evaluating. Render it with `<program> dashboard --demo`.
    """
    from .analysis.demo import build
    from .analysis.demo_blog import BLOG_SCENARIO

    cfg = settings()
    try:
        scenario = cfg.target.demo_scenario() or BLOG_SCENARIO
    except Exception:
        scenario = BLOG_SCENARIO
    db, count = build(cfg.run.runs_dir, cfg.run.runs_dir / "demo.db", scenario)
    console.print(f"[green]{count} demo evaluation(s)[/green] about {scenario.description} -> {db}")
    console.print("[yellow]synthetic data — every evaluation is labelled DEMO[/yellow]")
    console.print(f"render it: [cyan]{PROGRAM} dashboard --demo --offline --open[/cyan]")


@baseline_app.command("use")
def baseline_use(
    evaluation: str = typer.Argument(..., help="Evaluation id, or a label (newest wins), or 'latest'."),
) -> None:
    """Nominate the evaluation that comparisons default to.

    Written to baseline.json, which both `compare` and the dashboard read — so the page and
    the terminal can never disagree about what the baseline is.
    """
    from .analysis import connect, resolve_runs

    cfg = settings()
    conn = connect(cfg.run.db_path)
    rows = resolve_runs(conn, evaluation)
    if not rows:
        console.print(f"[red]no evaluation matching {evaluation!r} — try `{PROGRAM} runs`[/red]")
        raise typer.Exit(2)
    run_id = rows[0]["run_id"]
    path = prov_mod.set_baseline_evaluation(REPO_ROOT, run_id)
    console.print(f"[green]baseline evaluation set to {run_id}[/green] ({rows[0]['label'] or 'no label'})")
    console.print(f"  suites: {', '.join(sorted({r['suite'] for r in rows}))}")
    console.print(f"  written to {path}")
