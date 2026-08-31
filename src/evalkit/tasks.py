"""Inspect task definitions.

One task per suite file. The task carries the suite's identity (name, version, content
hash), the target it ran against, and the run's provenance in its metadata, so an
``.eval`` log is self-describing: you can pick one up months later and know which code,
which prompts, which tenant and which judge produced it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from inspect_ai import Epochs, Task, task

from .config import Settings, env, settings
from .datasets import load_suite
from .mock import MockTarget
from .scorers import NO_JUDGE, default_scorers
from .solver import agent_conversation

DEFAULT_SUITE = "evals/suites/receivables.yaml"


@task
def suite_eval(
    suite: str = DEFAULT_SUITE,
    tags: str | None = None,
    ids: str | None = None,
    severity: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    simulated_user_model: str | None = None,
    today: str | None = None,
    mock: str = "",
    record_browser_trace: bool = True,
) -> Task:
    """Run one suite against the agent under test.

    ``epochs`` matters more than it looks: a browser-driven agent is stochastic, and a
    single pass cannot distinguish a real 5% improvement from noise. The default runs
    each sample several times and reports the mean, so the comparison in
    ``compare`` has a variance estimate to work with.
    """
    cfg: Settings = settings()
    # `--mock` swaps in the framework's offline target: the same loop, scorers and reports
    # with no app, no browser and no judge. Used to self-test the harness and to confirm
    # the comparison machinery can see a planted improvement.
    target = MockTarget(quality=mock) if mock else cfg.target
    base_seed = seed if seed is not None else cfg.run.seed
    suite_path = Path(suite)
    if not suite_path.is_absolute():
        suite_path = Path.cwd() / suite_path

    loaded = load_suite(
        suite_path,
        base_seed=base_seed,
        today=date.fromisoformat(today) if today else None,
        months_back=cfg.run.data_months_back,
    )
    selected = loaded.filter(
        tags=[t.strip() for t in tags.split(",")] if tags else None,
        ids=[i.strip() for i in ids.split(",")] if ids else None,
        severity=severity,
    )
    if not selected.cases:
        raise ValueError(f"no cases selected from {suite_path} (tags={tags}, ids={ids}, severity={severity})")

    n_epochs = epochs if epochs is not None else cfg.run.epochs

    # Run identity comes from the environment (the CLI sets it before building tasks).
    # Empty values are dropped rather than emitted: task metadata takes precedence over
    # the eval-level metadata, so an empty string here would blank out a label that the
    # caller passed to `eval(metadata=...)`.
    run_identity = {
        key: value
        for key, value in (
            ("run_id", env("RUN_ID")),
            ("change_id", env("CHANGE_ID")),
            ("label", env("LABEL")),
            ("is_baseline", env("IS_BASELINE")),
        )
        if value
    }

    return Task(
        name=f"{target.name}-{selected.name}",
        dataset=selected.to_dataset(),
        solver=agent_conversation(
            cfg=cfg,
            target=target,
            simulated_user_model=simulated_user_model,
            record_trace=record_browser_trace,
        ),
        # A mock run exercises the harness, not the agent: never spend judge calls on it.
        scorer=default_scorers(judge_model=NO_JUDGE if mock else None),
        # 'mean' keeps partial credit across epochs; 'max' answers "can it ever do this?",
        # which is the right question when debugging a newly added capability.
        epochs=Epochs(n_epochs, ["mean", "max"]),
        # Sample-level errors are expected (an app can 500); abandon the run only if a
        # fifth of the suite is failing, which means the environment is broken.
        fail_on_error=0.2,
        metadata={
            "suite": selected.name,
            "suite_version": selected.version,
            "suite_sha": selected.content_sha,
            "suite_path": str(suite_path),
            "base_seed": base_seed,
            "data_months_back": cfg.run.data_months_back,
            "epochs": n_epochs,
            "judge_model": NO_JUDGE if mock else cfg.judge.model,
            "target": target.name,
            # Flattened alongside the rest so `compare` and the dashboard can warn
            # when the tenant or the app moved between two evaluations.
            **{k: v for k, v in target.fingerprint().items() if isinstance(v, str)},
            **run_identity,
            "simulated_user_model": simulated_user_model or "",
            "mock": mock,
        },
        tags=[target.name, selected.name] + ([f"mock:{mock}"] if mock else []),
    )
