"""The shape of a demo history.

``demo-data`` fabricates a hill climb so the dashboard can be built, read and
argued with before anybody spends real runs on it. The *machinery* for that — writing the
database, the provenance, the traces, a throwaway git repo with real commits — belongs to
the framework and lives in :mod:`evalkit.analysis.demo`. The *content* does not: sample
ids, prompts, answers and rubric criteria describe some particular agent doing some
particular job.

So the content is a :class:`DemoScenario`. The framework ships one about a blog-writing
assistant (:mod:`evalkit.analysis.demo_blog`), deliberately unrelated to whatever you are
actually evaluating — a demo that reads like your own product invites exactly the
confusion the DEMO labels exist to prevent. A target may supply its own via
``Target.demo_scenario()``.

The scores are the argument, not the prose. A scenario should contain, at minimum: two
suites (so the page can show that suites are never pooled), one variation measured twice
(which is the only way a noise floor exists), a variation that is only a model change, a
clean win, and a regression that a rising average would hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DemoScenario:
    """Everything about a demo history that is not machinery.

    ``plan`` is one entry per evaluation, in time order::

        {"label": "ask-before-publishing",
         "variation": "v2-ask-first",          # evaluations sharing one tested the same thing
         "commit": ("prompt.py", "<new content>", "<commit message>") | None,
         "model": "GPT_5_4",
         "scores": {sample_id: [per-epoch rubric score, ...]}}

    ``scores_2`` gives the second suite's scores per variation, so one change can help one
    suite and hurt the other — the case a pooled number would hide.
    """

    #: What the demo agent does, for the page and the CLI ("a blog-writing assistant").
    description: str

    suite: str
    samples: list[str]
    plan: list[dict]

    suite_2: str
    samples_2: list[str]
    scores_2: dict[str, dict[str, list[float]]]

    #: The conversation, per sample id.
    prompts: dict[str, str] = field(default_factory=dict)
    good_answers: dict[str, str] = field(default_factory=dict)
    weak_answers: dict[str, str] = field(default_factory=dict)
    #: Samples where asking first is the correct behaviour, and what the user says back.
    clarify: dict[str, str] = field(default_factory=dict)
    user_replies: dict[str, str] = field(default_factory=dict)

    #: The process, per sample id: observed step identities, and who it delegated to.
    steps_by_sample: dict[str, list[str]] = field(default_factory=dict)
    subagent_by_sample: dict[str, str] = field(default_factory=dict)
    #: The step the `tool_calls` panel reports on.
    primary_step: str = "link:search"

    #: Rubric criterion ids per sample, so the Result view has real names to show.
    criteria: dict[str, list[str]] = field(default_factory=dict)
    #: What a failed `assertions` score says was missing.
    missing_assertion: str = "a required phrase"

    judge_model: str = "openai/gpt-5.5-2026-04-23"
    suite_sha: str = "demo-suite-01"
    #: Stands in for the target configuration a real run records. Obviously fake on
    #: purpose: a demo carrying a real tenant id is a demo that can be mistaken for a run.
    target_config: dict[str, str] = field(
        default_factory=lambda: {
            "base_url": "http://demo.invalid",
            "organization": "demo-organization",
        }
    )

    @property
    def all_samples(self) -> list[str]:
        return self.samples + self.samples_2
