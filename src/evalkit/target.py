"""The seam between the eval framework and the thing being evaluated.

Everything in ``evalkit`` is about *measuring* an agent: suites, templating, the
simulated user, the scorers, the paired statistics, the record. None of it knows how to
reach an agent, log into it, or read its tool calls off the wire. That knowledge lives
behind two protocols defined here, and a **target** package supplies them.

The rule the split enforces: *no module under ``evalkit`` may import a target.* A target
is resolved by name at runtime (see :mod:`evalkit.registry`), so the framework can be
pointed at another product without editing it.

A target owns four things the framework cannot know:

1. **How to open a conversation** — auth, navigation, and one :class:`ChatDriver`.
2. **How to read a turn** — turning whatever the product emits (SSE, websocket, an API
   response) into the normalized :class:`~evalkit.trace.ToolCall` /
   ``Subagent`` / ``Interrupt`` vocabulary the scorers grade.
3. **What counts as broken** — which HTTP failures invalidate a sample, and which of the
   product's own error strings mean "the agent gave up".
4. **What must be recorded** — the configuration whose change makes two evaluations
   incomparable (tenant, base URL, feature toggles).

And two pieces of *domain*, which are not plumbing but are just as product-specific: the
vocabulary templated prompts draw from, and what the judge is told it is grading. It may
also supply its own demo history, though the built-in one is deliberately about something
else entirely — see :mod:`evalkit.scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import typer

from .trace import DEFAULT_FAILURE_MARKERS, Interrupt, SseEvent, Subagent, ToolCall, Turn, agent_failure_in


class HarnessError(RuntimeError):
    """A failure of the harness or the app — never of the agent.

    Raised by a target when it cannot reach, authenticate against, or drive the product.
    The framework records it as an :class:`~evalkit.trace.InfraError` and *excludes* the
    sample from quality metrics, because a suite that scores infra flakiness cannot be
    hill-climbed. ``kind`` is what shows up in the record and on the dashboard.
    """

    kind = "harness"


@dataclass
class SampleContext:
    """Which sample a conversation is being opened for.

    Passed to :meth:`Target.open` because a target may legitimately need it — to pick a
    tenant per case, to replay a fixture, to name an artifact. Most ignore it.
    """

    sample_id: str
    epoch: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    """One agent turn, normalized.

    This is the whole contract for "what the agent did": every scorer reads the trace
    assembled from these, so a target that fills in ``tool_calls`` and ``subagents``
    honestly gets process grading for free, and one that only fills in ``text`` still
    gets output grading.
    """

    index: int
    text: str = ""
    # What a human would have seen, when that differs from the structured answer (DOM
    # text, say). Used as a fallback and shown in the transcript.
    rendered_text: str = ""
    trace_id: str | None = None
    latency_ms: int = 0
    events: list[SseEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    subagents: list[Subagent] = field(default_factory=list)
    interrupt: Interrupt | None = None
    tables: int = 0
    plots: int = 0
    tokens_streamed: int = 0
    first_token_ms: int | None = None
    errored: bool = False
    dag: dict[str, Any] | None = None

    @property
    def asks_question(self) -> bool:
        """Did the agent hand control back to the user?

        Either a formal approval interrupt, or a plain question in the prose — both mean
        the next thing that should happen is a user reply, which is what makes multi-turn
        evals necessary in the first place.
        """
        if self.interrupt is not None:
            return True
        tail = self.text.strip()[-400:]
        return "?" in tail


def turn_records(results: list[TurnResult], user_messages: list[tuple[int, str, str]]) -> list[Turn]:
    """Interleave user and assistant turns into a flat transcript for the record.

    Shared by every driver: the interleaving is the framework's business, not the
    target's. ``user_messages`` is ``(turn index, text, origin)``.
    """
    turns: list[Turn] = []
    by_index = {r.index: r for r in results}
    for index, text, origin in user_messages:
        turns.append(Turn(index=index, role="user", text=text, origin=origin))
        result = by_index.get(index)
        if result:
            turns.append(
                Turn(
                    index=index,
                    role="assistant",
                    text=result.text or result.rendered_text,
                    latency_ms=result.latency_ms,
                    trace_id=result.trace_id,
                    tables=result.tables,
                    plots=result.plots,
                )
            )
    return turns


@runtime_checkable
class ChatDriver(Protocol):
    """One conversation with the agent under test.

    The solver calls exactly these. A driver is used for a single sample and then closed
    by its target; it may hold a browser page, an HTTP session, or nothing at all.
    """

    #: Set once the conversation exists, if the product has such an identifier.
    chat_id: str | None

    async def start(self) -> None:
        """Get ready to talk (navigate, wait for the app to settle)."""

    async def new_chat(self) -> None:
        """Begin a fresh conversation, so no sample inherits another's context."""

    async def send(self, text: str) -> TurnResult:
        """Send one user message and wait for the agent's turn to finish."""

    async def approve_interrupt(self) -> TurnResult | None:
        """Approve a pending approval request the way a user would.

        Returns the agent turn the approval kicked off, or ``None`` when there was no
        formal card to click (the agent asked in prose, so it needs a prose answer).
        """

    async def agent_config(self) -> dict[str, Any]:
        """The agent selection in effect: model, sub-agents, overrides.

        Configuration under test. Two runs with different selections are not comparable
        and usually nothing visible says so.
        """

    async def diagnostics(self) -> dict[str, Any]:
        """Health observations for the conversation.

        ``httpErrors`` (each with ``status``/``url``, optionally ``turn``) is the only key
        the framework reads; anything else is kept in the record for a human.
        """


@dataclass
class Check:
    """One ``doctor`` row."""

    name: str
    ok: bool
    detail: str = ""


@runtime_checkable
class Target(Protocol):
    """A product the framework can evaluate."""

    #: Stable identifier, recorded with every run and used to name Inspect tasks.
    name: str
    #: Human-readable, for page titles and headings.
    display_name: str

    async def open(self, sample: SampleContext, artifacts_dir: Path, record_trace: bool = True) -> ChatDriver:
        """Open one conversation. Raise :class:`HarnessError` if the app is unreachable."""

    async def close(self, driver: ChatDriver, *, save_trace: bool, failed: bool) -> dict[str, str]:
        """Tear the conversation down; return any artifacts written (path by name)."""

    async def capture_failure(self, driver: ChatDriver, name: str) -> dict[str, str]:
        """Capture whatever helps diagnose a failure (screenshot, DOM, logs)."""

    def fingerprint(self) -> dict[str, Any]:
        """Configuration that decides whether two evaluations are comparable.

        Stored in the run's provenance and diffed by the dashboard. Anything that changes
        the *correct answer* belongs here — the tenant above all.
        """

    def default_repos(self) -> tuple[Path, ...]:
        """Repos whose git state is this target's code under test."""

    def classify_http_error(self, status: int, url: str) -> str | None:
        """Return an infra-error kind, or ``None`` when the failure is benign.

        Narrow by default: marking every failed request as infra can drop a whole suite's
        coverage to zero, which is the exact failure the exclusion mechanism exists to
        prevent.
        """

    def agent_failure_in(self, text: str) -> str | None:
        """Return the marker matched if this is the product's own failure message."""

    def tool_aliases(self) -> dict[str, str]:
        """Machine name -> display name, so suites need not know which they will see."""

    def vocabulary(self) -> dict[str, list[str]]:
        """Named word lists for templated prompts.

        ``{"vendor": [...]}`` gives a suite ``${randomVendor()}``. A vendor list, a
        currency list, a set of plausible people are the product's world, not the
        framework's — and a prompt drawing from the wrong world is a prompt about nothing.
        """

    def resolve_expected(self, spec: dict, bindings: dict) -> str | None:
        """Compute a figure's expected value independently of the agent, or None.

        The framework can compare numbers; only the target knows what "the AP aging total
        as of this date" is, or how to obtain it by a path the agent under test does not
        use. Returning None means unscoreable rather than wrong — a missing adjudicator is
        a fixture problem and must never be reported as a failing agent.
        """
        return None

    def judge_domain(self) -> str:
        """What the judge is told it is grading ("an AI accounting assistant").

        The judge prompt is the metric. A grader told the wrong domain applies the wrong
        standard of care, so this belongs to the target rather than to a default.
        """

    def demo_scenario(self):  # -> evalkit.scenario.DemoScenario | None
        """A fabricated history for ``demo-data``, or ``None`` for the built-in one.

        Most targets should return ``None``. The framework's demo is about a blog-writing
        assistant precisely because it is *not* the product under test: a demo history that
        looks like your own results is one somebody eventually screenshots into a decision.
        """

    def doctor_checks(self) -> list[Check]:
        """Preflight rows: is this target reachable, authenticated and pinned?"""

    def cli(self) -> typer.Typer | None:
        """Extra commands (login, probe), mounted under the CLI."""


class BaseTarget:
    """Sensible do-nothing defaults, so a target only implements what it has.

    Subclassing is optional — :class:`Target` is a protocol — but it keeps a minimal
    target to the three methods that actually matter (``open``, ``close``, ``fingerprint``).
    """

    name = "target"
    display_name = "Agent"
    #: Product strings that mean "the agent gave up" without the stream saying so.
    failure_markers: tuple[str, ...] = DEFAULT_FAILURE_MARKERS

    async def open(self, sample: SampleContext, artifacts_dir: Path, record_trace: bool = True) -> ChatDriver:
        raise NotImplementedError

    async def close(self, driver: ChatDriver, *, save_trace: bool, failed: bool) -> dict[str, str]:
        return {}

    async def capture_failure(self, driver: ChatDriver, name: str) -> dict[str, str]:
        return {}

    def fingerprint(self) -> dict[str, Any]:
        return {}

    def default_repos(self) -> tuple[Path, ...]:
        return ()

    def classify_http_error(self, status: int, url: str) -> str | None:
        return None

    def agent_failure_in(self, text: str) -> str | None:
        return agent_failure_in(text, self.failure_markers)

    def tool_aliases(self) -> dict[str, str]:
        return {}

    def vocabulary(self) -> dict[str, list[str]]:
        return {}

    def resolve_expected(self, spec: dict, bindings: dict) -> str | None:
        """No independent source of truth by default. See the protocol for the contract."""
        return None

    def judge_domain(self) -> str:
        return "an AI assistant"

    def demo_scenario(self):
        return None

    def doctor_checks(self) -> list[Check]:
        return []

    def cli(self) -> typer.Typer | None:
        return None
