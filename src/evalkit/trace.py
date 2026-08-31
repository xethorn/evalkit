"""A normalized agentic trace for one eval sample.

This is the framework's vocabulary for "what the agent did", and the only thing the
scorers grade. It is deliberately product-agnostic: a target
(:mod:`evalkit.target`) observes its own product however it can — a tee\'d event stream,
a backend trace endpoint, what was rendered on screen — and reports the result in these
types. Adding a product means writing that translation, not touching a scorer.

Everything is JSON-serializable so it can be attached to the Inspect log and replayed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["started", "completed", "failed", "unknown"]


class ToolCall(BaseModel):
    """One observed step of the agent's work.

    ``name`` is the *machine-readable* identity, and that distinction is the point.
    Products often label a step with an LLM-written sentence ("Checked AP expense records
    for May 2025 and found no matching activity"), which is unusable as an assertion — the
    wording changes run to run. A target must derive a stable identity for ``name`` and
    say where it came from in ``kind``; the prose belongs in ``detail``, so a human reading
    the trace still sees what happened.
    """

    name: str
    subagent: str | None = None
    status: ToolStatus = "unknown"
    turn: int = 0
    # Which of the target's views reported this call. Free-form, recorded so a suite
    # author can tell a streamed observation from a reconstructed one.
    source: str = "progress"
    detail: str | None = None
    started_ms: int | None = None
    # Where the identity came from, so a suite author knows what they can assert on.
    kind: Literal["tool", "link", "endpoint", "connector", "unnamed"] = "tool"

    @property
    def key(self) -> str:
        return self.name.strip().lower()


class Subagent(BaseModel):
    """A sub-agent the orchestrator delegated to.

    Routing is often the thing you actually want to assert: an expense question should
    reach the Accounts Payable agent, and a change that silently reroutes it elsewhere is
    a regression no output-quality score will show you.
    """

    name: str
    turn: int = 0
    source: str = "progress"
    summary: str | None = None


class Interrupt(BaseModel):
    """A human-in-the-loop approval request (the agent asking before it acts)."""

    turn: int
    tool: str | None = None
    title: str | None = None
    fields: list[str] = Field(default_factory=list)
    decision: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SseEvent(BaseModel):
    turn: int
    at_ms: int
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    index: int
    role: Literal["user", "assistant"]
    text: str = ""
    # Why this user message exists: the opening prompt, a scripted answer, or the
    # simulated user reacting to a question. Recorded so multi-turn runs are auditable.
    origin: str | None = None
    latency_ms: int | None = None
    trace_id: str | None = None
    tables: int = 0
    plots: int = 0


class InfraError(BaseModel):
    """A harness/app failure, as opposed to the agent answering badly.

    Kept separate so infra flakiness never lands in the quality metric.
    """

    kind: str
    message: str
    turn: int | None = None


class AgenticTrace(BaseModel):
    sample_id: str = ""
    chat_id: str | None = None
    prompt: str = ""
    turns: list[Turn] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    subagents: list[Subagent] = Field(default_factory=list)
    interrupts: list[Interrupt] = Field(default_factory=list)
    sse_events: list[SseEvent] = Field(default_factory=list)
    dags: dict[str, dict[str, Any]] = Field(default_factory=dict)
    infra_errors: list[InfraError] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    # The agent selection actually in effect: model, sub-agents, overrides. This is
    # configuration under test — two runs with different selections are not comparable,
    # and usually nothing visible tells you they differ.
    agent_config: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # Turns where the agent itself reported failure. Distinct from infra_errors: the app
    # worked, the agent answered, and the answer was "I couldn't". That is a real quality
    # failure (so it is scored, not excluded) but it must be visible as its own rate —
    # an agent that starts erroring 20% of the time looks like a mild score dip otherwise.
    agent_errors: list[str] = Field(default_factory=list)
    # True when the turn budget ran out with the agent still asking for something.
    unconverged: bool = False
    total_ms: int = 0
    truncated_events: int = 0

    # -- convenience views used by scorers ---------------------------------
    @property
    def final_answer(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "assistant" and turn.text.strip():
                return turn.text
        return ""

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def tool_names(self) -> list[str]:
        """Tool names in call order, de-duplicated only on consecutive repeats."""
        out: list[str] = []
        for call in self.tool_calls:
            if not out or out[-1] != call.key:
                out.append(call.key)
        return out

    @property
    def tool_name_set(self) -> set[str]:
        return {c.key for c in self.tool_calls}

    @property
    def subagent_names(self) -> list[str]:
        out: list[str] = []
        for agent in self.subagents:
            if agent.name not in out:
                out.append(agent.name)
        return out

    @property
    def failed(self) -> bool:
        return bool(self.infra_errors)

    def count(self, tool: str) -> int:
        return sum(1 for c in self.tool_calls if c.key == tool.strip().lower())

    def transcript_text(self) -> str:
        return "\n\n".join(f"{t.role.upper()}: {t.text}" for t in self.turns if t.text.strip())


# Things a product tends to say when a turn fails without the stream emitting an `error`
# event. Observed verbatim in a baseline run: the agent returned "I was unable to generate
# a response. Please try again." and the harness scored it as a normal answer.
#
# These are the generic ones. A target adds its own via `Target.failure_markers` — the
# exact wording is product-specific, and missing it means scoring a failure as an answer.
DEFAULT_FAILURE_MARKERS = (
    "unable to generate a response",
    "something went wrong",
    "an error occurred while",
    "please try again later",
    "i ran into an error",
)


def agent_failure_in(text: str, markers: tuple[str, ...] = DEFAULT_FAILURE_MARKERS) -> str | None:
    """Return the marker matched, if the text is the product's own failure message."""
    lowered = (text or "").lower()
    # Only the tail: a long, genuine answer that happens to discuss errors is not a failure.
    tail = lowered[-400:] if len(lowered) > 400 else lowered
    return next((m for m in markers if m in tail), None)


def merge_tool_calls(observed: list[ToolCall], extra: list[ToolCall]) -> list[ToolCall]:
    """Prefer the live-observed calls; add ones only a second source saw.

    Targets typically have two views of a turn — what the stream announced and what the
    backend recorded — and neither is complete. The live view is authoritative on ordering
    and timing; the other catches steps that ran without announcing themselves, which is a
    real source of silent behaviour changes.
    """
    seen = {(c.turn, c.key) for c in observed}
    return observed + [c for c in extra if (c.turn, c.key) not in seen]


