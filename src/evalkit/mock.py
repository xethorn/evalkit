"""An offline target: the framework talking to itself.

Two jobs, both of which an eval framework needs badly:

* **Self-test.** The scorers, the store, the paired statistics and the reports are the
  parts most likely to be silently wrong, and they should be testable without a running
  app, a login, or a judge key. ``run --mock`` exercises the whole pipeline.
* **Known-answer calibration.** Because the mock's behaviour is dialled by
  ``--mock-quality``, you can verify the comparison machinery actually detects an
  improvement you *know* is there before trusting it to detect one you only hope is
  there. A suite that can't see a planted improvement won't see a real one.

It replays fixtures from ``evals/fixtures/<sample-id>.json`` when present, and otherwise
synthesizes a deterministic conversation from the sample's own expectations. No target
package, no app and no network are involved, which is the point: the parts of this repo
most likely to be silently wrong can be tested with nothing installed but the framework.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .target import BaseTarget, ChatDriver, SampleContext, TurnResult
from .trace import Interrupt, SseEvent, Subagent, ToolCall

FIXTURE_DIR = Path("evals/fixtures")


class MockChatDriver:
    """A :class:`~evalkit.target.ChatDriver` that answers from the sample's own expectations."""

    def __init__(self, sample_id: str, epoch: int, metadata: dict[str, Any], quality: str = "good"):
        self.sample_id = sample_id
        self.metadata = metadata or {}
        self.quality = quality
        self.chat_id = f"mock-{sample_id}-{epoch}"
        self.turn_index = -1
        self.rng = random.Random(f"{sample_id}:{epoch}:{quality}")
        self.fixture = self._load_fixture()
        self._asked = False

    def _load_fixture(self) -> dict[str, Any] | None:
        path = FIXTURE_DIR / f"{self.sample_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    async def start(self) -> None:
        return None

    async def new_chat(self) -> None:
        return None

    async def agent_config(self) -> dict[str, Any]:
        return {"selectedAgents": ["mock"], "selectedModel": "mock", "composer": "mock"}

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "httpErrors": [],
            "pageErrors": [],
            "requests": [],
            "truncated": 0,
            # On the record, so nobody mistakes a mock run for a measurement of the agent.
            "notes": [f"MOCK RUN (quality={self.quality}) — no app was contacted"],
        }

    async def approve_interrupt(self) -> TurnResult | None:
        if self.quality == "poor":
            return None
        self.turn_index += 1
        return TurnResult(
            index=self.turn_index,
            text="Approved. The entry is now queued as a proposal awaiting your final review.",
            latency_ms=1200,
            tool_calls=[ToolCall(name="call_write_endpoint", status="completed", turn=self.turn_index)],
        )

    async def send(self, text: str) -> TurnResult:  # noqa: ARG002 - mirrors the real driver
        self.turn_index += 1
        turn = self.turn_index
        if self.fixture:
            return self._from_fixture(turn)
        return self._synthesize(turn)

    def _from_fixture(self, turn: int) -> TurnResult:
        turns = self.fixture.get("turns", [])
        raw = turns[min(turn, len(turns) - 1)]
        return TurnResult(
            index=turn,
            text=str(raw.get("text", "")),
            latency_ms=int(raw.get("latency_ms", 1000)),
            tool_calls=[ToolCall(**{**c, "turn": turn}) for c in raw.get("tool_calls", [])],
            interrupt=Interrupt(**{**raw["interrupt"], "turn": turn}) if raw.get("interrupt") else None,
            events=[SseEvent(turn=turn, at_ms=0, event="done", data={})],
        )

    def _synthesize(self, turn: int) -> TurnResult:
        """Build a plausible answer straight from the sample's expectations.

        A "good" mock satisfies the declared assertions and tool expectations; a "poor"
        one drops content, over-calls tools and skips the clarifying question. The gap
        between them is the planted improvement the comparison machinery must detect.
        """
        expect = self.metadata.get("expect") or {}
        tools_spec = expect.get("tools") or {}
        good = self.quality == "good"
        must_ask = expect.get("must_ask")

        # Turn 0 of a must_ask sample: a good agent asks first.
        if turn == 0 and must_ask and good and not self._asked:
            self._asked = True
            return TurnResult(
                index=turn,
                text="Before I pull the numbers — which subsidiary should this cover, and which period?",
                latency_ms=900,
                tool_calls=[],
                events=[SseEvent(turn=turn, at_ms=0, event="done", data={})],
            )

        required = list(tools_spec.get("required") or [])
        required_any = list(tools_spec.get("required_any") or [])
        calls: list[ToolCall] = []
        for name in required + (required_any[:1] if required_any else []):
            calls.append(ToolCall(name=name, status="completed", turn=turn, detail="mock call"))
        if not good:
            # The classic regression: same query over and over, plus a budget overrun.
            noisy = (required_any or required or ["a_tool"])[0]
            calls += [ToolCall(name=noisy, status="completed", turn=turn) for _ in range(6)]
            for name in tools_spec.get("forbidden") or []:
                calls.append(ToolCall(name=name, status="completed", turn=turn))

        must = expect.get("must_contain") or []
        body = (
            "Here is the breakdown you asked for, itemized, and attributed to the data "
            "it came from for the period stated."
            if good
            else "Done."
        )
        if good:
            body += " " + " ".join(str(m) for m in must)

        # Route to whichever sub-agent the case expects, so the routing scorer has
        # something real to agree or disagree with.
        expected_agents = [str(a) for a in ((expect.get("agents") or {}).get("required") or [])]

        return TurnResult(
            index=turn,
            text=body,
            rendered_text=body,
            latency_ms=self.rng.randint(3000, 20000) if good else self.rng.randint(60000, 200000),
            tool_calls=calls,
            subagents=[Subagent(name=name, turn=turn) for name in expected_agents] if good else [],
            events=[SseEvent(turn=turn, at_ms=0, event="done", data={})],
            trace_id=f"mock-trace-{turn}",
        )


class MockTarget(BaseTarget):
    """The framework's offline self-test target (``EVAL_TARGET=mock``).

    ``quality`` is what makes it useful rather than merely offline: "good" satisfies the
    declared expectations and "poor" violates them in the classic ways, so the gap between
    two mock evaluations is an improvement of *known* size. A comparison that cannot see it
    will not see a real one either.
    """

    name = "mock"
    display_name = "Mock agent"

    def __init__(self, quality: str = "good"):
        self.quality = quality

    async def open(self, sample: SampleContext, artifacts_dir: Path, record_trace: bool = True) -> ChatDriver:
        return MockChatDriver(sample.sample_id, sample.epoch, sample.metadata, quality=self.quality)


#: The instance ``EVAL_TARGET=mock`` resolves to.
mock_target = MockTarget()
