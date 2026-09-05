"""An example target integration for evalkit.

To run evalkit with this target, set the environment variable EVAL_TARGET:
    EVAL_TARGET=examples.simple_target:my_target evalkit doctor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.target import BaseTarget, ChatDriver, Check, SampleContext, TurnResult
from evalkit.trace import ToolCall


class SimpleChatDriver:
    """An example ChatDriver driving a mock or real custom agent session."""

    def __init__(self, sample_id: str, epoch: int) -> None:
        self.sample_id = sample_id
        self.epoch = epoch
        self.chat_id = f"simple-chat-{sample_id}-{epoch}"
        self.turn_index = -1

    async def start(self) -> None:
        """Initialize session or establish connection with the agent."""

    async def new_chat(self) -> None:
        """Start a fresh conversation for this sample."""

    async def send(self, text: str) -> TurnResult:
        """Send a user message to the agent and return the turn result."""
        self.turn_index += 1
        response_text = f"Received prompt: '{text}'. Here is the calculated result."
        return TurnResult(
            index=self.turn_index,
            text=response_text,
            rendered_text=response_text,
            latency_ms=150,
            tool_calls=[
                ToolCall(
                    name="calculator",
                    status="completed",
                    turn=self.turn_index,
                    detail="performed calculation",
                )
            ],
        )

    async def approve_interrupt(self) -> TurnResult | None:
        """Handle pending user interrupts if applicable."""
        return None

    async def agent_config(self) -> dict[str, Any]:
        """Return agent configuration under test."""
        return {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "version": "1.0.0",
        }

    async def diagnostics(self) -> dict[str, Any]:
        """Return health or connection diagnostics."""
        return {
            "httpErrors": [],
            "status": "healthy",
        }


class SimpleTarget(BaseTarget):
    """An example Target integrating a custom AI agent framework into evalkit."""

    name = "simple-agent"
    display_name = "Simple Agent Example"

    async def open(
        self,
        sample: SampleContext,
        artifacts_dir: Path,
        record_trace: bool = True,
    ) -> ChatDriver:
        """Open a new conversation driver for a given sample."""
        return SimpleChatDriver(sample.sample_id, sample.epoch)

    def fingerprint(self) -> dict[str, Any]:
        """Return configuration that uniquely identifies this test environment."""
        return {
            "environment": "example",
            "agent_version": "1.0.0",
        }

    def doctor_checks(self) -> list[Check]:
        """Return preflight diagnostic checks for `evalkit doctor`."""
        return [
            Check(name="Agent API Connection", ok=True, detail="Successfully connected to simple-agent"),
            Check(name="API Key", ok=True, detail="Key configured"),
        ]


my_target = SimpleTarget()
