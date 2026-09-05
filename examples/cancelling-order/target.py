"""Order Cancellation Target example for evalkit.

Demonstrates a customer support target evaluating order lookup, multi-turn
clarifications, and order cancellation/refund policy execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.target import BaseTarget, ChatDriver, Check, SampleContext, TurnResult
from evalkit.trace import ToolCall


class OrderCancelChatDriver:
    """ChatDriver simulating an order cancellation & support conversation."""

    def __init__(self, sample_id: str, epoch: int) -> None:
        self.sample_id = sample_id
        self.epoch = epoch
        self.chat_id = f"cancel-session-{sample_id}-{epoch}"
        self.turn_index = -1

    async def start(self) -> None:
        """Initialize session."""

    async def new_chat(self) -> None:
        """Reset conversation state."""

    async def send(self, text: str) -> TurnResult:
        """Process user message and return response with tool activity."""
        self.turn_index += 1

        if "cancel" in text.lower() or "ord-9921" in text.lower():
            response_text = (
                "Order ORD-9921 has been cancelled according to policy. A refund of $85.00 has "
                "been processed."
            )
            tool_calls = [
                ToolCall(
                    name="lookup_order",
                    status="completed",
                    turn=self.turn_index,
                    detail="ORD-9921 -> status: unfulfilled",
                ),
                ToolCall(
                    name="cancel_order",
                    status="completed",
                    turn=self.turn_index,
                    detail="ORD-9921 -> cancelled & refunded",
                ),
            ]
        else:
            response_text = f"Can you provide your order ID? Text received: '{text}'."
            tool_calls = []

        return TurnResult(
            index=self.turn_index,
            text=response_text,
            rendered_text=response_text,
            latency_ms=210,
            tool_calls=tool_calls,
        )

    async def approve_interrupt(self) -> TurnResult | None:
        return None

    async def agent_config(self) -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "temperature": 0.0,
            "version": "2.1.0",
        }

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "httpErrors": [],
            "status": "healthy",
        }


class OrderCancelTarget(BaseTarget):
    """Target for Order Cancellation & Customer Support agent."""

    name = "cancelling-order-agent"
    display_name = "Cancelling Order Support Target"

    async def open(
        self,
        sample: SampleContext,
        artifacts_dir: Path,
        record_trace: bool = True,
    ) -> ChatDriver:
        return OrderCancelChatDriver(sample.sample_id, sample.epoch)

    def judge_domain(self) -> str:
        return "an AI order management and refund support assistant"

    def tool_aliases(self) -> dict[str, str]:
        return {
            "lookup_order": "Order Lookup",
            "cancel_order": "Order Cancellation",
        }

    def vocabulary(self) -> dict[str, list[str]]:
        return {
            "order_id": ["ORD-9921", "ORD-4021", "ORD-1102"],
        }

    def fingerprint(self) -> dict[str, Any]:
        return {
            "store_tenant": "acme-retail",
            "refund_policy_version": "2026.1",
        }

    def doctor_checks(self) -> list[Check]:
        return [
            Check(name="Order System API", ok=True, detail="Order service reachable"),
            Check(name="Payment Refund Gateway", ok=True, detail="Refund credentials valid"),
        ]


cancelling_order_target = OrderCancelTarget()
