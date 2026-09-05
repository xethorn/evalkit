"""An example UI target integration with vendor trace fetching (e.g. Braintrust).

Demonstrates driving an agent through a web UI (e.g. Playwright / browser session)
while querying a vendor observability / tracing API (such as Braintrust, LangSmith, or Phoenix)
to retrieve structured spans, tool execution events, and token usage for evalkit trace normalization.

Usage:
    EVAL_TARGET=examples.ui_trace_target:ui_trace_target evalkit doctor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.target import BaseTarget, ChatDriver, Check, SampleContext, TurnResult
from evalkit.trace import ToolCall


class VendorTraceClient:
    """Mock client for fetching traces from an observability vendor like Braintrust."""

    def __init__(self, api_key: str = "braintrust-test-key", project_id: str = "agent-ui-prod") -> None:
        self.api_key = api_key
        self.project_id = project_id

    async def fetch_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Query vendor API for spans and tool executions under trace_id."""
        return [
            {
                "span_id": f"span-tool-1-{trace_id}",
                "name": "browser_navigation",
                "status": "completed",
                "input": {"url": "https://app.example.com/dashboard"},
            },
            {
                "span_id": f"span-tool-2-{trace_id}",
                "name": "dom_click",
                "status": "completed",
                "input": {"selector": "#settings-button"},
            },
        ]


class UiTraceChatDriver:
    """ChatDriver interacting with an agent via UI and enriching trace with vendor data."""

    def __init__(self, sample_id: str, epoch: int, vendor_client: VendorTraceClient | None = None) -> None:
        self.sample_id = sample_id
        self.epoch = epoch
        self.vendor_client = vendor_client or VendorTraceClient()
        self.chat_id = f"ui-session-{sample_id}-{epoch}"
        self.turn_index = -1

    async def start(self) -> None:
        """Launch browser page / Playwright driver."""

    async def new_chat(self) -> None:
        """Navigate to fresh chat page in web app."""

    async def send(self, text: str) -> TurnResult:
        """Send message via UI DOM, wait for response, and fetch vendor traces."""
        self.turn_index += 1
        trace_id = f"bt-trace-{self.sample_id}-{self.turn_index}"

        # Fetch telemetry spans from Braintrust / vendor API for this turn
        spans = await self.vendor_client.fetch_spans(trace_id)

        tool_calls = [
            ToolCall(
                name=span["name"],
                status=span["status"],
                turn=self.turn_index,
                detail=str(span.get("input", {})),
            )
            for span in spans
        ]

        rendered = f"UI Agent completed task in browser for prompt: '{text}'."
        return TurnResult(
            index=self.turn_index,
            text=rendered,
            rendered_text=rendered,
            trace_id=trace_id,
            latency_ms=1250,
            tool_calls=tool_calls,
        )

    async def approve_interrupt(self) -> TurnResult | None:
        return None

    async def agent_config(self) -> dict[str, Any]:
        return {
            "interface": "web-ui",
            "trace_vendor": "braintrust",
            "browser": "chromium",
        }

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "httpErrors": [],
            "pageErrors": [],
            "status": "browser_ready",
        }


class UiTraceTarget(BaseTarget):
    """Evalkit Target combining UI web session with vendor trace fetching."""

    name = "ui-trace-agent"
    display_name = "UI Driver + Braintrust Vendor Trace Target"

    async def open(
        self,
        sample: SampleContext,
        artifacts_dir: Path,
        record_trace: bool = True,
    ) -> ChatDriver:
        return UiTraceChatDriver(sample.sample_id, sample.epoch)

    def judge_domain(self) -> str:
        return "an AI browser navigation assistant"

    def tool_aliases(self) -> dict[str, str]:
        return {
            "browser_navigation": "Navigate Page",
            "dom_click": "Click Element",
        }

    def fingerprint(self) -> dict[str, Any]:
        return {
            "ui_url": "https://app.example.com",
            "trace_vendor": "braintrust",
            "project_id": "agent-ui-prod",
        }

    def doctor_checks(self) -> list[Check]:
        return [
            Check(name="Browser Driver", ok=True, detail="Playwright / Chromium initialized"),
            Check(name="Braintrust Trace API", ok=True, detail="Vendor API key valid & project linked"),
        ]


ui_trace_target = UiTraceTarget()
