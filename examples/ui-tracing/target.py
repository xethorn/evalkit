"""UI Driver + Telemetry Trace Target example for evalkit.

Demonstrates driving an agent in a browser UI while retrieving telemetry spans,
tool executions, and model traces from a vendor observability API (such as Braintrust).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.target import BaseTarget, ChatDriver, Check, SampleContext, TurnResult
from evalkit.trace import ToolCall


class BraintrustTraceClient:
    """Client for fetching spans from vendor observability platform (e.g. Braintrust)."""

    def __init__(self, api_key: str = "braintrust-key", project: str = "ui-automation") -> None:
        self.api_key = api_key
        self.project = project

    async def fetch_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Fetch spans from Braintrust trace API."""
        # =========================================================================
        # REAL INTEGRATION GUIDE:
        # In a real setup, replace this mock with calls to the Braintrust SDK or API:
        #
        #   from braintrust import init_logger
        #   # or HTTP call:
        #   # async with httpx.AsyncClient() as client:
        #   #     res = await client.get(
        #   #         f"https://api.braintrust.dev/v1/trace/{trace_id}",
        #   #         headers={"Authorization": f"Bearer {self.api_key}"}
        #   #     )
        #   #     return res.json()["spans"]
        # =========================================================================

        # [MOCK DATA] Simulated spans returned from Braintrust trace API:
        return [
            {
                "span_id": f"span-nav-{trace_id}",
                "name": "browser_navigation",
                "status": "completed",
                "input": {"url": "https://app.example.com/checkout"},
            },
            {
                "span_id": f"span-click-{trace_id}",
                "name": "dom_click",
                "status": "completed",
                "input": {"selector": "button#cancel-order"},
            },
        ]


class UiTracingChatDriver:
    """ChatDriver automating web UI and fetching Braintrust vendor traces."""

    def __init__(self, sample_id: str, epoch: int) -> None:
        self.sample_id = sample_id
        self.epoch = epoch
        self.trace_client = BraintrustTraceClient()
        self.chat_id = f"ui-trace-session-{sample_id}-{epoch}"
        self.turn_index = -1

    async def start(self) -> None:
        """Launch Playwright browser session."""
        # REAL INTEGRATION GUIDE:
        # Initialize Playwright browser page here, e.g.:
        #   self.browser = await playwright.chromium.launch()
        #   self.page = await self.browser.new_page()

    async def new_chat(self) -> None:
        """Navigate to fresh session page."""
        # REAL INTEGRATION GUIDE:
        # Navigate browser page to clean chat session:
        #   await self.page.goto("https://app.example.com/chat/new")

    async def send(self, text: str) -> TurnResult:
        """Send input via UI DOM, wait for agent, and fetch vendor spans."""
        self.turn_index += 1
        trace_id = f"braintrust-trace-{self.sample_id}-{self.turn_index}"

        # REAL INTEGRATION GUIDE:
        # 1. Type message into DOM input element and click submit:
        #    await self.page.fill("#chat-input", text)
        #    await self.page.click("#send-button")
        # 2. Wait for assistant turn response element to settle.
        # 3. Read trace ID from response headers or DOM data attributes.

        # Fetch telemetry spans from Braintrust trace API for this turn
        spans = await self.trace_client.fetch_spans(trace_id)
        tool_calls = [
            ToolCall(
                name=span["name"],
                status=span["status"],
                turn=self.turn_index,
                detail=str(span.get("input", {})),
            )
            for span in spans
        ]

        # [MOCK DATA] Simulated rendered text from UI response:
        rendered = f"UI Agent completed navigation for query: '{text}'."
        return TurnResult(
            index=self.turn_index,
            text=rendered,
            rendered_text=rendered,
            trace_id=trace_id,
            latency_ms=1100,
            tool_calls=tool_calls,
        )

    async def approve_interrupt(self) -> TurnResult | None:
        return None

    async def agent_config(self) -> dict[str, Any]:
        return {
            "browser": "chromium",
            "telemetry_vendor": "braintrust",
            "version": "1.0.0",
        }

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "httpErrors": [],
            "status": "browser_connected",
        }


class UiTracingTarget(BaseTarget):
    """Target combining Playwright browser UI driving with Braintrust trace ingestion."""

    name = "ui-tracing-agent"
    display_name = "UI Automation + Braintrust Tracing Target"

    async def open(
        self,
        sample: SampleContext,
        artifacts_dir: Path,
        record_trace: bool = True,
    ) -> ChatDriver:
        return UiTracingChatDriver(sample.sample_id, sample.epoch)

    def judge_domain(self) -> str:
        return "an AI web navigation assistant"

    def tool_aliases(self) -> dict[str, str]:
        return {
            "browser_navigation": "Navigate Page",
            "dom_click": "Click Element",
        }

    def fingerprint(self) -> dict[str, Any]:
        return {
            "web_app_url": "https://app.example.com",
            "trace_vendor": "braintrust",
        }

    def doctor_checks(self) -> list[Check]:
        return [
            Check(name="Playwright Driver", ok=True, detail="Chromium engine ready"),
            Check(name="Braintrust Trace API", ok=True, detail="Project ui-automation linked"),
        ]


ui_tracing_target = UiTracingTarget()
