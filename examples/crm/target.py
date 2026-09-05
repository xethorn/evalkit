"""CRM Product API Target example for evalkit.

Demonstrates integrating an agent that calls a CRM API (/customers)
to query customer records and perform operations like updating birthdays via HTTP endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.target import BaseTarget, ChatDriver, Check, SampleContext, TurnResult
from evalkit.trace import ToolCall


class CrmChatDriver:
    """ChatDriver communicating with a CRM product API."""

    def __init__(self, sample_id: str, epoch: int, base_url: str = "https://api.crm.example.com/v1") -> None:
        self.sample_id = sample_id
        self.epoch = epoch
        self.base_url = base_url
        self.chat_id = f"crm-session-{sample_id}-{epoch}"
        self.turn_index = -1

    async def start(self) -> None:
        """Authenticate and establish session with CRM API."""
        # REAL INTEGRATION GUIDE:
        # Initialize HTTP client and authenticate against CRM API, e.g.:
        #   self.client = httpx.AsyncClient(
        #       base_url=self.base_url,
        #       headers={"Authorization": f"Bearer {os.environ['CRM_API_KEY']}"}
        #   )

    async def new_chat(self) -> None:
        """Reset CRM session state."""

    async def send(self, text: str) -> TurnResult:
        """Send message to CRM agent and record normalized tool calls."""
        self.turn_index += 1

        # =========================================================================
        # REAL INTEGRATION GUIDE:
        # In a real setup, dispatch the prompt to your CRM agent API endpoint:
        #
        #   res = await self.client.post("/agent/chat", json={"prompt": text})
        #   data = res.json()
        #   response_text = data["message"]
        #   tool_calls = [
        #       ToolCall(
        #           name=call["tool_name"],
        #           status=call["status"],
        #           turn=self.turn_index,
        #           detail=call["endpoint_payload"],
        #       )
        #       for call in data.get("tool_executions", [])
        #   ]
        # =========================================================================

        # [MOCK DATA] Simulated agent finding customer record and calling POST /customers/CUST-1042/birthday
        if "birthday" in text.lower():
            response_text = (
                "Found customer Jane Doe (CUST-1042). Updated birthday to 1990-05-15 via POST "
                "/customers/CUST-1042/birthday."
            )
            tool_calls = [
                ToolCall(
                    name="find_customer",
                    status="completed",
                    turn=self.turn_index,
                    detail="GET /customers?query=Jane%20Doe -> CUST-1042",
                ),
                ToolCall(
                    name="update_customer_birthday",
                    status="completed",
                    turn=self.turn_index,
                    detail="POST /customers/CUST-1042/birthday -> 200 OK",
                ),
            ]
        else:
            response_text = f"CRM Agent processed query: '{text}'."
            tool_calls = [
                ToolCall(
                    name="find_customer",
                    status="completed",
                    turn=self.turn_index,
                    detail="GET /customers",
                )
            ]

        return TurnResult(
            index=self.turn_index,
            text=response_text,
            rendered_text=response_text,
            latency_ms=280,
            tool_calls=tool_calls,
        )

    async def approve_interrupt(self) -> TurnResult | None:
        return None

    async def agent_config(self) -> dict[str, Any]:
        return {
            "crm_api_version": "v1",
            "base_url": self.base_url,
            "model": "claude-3-5-sonnet",
        }

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "httpErrors": [],
            "status": "connected",
        }


class CrmTarget(BaseTarget):
    """Evalkit Target for a CRM product API agent."""

    name = "crm-agent"
    display_name = "CRM Product API Target"

    async def open(
        self,
        sample: SampleContext,
        artifacts_dir: Path,
        record_trace: bool = True,
    ) -> ChatDriver:
        return CrmChatDriver(sample.sample_id, sample.epoch)

    def judge_domain(self) -> str:
        return "an AI CRM operations assistant"

    def tool_aliases(self) -> dict[str, str]:
        return {
            "find_customer": "CRM Customer Search",
            "update_customer_birthday": "CRM Birthday Update",
        }

    def vocabulary(self) -> dict[str, list[str]]:
        return {
            "customer_id": ["CUST-1042", "CUST-3091", "CUST-8812"],
            "birthday": ["1990-05-15", "1985-11-22", "1998-03-01"],
        }

    def fingerprint(self) -> dict[str, Any]:
        return {
            "crm_environment": "production-api",
            "api_version": "v1",
        }

    def doctor_checks(self) -> list[Check]:
        return [
            Check(name="CRM API Endpoint", ok=True, detail="https://api.crm.example.com/v1 reachable"),
            Check(name="CRM Auth Token", ok=True, detail="Token valid with /customers write scope"),
        ]


crm_target = CrmTarget()
