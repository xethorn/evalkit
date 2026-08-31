"""The simulated user: what happens when the agent asks a question.

Multi-turn evals need a counterpart, and the counterpart has to be *cheap to specify*
and *hard to over-help with*. Two layers:

1. **Scripted answers** (deterministic, preferred). Each rule matches on the agent's
   question and/or the tool it wants to approve, and supplies the reply. Deterministic
   answers keep the eval comparable between runs — an LLM user that answers differently
   each run adds variance you cannot attribute.
2. **A simulated-user model** (fallback). Given a persona and a fixed set of *facts the
   user knows*, it answers in one or two sentences. It is instructed never to volunteer
   the solution: an over-helpful simulated user turns a failing agent into a passing one.

Either way the reply's `origin` is recorded, so a transcript shows whether a pass came
from the script or from the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

InterruptPolicy = Literal["approve", "reject", "answer", "never"]

STOP_TOKEN = "__STOP__"

# The persona carries the domain: a suite says who this user is and what they care
# about, so this prompt never has to assume an industry or a product.
SIMULATED_USER_SYSTEM = """You are role-playing a user talking to an AI assistant.

Persona: {persona}

Facts you know (the ONLY information you may provide):
{facts}

Rules:
- Reply as the user, in at most two short sentences. No preamble, no markdown.
- Answer only what was asked. If the answer is in your facts, give it.
- If the assistant asks for something not in your facts, say you don't have it and let it decide.
- NEVER suggest how to do the task, which tool to use, or correct the assistant's approach.
- If the assistant has already answered and is only making conversation, reply exactly {stop}.
- If the assistant is asking the same thing again, reply exactly {stop}.
"""


@dataclass
class AnswerRule:
    """One scripted reply."""

    reply: str
    when: str | None = None            # regex against the assistant's message
    when_tool: str | None = None       # matches the pending interrupt's tool name
    once: bool = True
    used: int = 0

    def matches(self, question: str, tool: str | None) -> bool:
        if self.once and self.used:
            return False
        if self.when_tool and (tool or "").lower() != self.when_tool.lower():
            return False
        # A rule with only `when_tool` matches any question asked for that tool.
        return self.when is None or bool(re.search(self.when, question, re.IGNORECASE | re.DOTALL))


@dataclass
class UserPolicy:
    """How the simulated user behaves for one sample."""

    persona: str = "A staff accountant at a mid-size SaaS company. Busy, terse, trusts the tool."
    facts: list[str] = field(default_factory=list)
    rules: list[AnswerRule] = field(default_factory=list)
    interrupts: InterruptPolicy = "approve"
    max_turns: int = 6
    model: str | None = None  # None => the fallback model configured for the run

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UserPolicy:
        data = data or {}
        rules = [
            AnswerRule(
                reply=str(r["reply"]),
                when=r.get("when"),
                when_tool=r.get("when_tool"),
                once=bool(r.get("once", True)),
            )
            for r in data.get("answers", [])
        ]
        return cls(
            persona=data.get("persona", cls.persona),
            facts=[str(f) for f in data.get("facts", [])],
            rules=rules,
            interrupts=data.get("interrupts", "approve"),
            max_turns=int(data.get("max_turns", 6)),
            model=data.get("model"),
        )

    def scripted(self, question: str, tool: str | None) -> str | None:
        for rule in self.rules:
            if rule.matches(question, tool):
                rule.used += 1
                return rule.reply
        return None


@dataclass
class UserReply:
    text: str
    origin: str  # "scripted" | "simulated" | "stop"

    @property
    def is_stop(self) -> bool:
        return self.text.strip() == STOP_TOKEN or self.origin == "stop"


async def next_reply(
    policy: UserPolicy,
    question: str,
    transcript: str,
    tool: str | None = None,
    model: Any | None = None,
) -> UserReply:
    """Produce the next user message, preferring the script."""
    scripted = policy.scripted(question, tool)
    if scripted is not None:
        return UserReply(text=scripted, origin="scripted")

    if model is None:
        # No fallback model configured: end the conversation rather than invent a user.
        return UserReply(text=STOP_TOKEN, origin="stop")

    facts = "\n".join(f"- {f}" for f in policy.facts) or "- (nothing beyond your original request)"
    system = SIMULATED_USER_SYSTEM.format(persona=policy.persona, facts=facts, stop=STOP_TOKEN)
    prompt = (
        f"Conversation so far:\n\n{transcript[-8000:]}\n\n"
        "The assistant's latest message is above. Reply as the user."
    )
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    out = await model.generate([ChatMessageSystem(content=system), ChatMessageUser(content=prompt)])
    text = (out.completion or "").strip()
    if not text:
        return UserReply(text=STOP_TOKEN, origin="stop")
    return UserReply(text=text, origin="simulated")
