"""Verify the numbers, not just the way they are presented.

Every other scorer in this package grades *method*: did the answer name its source, state
the population it checked, cross-foot against itself. All of that can be satisfied by an
answer whose figures are wrong, which for a quantitative agent is the failure that matters
most. A change that improved disclosure while corrupting arithmetic would score better.

This scorer closes that gap, and the shape of it is the point:

* **Extraction is a model's job.** Pulling "what number did it report for the target
  metric" out of prose is a task models are reliable at, and it is asked in isolation, with
  no notion of whether the number is right.
* **Verification is not.** The comparison is `Decimal` arithmetic against a value the
  *target* computed from an independent source of truth by a path the agent never used. An LLM
  is never asked whether arithmetic is correct — the least reliable thing to ask it, and the
  easiest thing to compute.

The expected value is resolved at scoring time rather than written into the suite. A
literal pinned in YAML is right on the day it is written and silently wrong afterwards:
a value recomputed against the same database the agent just queried cannot rot.

The target owns the resolution (`Target.resolve_expected`), because what counts as the
expected value is the product's business, not the framework's. A target that offers no
resolver simply gets no figure scores — excluded, never zeroed, so an unconfigured
adjudicator never looks like an agent that got worse.
"""

from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal, InvalidOperation

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from ..config import settings
from ..solver import trace_from_state
from .common import EXCLUDED, EXCLUSION_REASON, QUALITY_METRICS

EXTRACT_SYSTEM = """\
You extract figures from a financial assistant's answer. You never judge whether a figure \
is correct, and you never compute anything. If the answer does not report the figure asked \
for, return null for it. Return only JSON."""

EXTRACT_PROMPT = """\
The assistant was asked:

{prompt}

Its answer:

{answer}

Extract each of the following figures exactly as the assistant reported it. Return a bare \
number with no currency symbol, no thousands separators, and no percent sign. Use null when \
the assistant did not report that figure.

{wanted}

Return JSON of the form {{"figures": {{"<name>": <number or null>, ...}}}} and nothing else.\
"""


def _to_decimal(v: object) -> Decimal | None:
    if v is None:
        return None
    try:
        # Strip anything a model might leave on a number despite being told not to.
        return Decimal(re.sub(r"[^0-9.\-]", "", str(v)) or "x")
    except (InvalidOperation, ValueError):
        return None


async def _extract(cfg, judge_model: str, prompt: str):
    """One extraction call. Temperature is omitted unless configured — reasoning models
    reject it outright, which is why JudgeConfig.temperature defaults to None."""
    extractor = get_model(
        judge_model,
        config=(
            GenerateConfig(temperature=cfg.judge.temperature)
            if cfg.judge.temperature is not None
            else GenerateConfig()
        ),
    )
    return await asyncio.wait_for(
        extractor.generate(
            [ChatMessageSystem(content=EXTRACT_SYSTEM), ChatMessageUser(content=prompt)]
        ),
        timeout=cfg.judge.timeout_s,
    )


@scorer(metrics=QUALITY_METRICS)
def figures(model: str | None = None):
    """Compare figures the answer reports against expected values."""
    cfg = settings()
    judge_model = model or cfg.judge.model

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        expect = (state.metadata or {}).get("expect") or {}
        specs = expect.get("figures") or []

        def excluded(reason: str, detail: str = "") -> Score:
            return Score(
                value=0.0,
                answer=detail or reason,
                explanation=detail,
                metadata={EXCLUDED: True, EXCLUSION_REASON: [reason]},
            )

        if not specs:
            return excluded("no_figures")
        if trace.infra_errors:
            return excluded("infra_error", "infra failure — excluded from quality metrics")

        # Resolve every expected value first. If the adjudicator cannot answer, this case
        # is unscoreable — that is a fixture or configuration problem, and reporting it as
        # a failing agent would be a lie about where the fault is.
        resolver = getattr(settings().target, "resolve_expected", None)
        if resolver is None:
            return excluded("no_resolver", "the target supplies no expected-value resolver")

        bindings = (state.metadata or {}).get("bindings") or {}
        wanted: dict[str, Decimal] = {}
        unresolved: list[str] = []
        for spec in specs:
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            try:
                got = resolver(spec, bindings)
            except Exception as exc:  # noqa: BLE001 - an adjudicator failure is never the agent's
                got, exc_note = None, f"{type(exc).__name__}: {exc}"
                unresolved.append(f"{name} ({exc_note[:80]})")
                continue
            dec = _to_decimal(got)
            if dec is None:
                unresolved.append(name)
            else:
                wanted[name] = dec
        if not wanted:
            return excluded(
                "expected_unresolved",
                "could not compute an expected value for: " + ", ".join(unresolved),
            )

        answer = trace.transcript_text()
        if not answer.strip():
            return Score(value=0.0, answer="empty", explanation="the agent produced no text")

        prompt = EXTRACT_PROMPT.format(
            prompt=trace.prompt,
            answer=answer[:24000],
            wanted="\n".join(f"- {n}: {specs_by_name(specs, n)}" for n in wanted),
        )
        # This scorer doubles the judge calls per sample, so it is the one most likely to
        # meet a rate limit — and the first real run lost every extraction that way. Retry
        # briefly rather than reporting a transient limit as an unscoreable case.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                out = await _extract(cfg, judge_model, prompt)
                raw = json.loads(re.search(r"\{.*\}", out.completion, re.S).group(0))
                reported = {k: _to_decimal(v) for k, v in (raw.get("figures") or {}).items()}
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
        if last_exc is not None:
            detail = f"{type(last_exc).__name__}: {last_exc}"
            return excluded("extraction_failed", detail[-300:] if len(detail) > 300 else detail)

        checks, hits = [], 0
        for name, expected in wanted.items():
            tol = _to_decimal(next((s.get("tolerance") for s in specs if s.get("name") == name), "0.01"))
            tol = tol if tol is not None else Decimal("0.01")
            got = reported.get(name)
            if got is None:
                checks.append({"figure": name, "ok": False, "expected": str(expected), "reported": None,
                               "detail": "not reported in the answer"})
                continue
            ok = abs(got - expected) <= tol
            hits += 1 if ok else 0
            checks.append({"figure": name, "ok": ok, "expected": str(expected),
                           "reported": str(got), "delta": str(got - expected), "tolerance": str(tol)})

        value = hits / len(wanted)
        wrong = [c for c in checks if not c["ok"]]
        explanation = (
            f"all {len(wanted)} figure(s) match expected values"
            if not wrong
            else "; ".join(
                f"{c['figure']}: reported {c['reported']} vs expected {c['expected']}"
                if c["reported"] is not None
                else f"{c['figure']}: not reported (expected {c['expected']})"
                for c in wrong
            )
        )
        return Score(
            value=value,
            answer=f"{hits}/{len(wanted)} verified",
            explanation=explanation,
            metadata={"checks": checks, "unresolved": unresolved, "judge_model": judge_model},
        )

    return score


def specs_by_name(specs: list[dict], name: str) -> str:
    """The human description of a figure, for the extraction prompt."""
    for s in specs:
        if s.get("name") == name:
            return str(s.get("describe") or name)
    return name
