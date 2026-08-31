"""LLM-as-judge, plus the deterministic checks that should run *before* a judge.

Two rules shape this module:

* **Cheap and certain first.** ``assertions`` checks literal substrings (including the
  randomized values that went into the prompt) and numeric tolerance. If the answer
  omits the period it was asked about, no judge call is needed.
* **A judge must be pinned, structured, and quotable.** The model id is fixed in config
  and recorded; the rubric is graded criterion by criterion; each verdict carries the
  evidence quote that justified it. Free-text "score it 1-10" grading drifts between
  runs and cannot be audited, which makes hill-climbing against it meaningless.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from ..config import settings
from ..solver import trace_from_state
from .common import EXCLUDED, EXCLUSION_REASON, QUALITY_METRICS, clamp

# `{domain}` is filled from the target ("an AI assistant"). It is not
# decoration: a grader told the wrong domain applies the wrong standard of care, and the
# judge prompt *is* the metric.
JUDGE_SYSTEM = """You grade {domain}'s conversation against a rubric.

You are strict and evidence-driven:
- Judge ONLY what the assistant actually said or did. Never credit intent.
- A criterion is met only if you can quote the words that satisfy it.
- Asking a sensible clarifying question is good behaviour, not a failure, unless the
  rubric says the answer was required outright.
- Fabricated figures, invented account names, or claims of completed actions with no
  supporting tool activity are failures even if the prose is confident.
- If the assistant's reply is an error or a non-answer ("I was unable to generate a
  response", "something went wrong", an empty or truncated reply), then NO criterion is
  met — not even a criterion about what it refrained from doing. A failed turn must never
  earn credit for the mistakes it did not have the chance to make.

Return ONLY a JSON object:
{"criteria": [{"id": "<short id>", "met": true|false, "evidence": "<quote or 'none'>"}],
 "score": <0.0-1.0>, "verdict": "pass"|"partial"|"fail", "reasoning": "<2 sentences max>"}
"""

def judge_system(cfg) -> str:
    """The grading system prompt, with the target's domain filled in.

    Never fatal: a target that offers no domain gets the generic wording rather than a
    failed scorer, because losing the judge costs a run that took minutes per sample.
    """
    try:
        domain = cfg.target.judge_domain()
    except Exception:
        domain = "an AI assistant"
    # A plain substitution, not `.format`: the prompt ends with a literal JSON schema, and
    # formatting it would try to read those braces as fields.
    return JUDGE_SYSTEM.replace("{domain}", domain or "an AI assistant")


JUDGE_PROMPT = """## The user's request
{prompt}

## Rubric (each bullet is one criterion)
{rubric}

## Conversation transcript
{transcript}

## Observed tool activity (for grounding claims of completed work)
{tools}

Grade the rubric now."""


PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "grok": "GROK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _judge_available(model: str) -> bool:
    provider = model.split("/")[0]
    if provider in {"mockllm", "mock", "none"}:
        return False
    key = PROVIDER_KEYS.get(provider)
    # Providers we don't recognise (a local ollama, a proxy) are assumed reachable.
    return bool(os.environ.get(key)) if key else True


def _json_from(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@scorer(metrics=QUALITY_METRICS)
def rubric_judge(model: str | None = None):
    """Grade the conversation against the sample's rubric."""
    cfg = settings()
    judge_model = model or cfg.judge.model

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        expect = (state.metadata or {}).get("expect") or {}
        rubric = (expect.get("rubric") or "").strip()

        if not _judge_available(judge_model):
            # No credentials for the grader. Excluded rather than zeroed, so a missing key
            # never looks like an agent that suddenly got worse.
            return Score(
                value=0.0,
                answer="judge unavailable",
                explanation=f"no credentials for {judge_model}; set the provider API key to enable judging",
                metadata={EXCLUDED: True, EXCLUSION_REASON: ["judge_unavailable"], "judge_model": judge_model},
            )
        if trace.infra_errors:
            return Score(
                value=0.0,
                answer="not graded",
                explanation="infra failure — excluded from quality metrics",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        if not rubric:
            return Score(
                value=0.0,
                answer="no rubric",
                explanation="sample declares no rubric",
                metadata={EXCLUDED: True, EXCLUSION_REASON: ["no_rubric"]},
            )
        if not trace.transcript_text().strip():
            return Score(
                value=0.0,
                answer="empty",
                explanation="the agent produced no text at all",
                metadata={"judge_model": judge_model},
            )

        prompt = JUDGE_PROMPT.format(
            prompt=trace.prompt,
            rubric=rubric,
            transcript=trace.transcript_text()[:24000],
            tools=json.dumps(
                [
                    {"name": c.name, "subagent": c.subagent, "status": c.status, "turn": c.turn}
                    for c in trace.tool_calls
                ][:120],
                indent=1,
            ),
        )
        # A browser run costs minutes per sample. If the judge is unreachable — missing
        # provider package, rate limit, outage — that must cost only the judge's own score,
        # never the process and trace scores already collected. Before this was caught, a
        # missing dependency failed the whole run and discarded everything.
        generate_config = (
            GenerateConfig(temperature=cfg.judge.temperature)
            if cfg.judge.temperature is not None
            else GenerateConfig()
        )
        try:
            grader = get_model(judge_model, config=generate_config)
            out = await asyncio.wait_for(
                grader.generate(
                    [ChatMessageSystem(content=judge_system(cfg)), ChatMessageUser(content=prompt)]
                ),
                timeout=cfg.judge.timeout_s,
            )
        except TimeoutError:
            return Score(
                value=0.0,
                answer="judge timed out",
                explanation=f"{judge_model} did not answer within {cfg.judge.timeout_s}s",
                metadata={EXCLUDED: True, EXCLUSION_REASON: ["judge_timeout"], "judge_model": judge_model},
            )
        except Exception as exc:
            return Score(
                value=0.0,
                answer="judge failed",
                explanation=f"{type(exc).__name__}: {exc}"[:500],
                metadata={
                    EXCLUDED: True,
                    EXCLUSION_REASON: ["judge_error"],
                    "judge_model": judge_model,
                },
            )
        parsed = _json_from(out.completion or "")
        if not parsed:
            # A judge that cannot be parsed must not silently score zero.
            return Score(
                value=0.0,
                answer="unparseable",
                explanation=f"judge returned no JSON: {(out.completion or '')[:300]}",
                metadata={EXCLUDED: True, EXCLUSION_REASON: ["judge_unparseable"], "judge_model": judge_model},
            )

        criteria = parsed.get("criteria") or []
        met = [c for c in criteria if c.get("met")]
        # Prefer the criterion tally over the judge's own number: it is reproducible and
        # a reviewer can see which bullet moved.
        value = (len(met) / len(criteria)) if criteria else clamp(float(parsed.get("score", 0.0)))
        return Score(
            value=clamp(value),
            answer=str(parsed.get("verdict", "")),
            explanation=str(parsed.get("reasoning", ""))[:1000],
            metadata={
                "judge_model": judge_model,
                "criteria": criteria,
                "judge_self_score": parsed.get("score"),
                "criteria_met": len(met),
                "criteria_total": len(criteria),
            },
        )

    return score


def _numbers(text: str) -> list[float]:
    return [
        float(m.replace(",", ""))
        for m in re.findall(r"-?\d[\d,]*\.?\d*", text or "")
        if m.strip("-,.").replace(",", "")
    ]


@scorer(metrics=QUALITY_METRICS)
def assertions():
    """Deterministic content checks: required strings, banned strings, numeric targets.

    These are the assertions that must not depend on a judge's mood. ``must_contain``
    entries are rendered with the sample's own bindings, so an assertion can require the
    exact random period the prompt asked about without the suite hard-coding a date.
    """

    async def score(state: TaskState, target: Target) -> Score:
        trace = trace_from_state(state)
        expect = (state.metadata or {}).get("expect") or {}
        must = list(expect.get("must_contain") or [])
        must_not = list(expect.get("must_not_contain") or [])
        must_approx = expect.get("must_approx") or []

        if trace.infra_errors:
            return Score(
                value=0.0,
                explanation="infra failure — excluded",
                metadata={EXCLUDED: True, EXCLUSION_REASON: [e.kind for e in trace.infra_errors]},
            )
        if not must and not must_not and not must_approx:
            return Score(value=0.0, metadata={EXCLUDED: True, EXCLUSION_REASON: ["no_assertions"]})

        # Only the assistant's own words count. Including the user's turns here would
        # pass every `must_contain` that quotes the prompt — a silent false green.
        haystack = "\n".join(t.text for t in trace.assistant_turns).lower()
        missing = [s for s in must if s.lower() not in haystack]
        present = [s for s in must_not if s.lower() in haystack]

        # Numeric assertions compare against every number in the answer, so formatting
        # ("$1,234.50", "1234.5") never decides a pass.
        seen_numbers = _numbers(haystack)
        missing_numbers = [
            spec
            for spec in must_approx
            if not any(abs(n - spec["value"]) <= max(spec["tolerance"], abs(spec["value"]) * spec["tolerance"])
                       for n in seen_numbers)
        ]

        checks = len(must) + len(must_not) + len(must_approx)
        failures = len(missing) + len(present) + len(missing_numbers)

        return Score(
            value=clamp((checks - failures) / checks) if checks else 0.0,
            answer=f"{checks - failures}/{checks}",
            explanation=(
                (f"missing: {missing}; " if missing else "")
                + (f"forbidden present: {present}; " if present else "")
                + (f"numbers not found within tolerance: {missing_numbers}" if missing_numbers else "")
            )
            or "all content assertions passed",
            metadata={
                "missing": missing,
                "forbidden_present": present,
                "missing_numbers": missing_numbers,
            },
        )

    return score
