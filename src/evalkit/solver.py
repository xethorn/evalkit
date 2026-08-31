"""The solver: drive one sample through the agent under test and record its trace.

The agent lives behind a target, not behind a model call, so this solver replaces
Inspect's usual "call the model" step entirely. What it produces is the same shape
Inspect expects — messages and an output — plus a full ``AgenticTrace`` in the sample
store, which is what the tool-call and trace scorers grade.

Nothing here knows what the agent *is*. It opens a conversation through the target, runs
the multi-turn loop against the simulated user, and asks the target to classify whatever
went wrong. Swapping the product under test swaps the target and leaves this file alone.

Failure handling is the important detail: a timeout, an expired session or a 500 from the
app is recorded as an *infra error* and the sample is marked accordingly. Those samples
are excluded from quality metrics rather than counted as agent failures, because a suite
that scores infra flakiness cannot be hill-climbed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import span

from .config import Settings, env, settings
from .target import ChatDriver, HarnessError, SampleContext, Target, TurnResult, turn_records
from .trace import AgenticTrace, InfraError, SseEvent
from .users import UserPolicy, next_reply

TRACE_KEY = "trace"
INFRA_KEY = "infra_error"


def run_id() -> str:
    return env("RUN_ID", "adhoc")


@solver
def agent_conversation(
    cfg: Settings | None = None,
    target: Target | None = None,
    simulated_user_model: str | None = None,
    record_trace: bool = True,
) -> Solver:
    """A multi-turn conversation with the agent under test, recorded end to end."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        conf = cfg or settings()
        tgt = target or conf.target
        meta = state.metadata or {}
        policy = UserPolicy.from_dict(meta.get("user"))
        policy.max_turns = min(policy.max_turns, conf.run.max_user_turns)

        artifacts_dir = conf.run.runs_dir / run_id() / "artifacts" / _safe(str(state.sample_id)) / f"epoch{state.epoch}"
        trace = AgenticTrace(sample_id=str(state.sample_id), prompt=state.input_text)
        started = time.monotonic()

        user_messages: list[tuple[int, str, str]] = []
        results: list[TurnResult] = []
        driver: ChatDriver | None = None
        failed = False

        user_model = None
        model_name = policy.model or simulated_user_model
        if model_name:
            user_model = get_model(model_name)

        try:
            sample = SampleContext(sample_id=str(state.sample_id), epoch=state.epoch, metadata=meta)
            driver = await tgt.open(sample, artifacts_dir, record_trace=record_trace)
            await driver.start()
            await driver.new_chat()

            message = state.input_text
            origin = "prompt"
            turns_used = 0
            while turns_used < policy.max_turns:
                turns_used += 1
                async with span(f"user message {turns_used}", type="user"):
                    transcript().info({"origin": origin, "message": message})
                result = await driver.send(message)
                user_messages.append((result.index, message, origin))
                results.append(result)
                await _log_turn(result)

                if result.errored:
                    trace.infra_errors.append(
                        InfraError(
                            kind="stream_error",
                            message="agent stream emitted an error event",
                            turn=result.index,
                        )
                    )
                    break

                # An approval request is the agent asking permission. Approving through
                # the product's own affordance produces another agent turn, which we
                # record like any other.
                while result.interrupt is not None and policy.interrupts == "approve":
                    approved = await driver.approve_interrupt()
                    result.interrupt.decision = "approved" if approved else "no-card-on-screen"
                    if approved is None:
                        break
                    user_messages.append((approved.index, "[approved via UI]", "interrupt-approval"))
                    results.append(approved)
                    await _log_turn(approved)
                    result = approved
                    if result.errored:
                        break

                if result.errored or not result.asks_question:
                    break

                reply = await next_reply(
                    policy,
                    question=result.text or result.rendered_text,
                    transcript=_transcript_so_far(user_messages, results),
                    tool=result.interrupt.tool if result.interrupt else None,
                    model=user_model,
                )
                if reply.is_stop:
                    break
                message, origin = reply.text, reply.origin
            else:
                # Ran out of turns with the agent still asking. Recorded as a trace note
                # rather than an infra error: an agent that never converges is a finding
                # about the agent, and the `converges` scorer is what grades it.
                if results and results[-1].asks_question:
                    trace.unconverged = True
                    trace.notes.append(f"hit max_turns={policy.max_turns} with a question still open")

            diagnostics = await driver.diagnostics()
            trace.agent_config = await driver.agent_config()
            trace.turns = turn_records(results, user_messages)
            trace.chat_id = getattr(driver, "chat_id", None)
            for result in results:
                trace.tool_calls.extend(result.tool_calls)
                trace.subagents.extend(result.subagents)
                trace.sse_events.extend(_compact(result.events))
                if result.interrupt:
                    trace.interrupts.append(result.interrupt)
                if result.dag and result.trace_id:
                    trace.dags[result.trace_id] = result.dag
            for turn in trace.turns:
                if turn.role != "assistant":
                    continue
                marker = tgt.agent_failure_in(turn.text)
                if marker:
                    trace.agent_errors.append(f"turn {turn.index}: {marker!r}")
            trace.truncated_events = int(diagnostics.get("truncated") or 0)
            # A target may want something on the record that is not an error — "this run
            # never contacted the app", say. Notes are shown but never scored.
            trace.notes.extend(str(n) for n in (diagnostics.get("notes") or []))
            for err in diagnostics.get("httpErrors", []):
                classified = tgt.classify_http_error(int(err.get("status") or 0), str(err.get("url") or ""))
                if classified is None:
                    # Noise, not a failure. Recorded as a note so it stays visible without
                    # excluding the sample.
                    trace.notes.append(f"benign HTTP {err.get('status')} {err.get('url')}")
                    continue
                trace.infra_errors.append(
                    InfraError(kind=classified, message=str(err.get("url")), turn=err.get("turn"))
                )

        except HarnessError as exc:
            failed = True
            trace.infra_errors.append(InfraError(kind=getattr(exc, "kind", "harness"), message=str(exc)))
            if driver is not None:
                trace.artifacts.update(await tgt.capture_failure(driver, "harness-error"))
        except Exception as exc:  # unexpected: still an infra error, but flagged loudly
            failed = True
            trace.infra_errors.append(InfraError(kind="unexpected", message=f"{type(exc).__name__}: {exc}"))
            if driver is not None:
                trace.artifacts.update(await tgt.capture_failure(driver, "unexpected-error"))
        finally:
            if driver is not None:
                trace.artifacts.update(await tgt.close(driver, save_trace=record_trace or failed, failed=failed))

        trace.total_ms = int((time.monotonic() - started) * 1000)
        _persist(trace, conf, state)

        # Hand Inspect a normal-looking conversation so built-in viewers and model-graded
        # scorers work unchanged.
        state.messages = []
        for turn in trace.turns:
            if turn.role == "user":
                state.messages.append(ChatMessageUser(content=turn.text))
            else:
                state.messages.append(ChatMessageAssistant(content=turn.text))
        state.store.set(TRACE_KEY, trace.model_dump(mode="json"))
        state.store.set(INFRA_KEY, [e.model_dump() for e in trace.infra_errors])
        state.metadata["trace_summary"] = {
            "turns": len(trace.turns),
            "tools": trace.tool_names,
            "subagents": trace.subagent_names,
            "interrupts": len(trace.interrupts),
            "total_ms": trace.total_ms,
            "infra_errors": [e.kind for e in trace.infra_errors],
            "agent_errors": trace.agent_errors,
            "chat_id": trace.chat_id,
            "agent_config": trace.agent_config,
        }
        state.output.completion = trace.final_answer
        state.completed = True
        return state

    return solve


async def _log_turn(result: TurnResult) -> None:
    async with span(f"agent turn {result.index}", type="agent"):
        transcript().info(
            {
                "latency_ms": result.latency_ms,
                "first_token_ms": result.first_token_ms,
                "trace_id": result.trace_id,
                "tool_calls": [c.model_dump() for c in result.tool_calls],
                "subagents": [a.name for a in result.subagents],
                "interrupt": result.interrupt.model_dump() if result.interrupt else None,
                "tables": result.tables,
                "plots": result.plots,
                "text": result.text[:4000],
            }
        )


def _transcript_so_far(user_messages: list[tuple[int, str, str]], results: list[TurnResult]) -> str:
    lines = []
    by_index = {r.index: r for r in results}
    for index, text, _origin in user_messages:
        lines.append(f"USER: {text}")
        r = by_index.get(index)
        if r:
            lines.append(f"ASSISTANT: {r.text or r.rendered_text}")
    return "\n\n".join(lines)


def _compact(events: list[SseEvent]) -> list[SseEvent]:
    """Trim the event log before it is persisted.

    Token events are collapsed to one marker per turn: their content is already in the
    answer, and keeping thousands of them makes stored traces unreadable.
    """
    out: list[SseEvent] = []
    token_count = 0
    first_token: SseEvent | None = None
    for ev in events:
        if ev.event == "token":
            token_count += 1
            first_token = first_token or ev
            continue
        out.append(ev)
    if first_token is not None:
        out.append(
            SseEvent(
                turn=first_token.turn,
                at_ms=first_token.at_ms,
                event="tokens",
                data={"count": token_count, "first_token_at_ms": first_token.at_ms},
            )
        )
    return sorted(out, key=lambda e: (e.turn, e.at_ms))


def _persist(trace: AgenticTrace, conf: Settings, state: TaskState) -> None:
    """Write the trace next to the run's artifacts, and into the Inspect transcript."""
    path = conf.run.runs_dir / run_id() / "traces" / f"{_safe(str(state.sample_id))}.epoch{state.epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.model_dump(mode="json"), indent=2))
    trace.artifacts["trace_json"] = str(path)
    transcript().info(
        {
            "agentic_trace": {
                "tools": trace.tool_names,
                "subagents": trace.subagent_names,
                "interrupts": [i.model_dump() for i in trace.interrupts],
                "infra_errors": [e.model_dump() for e in trace.infra_errors],
                "trace_json": str(path),
                "dag_trace_ids": list(trace.dags),
            }
        },
        source="evalkit",
    )


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)[:80]


def trace_from_state(state: TaskState) -> AgenticTrace:
    """Rehydrate the trace inside a scorer."""
    raw: Any = state.store.get(TRACE_KEY)
    if not raw:
        return AgenticTrace(sample_id=str(state.sample_id))
    return AgenticTrace.model_validate(raw)
