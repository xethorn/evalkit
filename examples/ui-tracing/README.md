# UI Driving + Vendor Telemetry Ingestion Example

This example demonstrates driving a web UI session (using Playwright / browser automation) while ingesting structured spans and telemetry tool calls from a vendor observability platform (such as Braintrust).

## Structure

- `target.py`: Implements `UiTracingTarget` and `UiTracingChatDriver` which fetch telemetry spans from Braintrust.
- `suite.yaml`: Evaluation suite asserting browser navigation and DOM click actions.

## Running

```bash
export PYTHONPATH=.
export EVAL_TARGET=examples.ui-tracing.target:ui_tracing_target
evalkit doctor
evalkit run --suite examples/ui-tracing/suite.yaml --mock good
```
