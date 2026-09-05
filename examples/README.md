# evalkit Examples

This repository contains modular examples demonstrating how to integrate custom AI agent targets with `evalkit`, define evaluation suites, handle multi-turn user interactions, enforce process tool expectations, and ingest telemetry traces.

## Examples Overview

Each example lives in its own dedicated directory with its target driver, evaluation suite configuration, and documentation:

### 1. `examples/crm/`
- **Focus**: REST API / CRM Product Integration (`/customers`).
- **Target**: `examples.crm.target:crm_target`
- **Key Concepts**: Customer search and birthday updates via HTTP API calls (`POST /customers/{id}/birthday`), tool process matching, and custom target preflight checks (`evalkit doctor`).

### 2. `examples/cancelling-order/`
- **Focus**: Customer Support Agent & Order Management.
- **Target**: `examples.cancelling-order.target:cancelling_order_target`
- **Key Concepts**: Multi-turn user simulation (`user:` block with persona, facts, and scripted regex answers), order lookup, cancellation, and refund policy evaluation.

### 3. `examples/ui-tracing/`
- **Focus**: Browser UI Driver & Vendor Trace Telemetry Ingestion.
- **Target**: `examples.ui-tracing.target:ui_tracing_target`
- **Key Concepts**: Automating a web application via Playwright while ingesting trace spans and tool execution telemetry from vendor platforms like **Braintrust**, LangSmith, or Phoenix.

---

## Suite YAML Structure

Suite files define evaluation metadata and tasks:

```yaml
suite:
  name: example-suite-name
  description: "Description of what this evaluation suite tests."
  version: 1

defaults:
  severity: medium
  tags:
    - customer-support

tasks:
  - id: example-task-id
    prompt: "Prompt sent to the agent..."
    expect:
      must_contain:
        - "Expected response string"
      rubric: |
        - Criterion evaluated by the LLM judge.
      tools:
        required:
          - lookup_tool
        forbidden:
          - write_tool
        max_calls:
          lookup_tool: 2
        order:
          - lookup_tool
```

---

## Running the Examples

Set `PYTHONPATH=.` and select the target you want to test via `EVAL_TARGET`:

### Run Preflight Health Checks (`evalkit doctor`)

```bash
# CRM API Target
export PYTHONPATH=.
export EVAL_TARGET=examples.crm.target:crm_target
evalkit doctor

# Order Cancellation Target
export EVAL_TARGET=examples.cancelling-order.target:cancelling_order_target
evalkit doctor

# UI + Braintrust Tracing Target
export EVAL_TARGET=examples.ui-tracing.target:ui_tracing_target
evalkit doctor
```

### Run Suite Evaluations (`evalkit run`)

Run an evaluation against the offline mock engine or a custom suite file:

```bash
# Run CRM suite
evalkit run --suite examples/crm/suite.yaml --mock good

# Run Cancelling Order suite
evalkit run --suite examples/cancelling-order/suite.yaml --mock good

# Run UI Tracing suite
evalkit run --suite examples/ui-tracing/suite.yaml --mock good
```
