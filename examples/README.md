# evalkit Examples

This folder provides a complete example showing how to integrate a custom AI agent framework target with `evalkit`, structure evaluation suites (including multi-turn conversations and tool process checks), and evaluate results.

## Overview of Files

- **`simple_target.py`**: Demonstrates implementing `Target` (`SimpleTarget`) and `ChatDriver` (`SimpleChatDriver`).
  - `judge_domain()` specifies what domain the LLM grader assumes (e.g. `"an AI customer support assistant"`).
  - `tool_aliases()` translates raw tool call names to human-readable names.
  - `vocabulary()` provides domain terms for template expansions (`${randomVendor()}`, `${randomAccount()}`).
  - `fingerprint()` records environment parameters (agent version, tenant ID) for provenance tracking.
  - `doctor_checks()` adds target-specific preflight diagnostics to `evalkit doctor`.
- **`sample_suite.yaml`**: Demonstrates the evaluation suite structure.
  - `suite`: Structured suite metadata (`name`, `description`, `version`).
  - `defaults`: Default severity and tags for all tasks in the file.
  - `tasks`: Individual evaluation cases containing `id`, `prompt`, `severity`, `tags`, `expect` parameters (`must_contain`, `rubric`, `tools`), and optional multi-turn `user` simulation parameters.

## Suite Format and Tool Process Checks

Suite files support both output quality grading and tool execution process checks:

```yaml
suite:
  name: example-customer-support
  description: "Suite description..."
  version: 1

defaults:
  severity: medium

tasks:
  - id: sample-task-1
    prompt: "..."
    expect:
      must_contain:
        - "expected response text"
      rubric: |
        - Criterion 1 evaluated by LLM judge.
      tools:
        required:
          - lookup_account_details
        forbidden:
          - execute_db_write
        max_calls:
          lookup_account_details: 2
        order:
          - lookup_account_details
```

Supported `tools` expectations:
- `required`: Tools that must be invoked.
- `forbidden`: Tools that must not be invoked.
- `max_calls`: Maximum call limits per tool.
- `order`: Subsequence ordering constraint for tool calls.

## Multi-Turn User Simulation

When evaluating agents that ask clarifying questions or require information across multiple turns, `evalkit` provides a simulated user layer configured under the `user:` field in task definitions:

- **`persona`**: Brief background on who the user is.
- **`facts`**: List of facts the simulated user knows.
- **`answers`**: Scripted deterministic responses matched against the agent's questions using regex (`when:`) to keep multi-turn evaluations reproducible across runs.

## How LLM Judges Work

In `evalkit`, quality evaluation uses **LLM-as-a-judge** paired with deterministic assertions:
1. **Assertions**: `must_contain`, `must_not_contain`, and `must_approx` check exact strings or numeric values first.
2. **Rubrics**: `expect.rubric` contains bullet points evaluated by the LLM judge (e.g., Anthropic Claude or OpenAI GPT). The system prompt is automatically customized using `target.judge_domain()` so the judge grades using domain-appropriate criteria.

## Running the Examples

### 1. Preflight Diagnostics (`evalkit doctor`)

Set `PYTHONPATH=.` and `EVAL_TARGET` to point to the example target instance:

```bash
export PYTHONPATH=.
export EVAL_TARGET=examples.simple_target:my_target
evalkit doctor
```

### 2. Running an Offline Self-Test

Run `evalkit` against the offline mock target to verify the pipeline without making live API/app calls:

```bash
evalkit run --mock good
```

To run against a specific suite file using the mock target:

```bash
evalkit run --suite examples/sample_suite.yaml --mock good
```
