# CRM API Integration Example

This example shows how to evaluate an agent that interacts directly with a CRM API (`/customers`).

## Structure

- `target.py`: Implements `CrmTarget` and `CrmChatDriver` which map `/customers` API interactions to `ToolCall` events.
- `suite.yaml`: Evaluation suite asserting customer search and birthday update API operations.

## Running

```bash
export PYTHONPATH=.
export EVAL_TARGET=examples.crm.target:crm_target
evalkit doctor
evalkit run --suite examples/crm/suite.yaml --mock good
```
