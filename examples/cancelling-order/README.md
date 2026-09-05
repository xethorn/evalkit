# Order Cancellation Example

This example demonstrates evaluating an agent that handles order lookup, multi-turn customer interaction, and order cancellation/refund policies.

## Structure

- `target.py`: Implements `OrderCancelTarget` and `OrderCancelChatDriver`.
- `suite.yaml`: Defines order cancellation scenarios including a multi-turn clarification test.

## Running

```bash
export PYTHONPATH=.
export EVAL_TARGET=examples.cancelling-order.target:cancelling_order_target
evalkit doctor
evalkit run --suite examples/cancelling-order/suite.yaml --mock good
```
