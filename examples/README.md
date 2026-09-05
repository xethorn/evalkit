# evalkit Examples

This folder contains an example showing how to integrate a custom AI agent framework target with `evalkit`.

## Files

- `simple_target.py`: Implements a custom `Target` (`SimpleTarget`) and `ChatDriver` (`SimpleChatDriver`).
- `sample_suite.yaml`: An example dataset/suite with evaluation samples and test assertions.

## Usage

1. Set `PYTHONPATH=.` and `EVAL_TARGET` to point to the custom target instance:

```bash
export PYTHONPATH=.
export EVAL_TARGET=examples.simple_target:my_target
```

2. Run `evalkit doctor` to verify target registration and health checks:

```bash
evalkit doctor
```

3. Run an evaluation using the mock target or custom target:

```bash
evalkit run --mock good
```
