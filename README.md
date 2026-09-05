# evalkit

A local, record-keeping eval framework for conversational agents, built on
[Inspect AI](https://inspect.aisi.org.uk/).

It owns the parts that decide whether a number means anything — suites and templating, the
simulated user, the scorers, run provenance, the paired statistics, the dashboard — and
none of the parts that know what is being evaluated. **Nothing in this package names a
product.** The agent under test sits behind a *target*, resolved at runtime from
`EVAL_TARGET`. `tests/test_boundary.py` fails if that stops being true.

Commands below are written as `evalkit …`. The console script's actual name belongs to
whichever distribution ships the framework, declared in its `[project.scripts]`; messages
the CLI prints use whatever it was invoked as.

## Dependencies

Six, and the package imports exactly these:

```
inspect-ai      logs, viewer, scoring, epochs
pydantic        the trace model
pyyaml          suites
typer, rich     the CLI
python-dotenv   .env
```

No browser driver, and no LLM provider client. `inspect_ai.get_model` loads whichever
provider `EVAL_JUDGE_MODEL` names, so those are an extra — `pip install ".[judges]"` for
`anthropic` and `openai` — and a target declares whatever it needs as an extra of its own
(this repo's browser target adds `playwright` that way). `tests/test_boundary.py` fails if
the framework imports something it does not declare, or declares something it never
imports.

## Writing a target

```python
from evalkit.target import BaseTarget, ChatDriver, SampleContext

class MyTarget(BaseTarget):
    name = "my-agent"
    display_name = "My Agent"

    async def open(self, sample: SampleContext, artifacts_dir, record_trace=True) -> ChatDriver:
        ...   # auth, navigate, return something that can send() and receive turns
```

```bash
EVAL_TARGET=my_package:my_target evalkit doctor
```

That is the minimum. Everything below is optional and defaulted in `BaseTarget`, but each
one buys something the framework cannot work out on its own:

| method | what it buys |
| --- | --- |
| `close`, `capture_failure` | artifacts on failure — a screenshot beats a stack trace |
| `classify_http_error` | failures that invalidate a sample get *excluded*, not scored as the agent failing |
| `agent_failure_in` / `failure_markers` | "I was unable to generate a response" scored as a failure rather than an answer |
| `fingerprint` | what makes two evaluations incomparable (tenant, URL, feature flags) — recorded and diffed |
| `default_repos` | which repos' git state is the code under test |
| `tool_aliases` | suites can name a tool once, however the product spells it on the wire |
| `vocabulary` | `${randomVendor()}` and friends, drawn from *your* world |
| `judge_domain` | what the grader is told it is grading — the judge prompt is the metric |
| `doctor_checks`, `cli` | your own preflight rows and CLI commands |
| `demo_scenario` | your own demo history, if the built-in one does not fit |

## The two protocols

**`ChatDriver`** is one conversation: `start`, `new_chat`, `send`, `approve_interrupt`,
`agent_config`, `diagnostics`. `send` returns a `TurnResult`, which is the whole contract
for "what the agent did". Fill in `text` and you get output grading; fill in `tool_calls`
and `subagents` too and you get process grading — routing, call budgets, loop detection —
for free.

**`Target`** is the product: it opens drivers, says what counts as broken, and declares
what must be recorded.

## Layout

```
target.py       the seam: Target, ChatDriver, TurnResult, SampleContext, HarnessError
registry.py     EVAL_TARGET -> a target, imported lazily and cached
config.py       run shape, judge, paths (EVAL_*; see EVAL_ENV_PREFIXES for fallbacks)
datasets.py     loading and filtering suites
templating.py   seeded ${...} prompts — the engine; the words come from the target
users.py        the simulated user: scripted answers first, a model as fallback
solver.py       the multi-turn loop, recorded end to end
trace.py        the normalized trace every scorer reads
scorers/        rubric judge, assertions, tool calls, routing, budgets, loops, infra
provenance.py   what exactly was under test: repo shas, diffs, a content hash of the ruler
scenario.py     the shape of a demo history
analysis/       store, paired statistics, reports, dashboard (+ demo_blog.py)
  assets/       the page itself: dashboard.html is the shell, .css and .js the rest
mock.py         the framework's own offline target, for self-testing
cli.py          the command surface
```

## Self-test

`evalkit run --mock good|poor` runs the entire loop — solver, scorers, store, statistics,
reports — against `mock.py`, with no target package, no app and no judge. `--mock poor`
violates the declared expectations in the usual ways and `--mock good` satisfies them, so
the gap between two mock evaluations is an improvement of *known* size:

```bash
make smoke   # plants that improvement and checks the statistics detect it
```

If that fails, no number this framework reports can be trusted.

`evalkit demo-data` fabricates a nine-evaluation history — including a repeat that gives a
noise floor, a model-only change, a clean win and a regression a rising average would hide
— so the dashboard can be built and argued with before anyone spends real runs. It is
about a blog-writing assistant, deliberately unrelated to whatever you are evaluating: a
demo that looks like your own results is one somebody eventually screenshots into a
decision.

## License

[MIT](LICENSE)

