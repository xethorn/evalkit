"""Seeded prompt templating.

Eval prompts must vary (so the agent can't be tuned to one literal string) *and* be
reproducible (so a regression can be replayed). Both come from the same place: every
``${...}`` expression is resolved by a per-sample seeded RNG, and every resolution is
recorded as a binding that gets stored alongside the run.

    >>> r = render("Create an expense report for ${newDate('Month YYYY')}", seed=7)
    >>> r.text
    'Create an expense report for March 2026'
    >>> r.bindings
    {'newDate(\\'Month YYYY\\')': 'March 2026'}

Named bindings let one value appear in several places, including in the expected
answer of a scorer::

    Create an expense report for ${newDate('Month YYYY') as period}, and title it "${period}"
"""

from __future__ import annotations

import ast
import calendar
import random
import re
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

EXPR_RE = re.compile(r"\$\{([^{}]+)\}")
CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AS_RE = re.compile(r"^(?P<expr>.*?)\s+as\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$", re.DOTALL)

MONTHS = list(calendar.month_name)[1:]
MONTHS_ABBR = list(calendar.month_abbr)[1:]

# Longest-first so 'YYYY' wins over 'YY' and 'MMMM' over 'MM'.
_FORMAT_TOKENS = [
    ("YYYY", lambda d: f"{d.year:04d}"),
    ("YY", lambda d: f"{d.year % 100:02d}"),
    ("MMMM", lambda d: MONTHS[d.month - 1]),
    ("Month", lambda d: MONTHS[d.month - 1]),
    ("MMM", lambda d: MONTHS_ABBR[d.month - 1]),
    ("Mon", lambda d: MONTHS_ABBR[d.month - 1]),
    ("MM", lambda d: f"{d.month:02d}"),
    ("DD", lambda d: f"{d.day:02d}"),
    ("Qq", lambda d: f"Q{(d.month - 1) // 3 + 1}"),
    ("Q", lambda d: f"Q{(d.month - 1) // 3 + 1}"),
    ("M", lambda d: str(d.month)),
    ("D", lambda d: str(d.day)),
]


def format_date(d: date, fmt: str) -> str:
    """Format ``d`` using human tokens (``Month YYYY``, ``YYYY-MM-DD``, ``Q YYYY``...)."""
    out: list[str] = []
    i = 0
    while i < len(fmt):
        for token, fn in _FORMAT_TOKENS:
            if fmt.startswith(token, i):
                out.append(fn(d))
                i += len(token)
                break
        else:
            out.append(fmt[i])
            i += 1
    return "".join(out)


@dataclass
class RenderResult:
    text: str
    bindings: dict[str, str] = field(default_factory=dict)
    seed: int = 0


class TemplateError(ValueError):
    pass


class _Functions:
    """The expression vocabulary available inside ``${...}``.

    Every function draws from ``self.rng`` only, so a seed fully determines the prompt.
    ``today`` is injected rather than read from the clock: a stored run replays identically.
    """

    def __init__(
        self,
        rng: random.Random,
        today: date,
        months_back: int = 18,
        vocabulary: dict[str, list[str]] | None = None,
    ):
        self.rng = rng
        self.today = today
        self.vocabulary = vocabulary or {}
        # How far back random dates may reach. Set from the tenant's actual data window:
        # a prompt about a month with no data measures the fixtures, not the agent, and
        # reads as a hallucination test that the agent passes for the wrong reason.
        self.months_back = months_back

    # -- dates -------------------------------------------------------------
    def newDate(self, fmt: str = "YYYY-MM-DD", *, max_months_back: int | None = None) -> str:
        """A random past date, biased to complete months (typical accounting periods)."""
        months_back = self.rng.randint(1, max_months_back or self.months_back)
        d = _add_months(self.today, -months_back)
        day = self.rng.randint(1, calendar.monthrange(d.year, d.month)[1])
        return format_date(d.replace(day=day), fmt)

    def newMonth(self, fmt: str = "Month YYYY", *, max_months_back: int | None = None) -> str:
        """A random past month (day pinned to the 1st so 'Month YYYY' is unambiguous)."""
        months_back = self.rng.randint(1, max_months_back or self.months_back)
        return format_date(_add_months(self.today, -months_back), fmt)

    def monthsAgo(self, n: int, fmt: str = "Month YYYY") -> str:
        return format_date(_add_months(self.today, -int(n)), fmt)

    def daysAgo(self, n: int, fmt: str = "YYYY-MM-DD") -> str:
        return format_date(self.today - timedelta(days=int(n)), fmt)

    def todayDate(self, fmt: str = "YYYY-MM-DD") -> str:
        return format_date(self.today, fmt)

    def lastQuarter(self, fmt: str = "Qq YYYY") -> str:
        q_start_month = ((self.today.month - 1) // 3) * 3 + 1
        prev = _add_months(self.today.replace(day=1, month=q_start_month), -3)
        return format_date(prev, fmt)

    # -- scalars -----------------------------------------------------------
    def randomInt(self, low: int, high: int) -> str:
        return str(self.rng.randint(int(low), int(high)))

    def randomAmount(self, low: float = 50, high: float = 5000, decimals: int = 2) -> str:
        return f"{self.rng.uniform(float(low), float(high)):.{int(decimals)}f}"

    def choice(self, *options: str) -> str:
        if not options:
            raise TemplateError("choice() needs at least one option")
        return str(self.rng.choice(list(options)))

    def uuid(self) -> str:
        return str(_uuid.UUID(int=self.rng.getrandbits(128), version=4))

    # -- domain vocabulary -------------------------------------------------
    # Not defined here. A vendor list is the product's world, not the framework's, so the
    # target supplies its own named lists and each becomes a `random<Name>()` generator:
    # {"vendor": [...]} gives a suite `${randomVendor()}`. See `_vocabulary_functions`.

    def _vocabulary_functions(self) -> dict[str, object]:
        return {
            f"random{name[:1].upper()}{name[1:]}": (lambda options=options: str(self.rng.choice(options)))
            for name, options in self.vocabulary.items()
            if options
        }


def _add_months(d: date, months: int) -> date:
    total = (d.year * 12 + d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _parse_args(raw: str) -> tuple[list, dict]:
    raw = raw.strip()
    if not raw:
        return [], {}
    try:
        call = ast.parse(f"f({raw})", mode="eval").body
    except SyntaxError as exc:  # pragma: no cover - surfaced as TemplateError
        raise TemplateError(f"cannot parse arguments: {raw!r}") from exc
    args = [ast.literal_eval(a) for a in call.args]  # type: ignore[union-attr]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}  # type: ignore[union-attr]
    return args, kwargs


def render(
    template: str,
    seed: int,
    today: date | None = None,
    bindings: dict[str, str] | None = None,
    months_back: int = 18,
    vocabulary: dict[str, list[str]] | None = None,
) -> RenderResult:
    """Resolve every ``${...}`` in ``template`` with a seeded RNG.

    ``bindings`` may pre-seed named values (used to render a sample's expected answer
    with the *same* values that went into its prompt).
    """
    rng = random.Random(seed)
    fns = _Functions(
        rng,
        today or date.today(),
        months_back=months_back,
        vocabulary=target_vocabulary() if vocabulary is None else vocabulary,
    )
    vocab_fns = fns._vocabulary_functions()
    resolved: dict[str, str] = dict(bindings or {})

    def substitute(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        name: str | None = None
        if (m := AS_RE.match(raw)) is not None:
            raw, name = m.group("expr").strip(), m.group("name")

        if IDENT_RE.match(raw):
            if raw not in resolved:
                raise TemplateError(f"unknown binding ${{{raw}}} (bind it earlier with `as {raw}`)")
            value = resolved[raw]
        else:
            call = CALL_RE.match(raw)
            if not call:
                raise TemplateError(f"unsupported expression ${{{raw}}}")
            fname, arg_src = call.group(1), call.group(2)
            fn = None if fname.startswith("_") else getattr(fns, fname, None) or vocab_fns.get(fname)
            if fn is None:
                raise TemplateError(
                    f"unknown template function {fname!r}; available: "
                    f"{', '.join(available(fns.vocabulary))}"
                )
            args, kwargs = _parse_args(arg_src)
            value = str(fn(*args, **kwargs))
            resolved[raw] = value

        if name:
            resolved[name] = value
        return value

    text = EXPR_RE.sub(substitute, template)
    return RenderResult(text=text, bindings=resolved, seed=seed)


def target_vocabulary() -> dict[str, list[str]]:
    """The word lists the configured target contributes.

    Resolved here rather than threaded through every caller, and never fatal: a suite that
    uses no vocabulary must still render when no target is installed.
    """
    try:
        from .config import settings

        return settings().target.vocabulary()
    except Exception:
        return {}


def available(vocabulary: dict[str, list[str]] | None = None) -> list[str]:
    """Every function a template may call: the built-ins plus the target's vocabulary."""
    builtins = {
        n for n in dir(_Functions) if not n.startswith("_") and n not in {"rng", "today", "vocabulary"}
    }
    vocab = target_vocabulary() if vocabulary is None else vocabulary
    return sorted(builtins | {f"random{name[:1].upper()}{name[1:]}" for name in vocab if vocab[name]})


def sample_seed(base_seed: int, sample_id: str, epoch: int = 0) -> int:
    """Deterministic per-(run, sample, epoch) seed, stable across suite reordering."""
    import hashlib

    h = hashlib.sha256(f"{base_seed}:{sample_id}:{epoch}".encode()).digest()
    return int.from_bytes(h[:8], "big")
