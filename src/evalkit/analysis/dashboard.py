"""The hill-climb console: a static page answering "are we climbing, and what did it?".

The Inspect viewer is a single-run inspector — it explains why one sample behaved as it
did and has no notion of a second run. This page is the other half: it joins runs on
``sample_id`` so you can see *which* samples moved, whether they moved or are merely
flickering, and the git diff that sits between each pair of columns.

Three deliberate choices:

* **The heatmap is the primary view, not the trend line.** A trend line says a number
  moved; only a sample × run grid distinguishes "fixed" from "flaky" from "never worked".
* **The noise floor is drawn.** Runs that share a ``change_id`` tested identical code, so
  the spread between them is this suite's measured noise. Any delta inside that band is
  not an improvement, and the page says so rather than leaving it to the reader.
* **Coverage sits beside every quality number.** A score that rose because half the
  samples stopped being gradeable is the most flattering artifact a harness can produce.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from ..config import REPO_ROOT
from ..provenance import baseline_evaluation
from .compare import LOWER_IS_BETTER, PASS_THRESHOLD
from .store import metric_value, sample_values
from .versions import VersionDiff, load_provenance, version_diff

# The page's own assets. Kept as real .html, .css and .js files rather than string
# literals in this module: they are 1,300 lines between them, and inside a Python string
# an editor cannot highlight them, a linter cannot read them, and a stray quote or brace
# fails at render time instead of in review. The template holds the page shell — head,
# app bar, footer, script tags — and every fragment this module computes goes into it
# through a `{{name}}` slot. Read once at import — the dashboard is regenerated from the
# database on every run, so there is nothing to reload.
ASSETS = Path(__file__).parent / "assets"
TEMPLATE = (ASSETS / "dashboard.html").read_text()
CSS = (ASSETS / "dashboard.css").read_text()
PAGE_JS = (ASSETS / "dashboard.js").read_text()
_DARK_TOKENS = (ASSETS / "dashboard-dark.css").read_text()

# The tool's name on its own page. The framework is not allowed to know what it is
# evaluating (see tests/test_boundary.py), so the heading names the harness and the
# subject sits beside it in smaller type.
BRAND = "evalkit"

_SLOT = re.compile(r"\{\{(\w+)\}\}")


def fill(template: str, **slots: str) -> str:
    """Substitute every ``{{name}}`` slot in the template, in a single pass.

    One pass rather than a chain of ``str.replace``: the stylesheet, the JSON payload and
    a sample's own text all end up on this page, and a value that happens to contain
    ``{{...}}`` must land as content instead of becoming a slot for the next substitution
    to fill. An unknown slot raises here rather than shipping a page with ``{{tabs}}``
    printed across the top of it.
    """

    def one(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in slots:
            raise KeyError(f"dashboard.html asks for {{{{{name}}}}}, which build_dashboard does not provide")
        return slots[name]

    return _SLOT.sub(one, template)


# The dark palette is defined once and used twice: under `prefers-color-scheme` for
# viewers who never touched the toggle, and under an explicit `data-theme`. The stylesheet
# marks the spot with DARKVARS rather than repeating thirty custom properties.
CSS = CSS.replace(
    "DARKVARS",
    "@media (prefers-color-scheme: dark) {\n"
    '  :root:where(:not([data-theme="light"])) {\n'
    "    color-scheme: dark;" + _DARK_TOKENS + "  }\n}\n"
    ':root[data-theme="dark"] {\n  color-scheme: dark;' + _DARK_TOKENS + "}\n",
)

# The scorer whose heatmap opens expanded, and the order panels appear in.
PRIMARY = "rubric_judge"
QUALITY_ORDER = [
    "rubric_judge",
    "assertions",
    "tool_calls",
    "subagents",
    "within_budget",
    "converges",
    "asks_when_underspecified",
]
DIAGNOSTIC_ORDER = ["infra_ok", "agent_error_rate", "coverage", "latency_ms", "tool_call_count", "failed_tool_calls"]


# A mock run sets this judge, which is how the dashboard recognises one.
NO_JUDGE = "none/offline"


@dataclass
class RunColumn:
    run_id: str
    suite: str
    label: str
    created_at: str
    change_id: str
    is_baseline: bool
    epochs: int
    suite_sha: str = ""
    judge_model: str = ""
    harness: str = ""
    # True when this run graded only some of the suite's samples (`--ids`, `--tags`).
    # Such a column is still shown — history is history — but it is excluded from the
    # headline delta and from the noise floor, because its mean is over a different,
    # easier or harder, subset.
    partial: bool = False
    graded: int = 0
    of: int = 0

    @property
    def mock(self) -> bool:
        return self.judge_model == NO_JUDGE

    @property
    def comparable(self) -> bool:
        return not self.mock and not self.partial

    @property
    def short(self) -> str:
        return self.label or self.run_id[:18]

    @property
    def when(self) -> str:
        return (self.created_at or "")[:16].replace("T", " ")


@dataclass
class Cell:
    mean: float | None
    epochs: list[float] = field(default_factory=list)

    @property
    def spread(self) -> str:
        if not self.epochs:
            return "not graded"
        passing = sum(1 for v in self.epochs if v >= PASS_THRESHOLD)
        return f"{passing}/{len(self.epochs)} epochs passed · values {', '.join(f'{v:.2f}' for v in self.epochs)}"

    @property
    def flaky(self) -> bool:
        """Did this sample both pass and fail within a single run?"""
        if len(self.epochs) < 2:
            return False
        return any(v >= PASS_THRESHOLD for v in self.epochs) and any(v < PASS_THRESHOLD for v in self.epochs)


@dataclass
class ScorerGrid:
    scorer: str
    samples: list[str]
    cells: dict[tuple[str, str], Cell]          # (sample_id, run_id) -> cell
    run_means: dict[str, float | None]
    run_stderr: dict[str, float | None]
    coverage: dict[str, float | None]
    noise: float | None
    noise_source: str

    @property
    def lower_is_better(self) -> bool:
        return self.scorer in LOWER_IS_BETTER

    def row_state(self, sample: str, runs: list[RunColumn]) -> tuple[str, str | None]:
        """What this row says, and by how much it moved.

        Returns ``(state, delta)``. The distinction that matters: with only one comparable
        run there is no trajectory, so the state describes the *level* ("passing",
        "partial", "failing") and the delta is ``None``. Calling a single 0.67 measurement
        "never passed" reads as a verdict on the agent when it is only a first reading.

        Only comparable runs count. Reading "improved" across a column that graded a
        different subset, or ran under a different scorer version, would be reading the
        harness's own history as the agent's progress.
        """
        cells = [self.cells.get((sample, r.run_id)) for r in runs if r.comparable]
        graded = [c for c in cells if c is not None and c.mean is not None]
        if not graded:
            return "ungraded", None
        means = [c.mean for c in graded if c.mean is not None]
        latest = means[-1]

        if len(means) < 2:
            if any(c.flaky for c in graded):
                return "flaky", None
            if latest >= PASS_THRESHOLD:
                return "passing", None
            return ("failing" if latest <= 0.001 else "partial"), None

        delta = latest - means[-2]
        signed = -delta if self.lower_is_better else delta
        if any(c.flaky for c in graded):
            return "flaky", f"{delta:+.2f}"
        if abs(delta) <= 0.01:
            return ("passing" if latest >= PASS_THRESHOLD else "flat"), f"{delta:+.2f}"
        return ("improved" if signed > 0 else "regressed"), f"{delta:+.2f}"


def _stderr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return stdev(values) / (len(values) ** 0.5)


def build_grid(conn: sqlite3.Connection, runs: list[RunColumn], scorer: str) -> ScorerGrid:
    per_run: dict[str, dict[str, list[float]]] = {
        r.run_id: sample_values(conn, r.run_id, r.suite, scorer) for r in runs
    }
    samples = sorted({s for values in per_run.values() for s in values})

    cells: dict[tuple[str, str], Cell] = {}
    run_means: dict[str, float | None] = {}
    run_stderr: dict[str, float | None] = {}
    for run in runs:
        means = []
        for sample in samples:
            epochs = per_run[run.run_id].get(sample, [])
            cell = Cell(mean=fmean(epochs) if epochs else None, epochs=epochs)
            cells[(sample, run.run_id)] = cell
            if cell.mean is not None:
                means.append(cell.mean)
        run_means[run.run_id] = fmean(means) if means else None
        run_stderr[run.run_id] = _stderr(means)

    coverage = {r.run_id: metric_value(conn, r.run_id, r.suite, scorer, "coverage") for r in runs}
    noise, source = noise_floor(runs, cells, samples)
    return ScorerGrid(scorer, samples, cells, run_means, run_stderr, coverage, noise, source)


def noise_floor(
    runs: list[RunColumn], cells: dict[tuple[str, str], Cell], samples: list[str]
) -> tuple[float | None, str]:
    """The suite's own noise, measured from runs that tested identical code.

    Without this number an improvement cannot be told from a re-roll. It is deliberately
    measured rather than assumed: two runs sharing a ``change_id`` differ only by sampling,
    so the spread of their per-sample deltas *is* the noise.
    """
    # Every axis of the measurement must match, not just the code. Two runs that share a
    # change_id but differ in suite hash, judge, harness version or graded sample set are
    # not a repeat measurement — pooling them reports the harness's own churn as the
    # agent's noise, which is how a suite ends up with a noise floor wide enough to hide
    # any regression.
    candidates = [r for r in runs if r.comparable]
    pairs: list[tuple[RunColumn, RunColumn]] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            same = (
                a.change_id
                and a.change_id == b.change_id
                and a.suite_sha == b.suite_sha
                and a.judge_model == b.judge_model
                and a.harness == b.harness
            )
            if same:
                pairs.append((a, b))
    if not pairs:
        return None, (
            "not measured — no two runs yet share the same code, suite, judge and harness. "
            "Run the same suite twice without changing anything to measure it."
        )

    deltas: list[float] = []
    for a, b in pairs:
        for sample in samples:
            ca, cb = cells.get((sample, a.run_id)), cells.get((sample, b.run_id))
            if ca and cb and ca.mean is not None and cb.mean is not None:
                deltas.append(cb.mean - ca.mean)
    if len(deltas) < 2:
        return None, "not measured — too few paired samples across same-code runs"
    band = 1.96 * (stdev(deltas) / (len(deltas) ** 0.5))
    return band, f"measured from {len(pairs)} same-code run pair(s), {len(deltas)} paired samples"


# --------------------------------------------------------------------------- render
def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# Sequential ramp, seven steps, referenced through CSS variables so each theme runs it in
# the direction that puts HIGH at the prominent end.
#
# This is the subtle trap in a themed heatmap: on a light surface "high" must be the dark,
# saturated step, but on a dark surface that same step recedes into the background while
# the pale low step glows. Reusing one set of hexes makes 0.00 the loudest cell on the
# page and 1.00 nearly invisible — the scale reads backwards to anyone glancing at it.
RAMP_STEPS = 7


def ramp_step(value: float, lower_is_better: bool = False) -> int:
    """Return the ramp index (0 = worst, 6 = best) for a 0-1 score."""
    v = 1.0 - value if lower_is_better else value
    return min(RAMP_STEPS - 1, max(0, round(max(0.0, min(1.0, v)) * (RAMP_STEPS - 1))))


# Labels are honest about what one measurement can support. "passing" / "partial" /
# "failing" describe a level; only "improved" / "regressed" / "flat" claim a direction.
VERDICT_LABEL = {
    "improved": ("improved", "good"),
    "regressed": ("regressed", "critical"),
    "flat": ("unchanged", "neutral"),
    "flaky": ("flaky", "warning"),
    "passing": ("passing", "good"),
    "partial": ("partial credit", "serious"),
    "failing": ("failing", "critical"),
    "ungraded": ("not graded", "neutral"),
}


def _sparkline(values: list[float | None], width: int = 160, height: int = 34) -> str:
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < 2:
        return '<span class="muted">—</span>'
    lo = min(v for _, v in points)
    hi = max(v for _, v in points)
    span = (hi - lo) or 1.0
    n = max(1, len(values) - 1)

    def xy(i: int, v: float) -> tuple[float, float]:
        return (6 + i / n * (width - 12), height - 6 - (v - lo) / span * (height - 12))

    path = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(xy(i, v) for i, v in points))
    last_x, last_y = xy(*points[-1])
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-hidden="true"><path d="{path}" fill="none" stroke="var(--series-1)" '
        f'stroke-width="2" stroke-linecap="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" fill="var(--series-1)"/></svg>'
    )


def _axis_label(run: RunColumn, width: int = 22) -> str:
    """Short but still distinguishable — two evaluations of the same variation share a
    label prefix, and truncating both to the same string makes the axis useless."""
    label = run.short.replace("DEMO ", "")
    return label if len(label) <= width else label[: width - 1] + "\u2026"


def trend_chart(grid: ScorerGrid, runs: list[RunColumn], baseline_value: float | None = None) -> str:
    """Mean per run with ±stderr, over the measured noise band.

    Only comparable runs are plotted. With fewer than two of them there is no trend, and
    the page says that instead of drawing a line through a single point.
    """
    comparable = [r for r in runs if r.comparable and grid.run_means[r.run_id] is not None]
    if len(comparable) < 2:
        omitted = len([r for r in runs if not r.comparable])
        return (
            '<p class="note">Only one comparable run so far, so there is no trend to draw yet. '
            f"{omitted} earlier column(s) are excluded from the trend because they graded a "
            "different subset of the suite or ran under a different judge or harness version. "
            "Run this suite again, unchanged, to establish the noise floor.</p>"
        )
    # Wider than the old 720, and the labels rotate: a hill accumulates evaluations, and
    # horizontal labels collided into an unreadable smear at nine of them.
    width, height = 1600, 300
    pad_l, pad_r, pad_t, pad_b = 100, 110, 22, 88
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = max(1, len(runs) - 1)

    def x(i: int) -> float:
        return pad_l + (i / n) * plot_w if len(runs) > 1 else pad_l + plot_w / 2

    def y(v: float) -> float:
        return pad_t + (1 - max(0.0, min(1.0, v))) * plot_h

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'data-pad-t="{pad_t}" data-plot-h="{plot_h}" data-pad-l="{pad_l}" data-plot-w="{plot_w}" '
        f'data-scorer="{esc(grid.scorer)}" '
        f'aria-label="{esc(grid.scorer)} mean per evaluation, with the noise floor and the '
        'selected baseline and candidate marked">'
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{pad_l}" y1="{y(tick):.1f}" x2="{pad_l + plot_w}" y2="{y(tick):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y(tick) + 4:.1f}" class="tick" text-anchor="end">{tick:.2f}</text>'
        )

    # The noise band is centred on the baseline, because that is the value every later
    # evaluation is judged against. Centring it on the first point instead makes the band
    # sit far from the comparison it is supposed to inform.
    first = baseline_value if baseline_value is not None else next(
        (grid.run_means[r.run_id] for r in runs if grid.run_means[r.run_id] is not None), None
    )
    if grid.noise and first is not None:
        top, bottom = y(first + grid.noise), y(first - grid.noise)
        parts.append(
            f'<rect data-role="noise-band" data-noise="{grid.noise:.6f}" '
            f'x="{pad_l}" y="{top:.1f}" width="{plot_w}" height="{abs(bottom - top):.1f}" '
            f'fill="var(--noise)" stroke="none"><title>Noise floor: ±{grid.noise:.3f} '
            f'({esc(grid.noise_source)}). A change inside this band is not a change.</title></rect>'
        )

    graded = [
        (i, r, grid.run_means[r.run_id])
        for i, r in enumerate(runs)
        if grid.run_means[r.run_id] is not None and r.comparable
    ]
    if len(graded) > 1:
        path = " ".join(
            f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for k, (i, _r, v) in enumerate(graded) if v is not None
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="var(--series-1)" '
            'stroke-width="2" stroke-linejoin="round"/>'
        )

    for i, run, value in graded:
        assert value is not None
        err = grid.run_stderr[run.run_id]
        if err:
            parts.append(
                f'<line x1="{x(i):.1f}" y1="{y(value + err):.1f}" x2="{x(i):.1f}" y2="{y(value - err):.1f}" '
                f'stroke="var(--series-1)" stroke-width="1.5" opacity="0.5"/>'
            )
        parts.append(
            f'<circle class="pt" data-eval="{esc(run.run_id)}" data-value="{value:.6f}" '
            f'cx="{x(i):.1f}" cy="{y(value):.1f}" r="5" fill="var(--series-1)" '
            f'stroke="var(--surface)" stroke-width="2"><title>{esc(run.short)}: {value:.3f}'
            f'{f" ±{err:.3f}" if err else ""} · {esc(run.when)}</title></circle>'
        )
        parts.append(
            f'<text x="{x(i):.1f}" y="{height - pad_b + 14:.1f}" class="tick" '
            f'text-anchor="end" transform="rotate(-32 {x(i):.1f} {height - pad_b + 14:.1f})">'
            f"{esc(_axis_label(run))}</text>"
        )

    if graded:
        i, _run, value = graded[-1]
        assert value is not None
        parts.append(
            f'<text x="{x(i) + 12:.1f}" y="{y(value) + 4:.1f}" class="endpoint">{value:.3f}</text>'
        )
    parts.append(
        f'<line data-role="guide-a" class="guide a" x1="0" y1="{pad_t}" x2="0" '
        f'y2="{pad_t + plot_h}" hidden></line>'
        f'<text data-role="guide-a-label" class="guide-label a" x="0" y="{pad_t - 4}" '
        'text-anchor="middle" hidden>baseline</text>'
        f'<line data-role="guide-b" class="guide b" x1="0" y1="{pad_t}" x2="0" '
        f'y2="{pad_t + plot_h}" hidden></line>'
        f'<text data-role="guide-b-label" class="guide-label b" x="0" y="{pad_t - 4}" '
        'text-anchor="middle" hidden>candidate</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def big_grid(grids: dict[str, ScorerGrid], runs: list[RunColumn]) -> str:
    """Every scorer in one table, grouped by scorer with a separator row between groups.

    Six separate panels made the same six evaluations repeat their headers six times and
    forced the reader to re-orient at each one. One table keeps the evaluation columns in a
    single alignment all the way down, which is what makes a hill legible.

    The sample column pins to the left and change/reading pin to the right, because a hill
    climb accumulates evaluations without bound: the columns scroll, the things you read
    them against do not.
    """
    body: list[str] = []
    for scorer, grid in grids.items():
        span = len(runs) + 3
        body.append(
            f'<tr class="group"><th scope="rowgroup" colspan="{span}">'
            # The label is pinned, not the cell: a sticky element that already spans the
            # full table width has nothing to slide against, so the text scrolls away and
            # leaves an empty band.
            f'<span class="group-inner"><span class="group-name">{esc(scorer)}</span>'
            f'<span class="group-note">{esc(_scorer_note(grid))}</span></span></th></tr>'
        )
        for sample in grid.samples:
            cells = []
            for run in runs:
                cell = grid.cells.get((sample, run.run_id))
                if cell is None or cell.mean is None:
                    cells.append(
                        f'<td class="cell absent" data-eval="{esc(run.run_id)}" '
                        f'data-scorer="{esc(scorer)}">'
                        '<span aria-label="not graded">—</span>'
                        "<title>not graded in this evaluation</title></td>"
                    )
                    continue
                step = ramp_step(cell.mean, grid.lower_is_better)
                flag = '<span class="flaky-dot" aria-hidden="true"></span>' if cell.flaky else ""
                fail = " fail" if cell.mean <= 0.001 and not grid.lower_is_better else ""
                display = _fmt_signal(cell.mean, scorer)
                dim = "" if run.comparable else " dim"
                cells.append(
                    f'<td class="cell s{step}{fail}{dim}" data-eval="{esc(run.run_id)}" '
                    f'data-value="{cell.mean:.6f}" data-scorer="{esc(scorer)}">'
                    f"<span>{display}</span>{flag}"
                    f"<title>{esc(sample)} · {esc(run.short)} — {esc(cell.spread)}</title></td>"
                )
            body.append(
                f'<tr data-sample="{esc(sample)}" data-scorer="{esc(scorer)}">'
                f'<th scope="row" class="rowhead"><code>{esc(sample)}</code></th>'
                f'{"".join(cells)}'
                '<td class="delta" data-role="change">–</td>'
                '<td class="verdict" data-role="reading"></td></tr>'
            )
        means = "".join(
            f'<td class="cell foot{"" if r.comparable else " dim"}" data-eval="{esc(r.run_id)}" '
            f'data-scorer="{esc(scorer)}"'
            + (
                f' data-value="{grid.run_means[r.run_id]:.6f}"'
                if grid.run_means[r.run_id] is not None
                else ""
            )
            + ">"
            f'{"–" if grid.run_means[r.run_id] is None else _fmt_signal(grid.run_means[r.run_id], scorer)}</td>'
            for r in runs
        )
        body.append(
            f'<tr class="summary"><th scope="row" class="rowhead">{esc(scorer)} mean</th>{means}'
            '<td class="delta" data-role="change" data-summary="1">–</td>'
            '<td class="verdict" data-role="reading" data-summary="1"></td></tr>'
        )

    # No controls in the header: both sides are chosen in the app bar, which is always on
    # screen, so per-column buttons were a second way to do the same thing. The columns
    # still *show* the selection, in the same two colours the chart guides use.
    head = "".join(
        f'<th scope="col" class="{"" if r.comparable else "dim"}" data-eval="{esc(r.run_id)}">'
        f'<span class="run-label">{esc(r.short)}</span>'
        f'<span class="run-meta">{esc(_variation_label(r.change_id)[:16]) or "—"}</span>'
        + (
            f'<span class="run-flag">{"mock" if r.mock else f"only {r.graded}/{r.of}"}</span>'
            if not r.comparable
            else ""
        )
        + "</th>"
        for r in runs
    )
    # Explicit column widths, because auto table layout hands every spare pixel to the
    # first column: with two evaluations on screen the sample labels were wider than the
    # comparison they label. Fixed layout gives the labels what they need and splits the
    # rest between the evaluations — and the min-width keeps a long hill scrolling rather
    # than squeezing twenty columns into the viewport.
    return (
        '<div class="scroll bigscroll">'
        f'<table class="grid bigtable" data-fixed="{ROWHEAD_W + DELTA_W + VERDICT_W}" '
        f'data-evalmin="{EVAL_MIN_W}">'
        f'<thead><tr><th scope="col" class="rowhead" style="width:{ROWHEAD_W}px">sample</th>{head}'
        f'<th scope="col" class="delta" style="width:{DELTA_W}px">change</th>'
        f'<th scope="col" class="verdict" style="width:{VERDICT_W}px">reading</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


# Column widths for the big grid. The evaluation columns take everything left over.
ROWHEAD_W = 240
DELTA_W = 62
VERDICT_W = 104
EVAL_MIN_W = 150


def _scorer_note(grid: ScorerGrid) -> str:
    noise = f"noise ±{grid.noise:.3f}" if grid.noise is not None else "noise unmeasured"
    direction = "lower is better" if grid.lower_is_better else ""
    return " · ".join(x for x in (noise, direction) if x)


def variation_of(prov: dict[str, Any]) -> tuple[str, str, str]:
    """Return (variation_id, model, git_change_id) for one evaluation."""
    declared = prov.get("variation") or {}
    model = str(declared.get("model") or "")
    if not model:
        # Real runs record the model the UI sent, per sample, in the trace. Fall back to
        # whatever provenance captured.
        model = str(prov.get("model") or "unknown")
    change = str(prov.get("change_id") or "")
    agents = ",".join(declared.get("agents") or [])
    return f"{change}|{model}|{agents}", model, change


TRANSCRIPT_CHARS = 6000
MAX_STEPS = 60
EXPLANATION_CHARS = 700


def _trace_details(runs_dir: Path, run_id: str, epoch: int) -> dict[str, dict[str, Any]]:
    """What each sample actually did, read from the traces the solver wrote.

    Embedded, not linked. A dashboard that sends you elsewhere to see what the agent said
    makes its own numbers hard to trust, because checking one costs a context switch — and
    the check you skip is the one that would have caught the bad score.
    """
    directory = runs_dir / run_id / "traces"
    if not directory.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(f"*.epoch{epoch}.json")):
        try:
            trace = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sample = str(trace.get("sample_id") or path.name.split(".epoch")[0])

        chat = []
        for turn in trace.get("turns", []):
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            chat.append(
                {
                    "role": turn.get("role") or "assistant",
                    "origin": turn.get("origin") or "",
                    "latency": turn.get("latency_ms"),
                    "text": text[:TRANSCRIPT_CHARS],
                }
            )
        steps = [
            {
                "name": c.get("name") or "",
                "subagent": c.get("subagent") or "",
                "status": c.get("status") or "",
                "source": c.get("source") or "",
                "detail": (c.get("detail") or "")[:240],
            }
            for c in (trace.get("tool_calls") or [])[:MAX_STEPS]
        ]
        out[sample] = {
            "prompt": (trace.get("prompt") or "")[:1200],
            "chat": chat,
            "steps": steps,
            "subagents": [a.get("name") for a in (trace.get("subagents") or []) if a.get("name")],
            "interrupts": [
                {
                    "tool": i.get("tool") or "",
                    "title": (i.get("title") or "")[:200],
                    "fields": i.get("fields") or [],
                    "decision": i.get("decision") or "",
                }
                for i in (trace.get("interrupts") or [])
            ],
            "totalMs": trace.get("total_ms"),
            "infra": [e.get("kind") for e in (trace.get("infra_errors") or [])],
            "notes": trace.get("notes") or [],
        }
    return out


def _readings(
    conn: sqlite3.Connection, run_id: str, suite: str, epoch: int, scorers: list[str]
) -> dict[str, dict[str, Any]]:
    """Why each scorer said what it said.

    The judge's per-criterion evidence, the assertions that failed, the tool checks — all
    of it already lives in Score.metadata, and it is the difference between reading "0.33"
    and having a reason to act on it.
    """
    out: dict[str, dict[str, Any]] = {}
    placeholders = ",".join("?" for _ in scorers)
    query = (
        "SELECT sample_id, scorer, value, answer, explanation, excluded, metadata "
        "FROM sample_scores WHERE run_id=? AND suite=? AND epoch=? "
        f"AND scorer IN ({placeholders})"
    )
    for row in conn.execute(query, (run_id, suite, epoch, *scorers)):
        try:
            meta = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        entry: dict[str, Any] = {
            "value": None if row["value"] is None else round(float(row["value"]), 4),
            "answer": (row["answer"] or "")[:120],
            "explanation": (row["explanation"] or "")[:EXPLANATION_CHARS],
            "excluded": bool(row["excluded"]),
        }
        for key in ("criteria", "missing", "forbidden_present", "checks", "observed_tools"):
            if meta.get(key):
                entry[key] = meta[key]
        out.setdefault(row["sample_id"], {})[row["scorer"]] = entry
    return out


def evaluation_payload(
    conn: sqlite3.Connection,
    runs_dir: Path,
    suite: str,
    runs: list[RunColumn],
) -> dict[str, Any]:
    """Everything the page needs, as plain data. No model, no server, no fetch."""
    scorers = list(QUALITY_ORDER) + list(DIAGNOSTIC_ORDER)
    grids = {name: build_grid(conn, runs, name) for name in scorers}
    grids = {n: g for n, g in grids.items() if any(v is not None for v in g.run_means.values())}

    samples = sorted({s for g in grids.values() for s in g.samples})
    evaluations: list[dict[str, Any]] = []
    variations: dict[str, dict[str, Any]] = {}

    for run in runs:
        prov = load_provenance(runs_dir, run.run_id) or {}
        variation_id, model, change = variation_of(prov)
        # Per-run (epoch) detail, so a single repetition can be investigated on its own.
        epochs = sorted(
            {
                r["epoch"]
                for r in conn.execute(
                    "SELECT DISTINCT epoch FROM sample_scores WHERE run_id = ? AND suite = ?",
                    (run.run_id, suite),
                )
            }
        )
        run_rows = []
        for epoch in epochs:
            per_scorer: dict[str, float] = {}
            per_sample: dict[str, dict[str, float]] = {}
            for name in grids:
                values = []
                for r in conn.execute(
                    """SELECT sample_id, value FROM sample_scores
                       WHERE run_id=? AND suite=? AND scorer=? AND epoch=?
                       AND excluded=0 AND value IS NOT NULL""",
                    (run.run_id, suite, name, epoch),
                ):
                    values.append(float(r["value"]))
                    per_sample.setdefault(r["sample_id"], {})[name] = round(float(r["value"]), 4)
                if values:
                    per_scorer[name] = fmean(values)
            run_rows.append(
                {
                    "index": epoch,
                    "score": per_scorer,
                    "samples": per_sample,
                    # Everything a run recorded is embedded, so reading one never means
                    # leaving the page. The viewer link stays as a way out to the full
                    # Inspect transcript, not as the only way in.
                    "detail": _trace_details(runs_dir, run.run_id, epoch),
                    "readings": _readings(conn, run.run_id, suite, epoch, list(grids)),
                }
            )

        cells: dict[str, dict[str, list[float]]] = {}
        for name, grid in grids.items():
            for sample in grid.samples:
                cell = grid.cells.get((sample, run.run_id))
                if cell and cell.epochs:
                    cells.setdefault(sample, {})[name] = [round(v, 4) for v in cell.epochs]

        variations.setdefault(
            variation_id,
            {
                "id": variation_id,
                "model": model,
                "changeId": change,
                "label": run.label or run.run_id,
                "representativeRun": run.run_id,
                "evaluations": [],
            },
        )["evaluations"].append(run.run_id)

        evaluations.append(
            {
                "id": run.run_id,
                "label": run.label or run.run_id,
                "when": run.when,
                "variationId": variation_id,
                "model": model,
                "isBaseline": run.is_baseline,
                "comparable": run.comparable,
                "partial": run.partial,
                "graded": run.graded,
                "of": run.of,
                "epochs": run.epochs,
                "score": {n: g.run_means[run.run_id] for n, g in grids.items()},
                "stderr": {n: g.run_stderr[run.run_id] for n, g in grids.items()},
                "coverage": {n: g.coverage[run.run_id] for n, g in grids.items()},
                "runs": run_rows,
                "cells": cells,
            }
        )

    # Diffs between distinct variations, computed once per unordered pair.
    diffs: dict[str, dict[str, Any]] = {}
    ids = list(variations)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            key = "||".join(sorted((a, b)))
            first, second = sorted((a, b))
            diff = version_diff(
                runs_dir,
                variations[first]["representativeRun"],
                variations[second]["representativeRun"],
            )
            model_change = (
                f"model: {variations[first]['model']} → {variations[second]['model']}"
                if variations[first]["model"] != variations[second]["model"]
                else ""
            )
            diffs[key] = {
                "from": first,
                "to": second,
                "modelChange": model_change,
                "summary": diff.summary,
                "html": diff_panel_body(diff, model_change),
            }

    baseline_id = recorded_baseline(runs)
    return {
        "suite": suite,
        "baselineId": baseline_id,
        "samples": samples,
        "quality": [n for n in QUALITY_ORDER if n in grids],
        "diagnostic": [n for n in DIAGNOSTIC_ORDER if n in grids],
        "lowerIsBetter": sorted(LOWER_IS_BETTER),
        "continuous": sorted(CONTINUOUS_SCORERS),
        "passThreshold": PASS_THRESHOLD,
        "noise": {n: g.noise for n, g in grids.items()},
        "noiseSource": {n: g.noise_source for n, g in grids.items()},
        "evaluations": evaluations,
        "variations": variations,
        "diffs": diffs,
    }


# Scorers whose value is a measurement rather than a 0-1 score. The pass threshold is
# meaningless for these, so no pass/fail "flip" can be computed from them — 17.8s is not a
# "fail" and an error rate of 0.17 dropping to 0.00 is an improvement, not a regression.
CONTINUOUS_SCORERS = {"latency_ms", "tool_call_count", "failed_tool_calls", "agent_error_rate"}


def recorded_baseline(runs: list[RunColumn]) -> str | None:
    """The evaluation the user nominated as the baseline, via `baseline use`.

    Read from baseline.json so the page and `compare` cannot disagree about what
    "baseline" means — the single most confusing thing a two-source-of-truth design does.
    """
    # Via the same helper the CLI uses. Deriving the path from runs_dir was a latent bug:
    # it silently stopped finding the nomination as soon as EVAL_RUNS_DIR pointed
    # anywhere other than the repo.
    declared = baseline_evaluation(REPO_ROOT)
    if declared and any(r.run_id == declared for r in runs):
        return str(declared)
    marked = [r for r in runs if r.is_baseline and r.comparable]
    if marked:
        return marked[-1].run_id
    comparable = [r for r in runs if r.comparable]
    return comparable[0].run_id if comparable else None


def diff_panel_body(diff: VersionDiff, model_change: str = "") -> str:
    """The contents of a "what changed" panel: model change, caveats, commits, patch."""
    caveats = [c for r in diff.repos for c in r.caveats]
    body = []
    if model_change:
        # Rendered as a patch rather than prose: a model swap is a change to the thing
        # under test exactly like a code edit, and reading it in the same shape as the
        # rest of the panel makes that equivalence obvious.
        before, _, after = model_change.partition(" → ")
        before = before.replace("model: ", "")
        body.append(
            '<div class="repo"><h4>agent configuration</h4>'
            '<pre class="patch">'
            '<span class="meta">diff --git a/agent.model b/agent.model</span>'
            '<span class="meta">--- a/agent.model</span>'
            '<span class="meta">+++ b/agent.model</span>'
            '<span class="hunk">@@ model @@</span>'
            f'<span class="del">-{esc(before)}</span>'
            f'<span class="add">+{esc(after)}</span>'
            "</pre></div>"
        )
    if diff.config_changes:
        body.append(
            '<p class="note warn"><strong>Configuration changed</strong> — these evaluations are '
            "not measuring the same thing: " + esc("; ".join(diff.config_changes)) + "</p>"
        )
    for caveat in caveats:
        body.append(f'<p class="note warn">{esc(caveat)}</p>')

    for repo in diff.repos:
        if repo.identical:
            continue
        commits = "".join(f"<li><code>{esc(c)}</code></li>" for c in repo.commits)
        stat = f'<pre class="stat">{esc(repo.diff_stat)}</pre>' if repo.diff_stat else ""
        patch = ""
        if repo.patch:
            lines = []
            for line in repo.patch.splitlines():
                cls = ""
                if line.startswith("+") and not line.startswith("+++"):
                    cls = "add"
                elif line.startswith("-") and not line.startswith("---"):
                    cls = "del"
                elif line.startswith("@@"):
                    cls = "hunk"
                elif line.startswith(("diff ", "index ", "--- ", "+++ ")):
                    cls = "meta"
                lines.append(f'<span class="{cls}">{esc(line)}</span>')
            more = (
                '<p class="note">Patch truncated. Full patch: '
                f"<code>{esc(repo.patch_file or 'runs/<evaluation>/provenance/')}</code></p>"
                if repo.patch_truncated
                else ""
            )
            # Joined with nothing, not a newline: each line is already `display:block`, and
            # inside a <pre> the newline character would add a second break, doubling the
            # height of every diff.
            patch = f'<div class="scroll"><pre class="patch">{"".join(lines)}</pre></div>{more}'
        body.append(
            f'<div class="repo"><h4>{esc(repo.repo)} '
            f'<code class="range">{esc(repo.from_sha[:10])}…{esc(repo.to_sha[:10])}</code></h4>'
            f'{f"<ul class=commits>{commits}</ul>" if commits else ""}{stat}{patch}</div>'
        )

    if not body:
        body.append(
            '<p class="note">No code difference and no model difference between these '
            "evaluations. Any delta between them is run-to-run noise — which makes this pair a "
            "useful measurement of how noisy the suite is.</p>"
        )
    return "".join(body)





def _variation_label(variation_id: str) -> str:
    """`change|model|agents` is the identity; this is how a human should read it."""
    parts = [p for p in variation_id.split("|") if p]
    return " · ".join(parts) if parts else variation_id


ALL_CANDIDATES = "__all__"


def _options(
    runs: list[RunColumn],
    selected: str | None,
    baseline_id: str | None,
    include_all: bool = False,
) -> str:
    """Only the *nominated* baseline is marked.

    `RunColumn.is_baseline` means something else — "this evaluation's code matched the
    pinned git baseline" — and marking every such evaluation "(baseline)" made two of them
    claim the title at once.
    """
    out = []
    if include_all:
        out.append(
            f'<option value="{ALL_CANDIDATES}"'
            f'{" selected" if selected == ALL_CANDIDATES else ""}>See all candidates</option>'
        )
    for run in runs:
        mark = " — baseline" if run.run_id == baseline_id else ""
        sel = " selected" if run.run_id == selected else ""
        out.append(
            f'<option value="{esc(run.run_id)}"{sel}>{esc(run.short)}{mark} · {esc(run.when)}</option>'
        )
    return "".join(out)


def compare_panel(suite: str) -> str:
    """The comparison result. Its controls live in the app bar, always in reach."""
    return (
        f'<section class="panel" id="compare-{esc(suite)}"><header><h2>Compare</h2>'
        '<p class="why">Paired per sample. A delta inside the noise floor is reported as noise, '
        "not as a win.</p></header>"
        '<div data-role="verdict" class="verdict-line pill neutral"></div>'
        '<div data-role="compare-meta" class="compare-meta"></div>'
        '<div class="scroll" data-role="compare-table"><table class="deltas">'
        "<thead><tr><th>scorer</th><th>baseline</th><th>candidate</th><th>change</th>"
        "<th>reading</th><th>samples</th><th>flips</th></tr></thead>"
        '<tbody data-role="compare-body"></tbody></table></div>'
        "</section>"
    )


def diff_dock(suite: str, active: bool) -> str:
    """The diff, docked to the right edge rather than sitting in the reading flow.

    It answers one question — what is different between these two variations — and you
    want it beside the numbers, not a scroll away from them.
    """
    return (
        f'<aside class="diffdock" data-suite-dock="{esc(suite)}"'
        f'{"" if active else " hidden data-inactive"} '
        'aria-label="Difference between the selected variations">'
        '<div class="dock-head">'
        '<h2 data-role="dock-heading">Diff</h2>'
        '<span class="dock-sub" data-role="diff-title"></span>'
        '<button type="button" class="dock-toggle" data-role="dock-toggle" aria-expanded="true" '
        'title="Collapse the panel">\u00d7</button>'
        "</div>"
        '<div class="dock-body" data-role="diff-body"></div>'
        "</aside>"
    )


def suite_controls(payload: dict[str, Any], runs: list[RunColumn], suite: str, active: bool) -> str:
    """Baseline and candidate pickers for one suite, rendered into the app bar.

    They sit in the header rather than in the comparison panel because everything below —
    the grid's deltas, the readings, the diff — is expressed against this pair, so the
    controls have to stay in reach while you scroll a long hill.
    """
    baseline = payload["baselineId"]
    # Opens on "See all": before you have a specific candidate in mind, the useful view is
    # every variation against the baseline at once.
    return (
        f'<div class="controls" data-suite-controls="{esc(suite)}"{"" if active else " hidden"}>'
        '<label class="field"><span class="field-label">Baseline</span>'
        f'<select data-role="pick-a">{_options(runs, baseline, baseline)}</select></label>'
        '<span class="vs" aria-hidden="true">vs</span>'
        '<label class="field"><span class="field-label">Candidate</span>'
        f'<select data-role="pick-b">'
        f"{_options(runs, ALL_CANDIDATES, baseline, include_all=True)}</select></label>"
        # Which of the two things the side panel is for. Diff needs a specific candidate;
        # traces do not, so with "See all candidates" chosen Diff is unavailable and says
        # so rather than silently doing nothing.
        '<div class="modes dockmodes" role="group" aria-label="Side panel">'
        '<button type="button" class="mode" data-dock-mode="diff" aria-pressed="true">Diff</button>'
        '<button type="button" class="mode" data-dock-mode="detail" aria-pressed="false">Traces</button>'
        "</div>"
        "</div>"
    )


def evaluations_panel(payload: dict[str, Any]) -> str:
    """Every run of every evaluation, in one table.

    This replaced a list of nine accordions. The accordions hid the one thing this view
    uniquely knows — how much a sample moves between runs of the *same* evaluation — behind
    a click each, and re-showed per-sample scores the grid above already gives you. Laid
    out flat, three cells of one colour say "stable" and three of different colours say
    "flaky" without reading a number.
    """
    evaluations = payload["evaluations"]
    if not evaluations:
        return ""

    groups = []
    runheads = []
    for ev in evaluations:
        span = max(1, len(ev["runs"]))
        mark = " baseline" if ev["id"] == payload["baselineId"] else ""
        groups.append(
            f'<th scope="colgroup" colspan="{span}" class="evgroup" data-eval="{esc(ev["id"])}">'
            f'<span class="run-label">{esc(ev["label"])}</span>'
            f'<span class="run-meta">{esc(ev["model"])} · {esc(_fmt_signal(ev["score"].get(PRIMARY), PRIMARY))}'
            f"{mark}</span></th>"
        )
        runheads.extend(
            f'<th scope="col" class="num" data-eval="{esc(ev["id"])}">{r["index"]}</th>'
            for r in ev["runs"]
        )

    rows = []
    for sample in payload["samples"]:
        cells = []
        present = False
        for ev in evaluations:
            for r in ev["runs"]:
                value = (r.get("samples", {}).get(sample) or {}).get(PRIMARY)
                if value is None:
                    cells.append(f'<td class="runcell absent" data-eval="{esc(ev["id"])}">—</td>')
                    continue
                present = True
                cells.append(
                    f'<td class="runcell s{ramp_step(value)}" data-eval="{esc(ev["id"])}">'
                    f'<button type="button" class="cellbtn" data-open-detail'
                    f' data-eval="{esc(ev["id"])}" data-sample="{esc(sample)}"'
                    f' data-run="{r["index"]}"'
                    f' title="{esc(ev["label"])} · run {r["index"]}">{value:.2f}</button></td>'
                )
        if present:
            rows.append(
                f'<tr><th scope="row" class="rowhead"><code>{esc(sample)}</code></th>'
                f'{"".join(cells)}</tr>'
            )

    return (
        '<section class="panel"><header><h2>Runs</h2>'
        f'<p class="why">{esc(PRIMARY)} per sample, one column per run. Three cells of one '
        "colour means the sample is stable; three different colours mean it is flaky, and its "
        "mean above is hiding that. Click any cell for that run's result, chat and trace."
        "</p></header>"
        '<div class="scroll bigscroll"><table class="grid runstable">'
        f'<thead><tr><th scope="col" class="rowhead" rowspan="2">sample</th>{"".join(groups)}</tr>'
        f'<tr>{"".join(runheads)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        "</section>"
    )


def _fmt_signal(value: float | None, scorer: str) -> str:
    if value is None:
        return "–"
    if scorer == "latency_ms":
        return f"{value / 1000:.1f}s"
    if scorer in CONTINUOUS_SCORERS:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _summary_line(payload: dict[str, Any], grid: ScorerGrid, runs: list[RunColumn]) -> str:
    """The same facts the stat tiles carried, as a sentence.

    Four boxed numbers took a third of the fold to say what one line says, and three of
    them were already visible elsewhere: the baseline is named in the app bar, the noise
    floor is on every scorer group in the grid, and the latest score is in the evaluation
    list.
    """
    comparable = [r for r in runs if r.comparable and grid.run_means[r.run_id] is not None]
    parts = []
    if comparable:
        latest = comparable[-1]
        parts.append(
            f"<b>{esc(grid.scorer)}</b> {esc(_fmt_signal(grid.run_means[latest.run_id], grid.scorer))} "
            f"at {esc(latest.short)}"
        )
    baseline = next((r for r in runs if r.run_id == payload["baselineId"]), None)
    if baseline is not None:
        parts.append(
            f"baseline {esc(_fmt_signal(grid.run_means.get(baseline.run_id), grid.scorer))} "
            f"({esc(baseline.short)})"
        )
    parts.append(
        f"noise ±{grid.noise:.3f}" if grid.noise is not None else "noise floor not yet measured"
    )
    total_runs = sum(len(e["runs"]) for e in payload["evaluations"])
    parts.append(
        f"{len(payload['variations'])} variations · {len(payload['evaluations'])} evaluations · "
        f"{total_runs} runs"
    )
    return f'<p class="summary-line" data-role="summary">{" · ".join(parts)}</p>'


def _legend() -> str:
    steps = "".join(f'<span class="swatch s{i}"></span>' for i in range(RAMP_STEPS))
    diverging = "".join(
        f'<span class="swatch {c}"></span>' for c in ("dn3", "dn2", "dn1", "d0", "dp1", "dp2", "dp3")
    )
    gold = "".join(f'<span class="swatch g{i}"></span>' for i in range(RAMP_STEPS))
    shared = (
        '<span><span class="swatch flaky"></span>passed and failed within the same evaluation</span>'
        '<span><span class="swatch absent"></span>not graded</span>'
    )
    return (
        '<div class="legend" data-legend="delta">'
        f'<span class="scale">worse {diverging} better, vs the baseline column</span>'
        f'<span class="scale">baseline (absolute) {gold}</span>{shared}'
        "</div>"
        '<div class="legend" data-legend="absolute" hidden>'
        f'<span class="scale">0.00 fails {steps} 1.00 passes</span>'
        f'<span><span class="swatch fail"></span>scored zero</span>{shared}'
        "</div>"
    )


def _diagnostics(payload: dict[str, Any], runs: list[RunColumn]) -> str:
    by_id = {e["id"]: e for e in payload["evaluations"]}
    rows = []
    for scorer in payload["diagnostic"]:
        values = [by_id[r.run_id]["score"].get(scorer) for r in runs]
        if all(v is None for v in values):
            continue
        cells = "".join(
            f'<td class="cell foot" data-eval="{esc(r.run_id)}">{esc(_fmt_signal(v, scorer))}</td>'
            for r, v in zip(runs, values, strict=True)
        )
        rows.append(f'<tr><th scope="row">{esc(scorer)}</th>{cells}<td>{_sparkline(values)}</td></tr>')
    for scorer in payload["quality"][:1]:
        values = [by_id[r.run_id]["coverage"].get(scorer) for r in runs]
        cells = "".join(
            f'<td class="cell foot{" warn" if (v or 1) < 0.999 else ""}" data-eval="{esc(r.run_id)}">'
            f"{esc(_fmt_signal(v, 'coverage'))}</td>"
            for r, v in zip(runs, values, strict=True)
        )
        rows.append(f'<tr><th scope="row">coverage</th>{cells}<td>{_sparkline(values)}</td></tr>')
    # `data-eval` on the header too: the body cells carry it and get hidden by the column
    # filter, and a header that does not hide leaves nine labels standing over two values.
    head = "".join(f'<th scope="col" data-eval="{esc(r.run_id)}">{esc(r.short)}</th>' for r in runs)
    return (
        '<div class="scroll"><table class="grid"><thead><tr><th scope="col">signal</th>'
        f'{head}<th scope="col">trend</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )






def suite_panel(
    conn: sqlite3.Connection,
    runs_dir: Path,
    suite: str,
    runs: list[RunColumn],
    active: bool,
) -> tuple[str, dict[str, Any], str, str]:
    """One tab's worth of dashboard, plus the data the page needs for it."""
    payload = evaluation_payload(conn, runs_dir, suite, runs)
    grids = {s: build_grid(conn, runs, s) for s in list(QUALITY_ORDER) + list(DIAGNOSTIC_ORDER)}
    grids = {s: g for s, g in grids.items() if any(v is not None for v in g.run_means.values())}
    primary = grids.get(PRIMARY) or (next(iter(grids.values())) if grids else None)

    body: list[str] = []
    if primary is not None:
        body += [
            '<section class="panel"><header><h2>Hill climb</h2></header>',
            _summary_line(payload, primary, runs),
            '<div class="exec" data-role="exec"></div>',
            trend_chart(primary, runs, baseline_value=primary.run_means.get(payload["baselineId"] or "")),
            "</section>",
        ]
    body.append(compare_panel(suite))
    body += [
        '<section class="panel"><header><h2>Every scorer, every sample</h2>'
        + "</header>",
        '<div class="modes" role="group" aria-label="Cell display">'
        '<button type="button" class="mode" data-mode="delta" aria-pressed="true">change vs baseline</button>'
        '<button type="button" class="mode" data-mode="absolute" aria-pressed="false">scores</button>'
        "</div>",
        big_grid(grids, runs),
        _legend(),
        "</section>",
        evaluations_panel(payload),
        '<section class="panel"><header><h2>Cost and health</h2>'
        '<p class="why">A gain bought with 40 seconds a turn is not a gain; a score that rose '
        "because samples stopped being gradeable is not a score.</p></header>",
        _diagnostics(payload, runs),
        "</section>",
    ]
    panel = (
        f'<div class="suite-panel" data-suite-panel="{esc(suite)}"'
        f'{"" if active else " hidden"}>{"".join(body)}</div>'
    )
    return panel, payload, suite_controls(payload, runs, suite, active), diff_dock(suite, active)


def _target_fact(prov: dict, key: str) -> str:
    """One target-config value, from wherever the record keeps it.

    Runs recorded before the framework/harness split hold `base_url` and `organization`
    at the top level; newer ones nest them under `target_config`, where a target puts
    whatever makes its evaluations comparable. Both are read, so old runs keep
    rendering.
    """
    nested = prov.get("target_config") or {}
    return str(nested.get(key) or prov.get(key) or "")


def build_dashboard(
    conn: sqlite3.Connection,
    runs_dir: Path,
    suites: dict[str, list[RunColumn]],
    local_fonts: bool = False,
    demo: bool = False,
    subject: str = "agent",
    program: str = "evalkit",
) -> str:
    """The whole dashboard: one tab per suite, in one standalone local file.

    Suites are tabs rather than separate files because they are the top-level thing you
    switch between, and because their results must never be pooled — see
    compare.compare_refs. The page carries evaluation results, internal identifiers and
    source patches, so it is meant to be opened with ``file://`` and to stay local.

    ``subject`` is what is being evaluated, shown beside the tool's own name. The page is
    headed by the harness rather than by the agent on purpose: it says what produced these
    numbers, and a page that led with the product's name would read like the product's own
    dashboard — most misleadingly of all when ``demo`` is set and none of it is real.
    """
    panels, payloads, tabs, controls, docks = [], {}, [], [], []
    for index, (suite, runs) in enumerate(suites.items()):
        panel, payload, control, dock = suite_panel(conn, runs_dir, suite, runs, active=index == 0)
        panels.append(panel)
        payloads[suite] = payload
        controls.append(control)
        docks.append(dock)
        tabs.append(
            f'<button type="button" role="tab" class="tab" data-suite-tab="{esc(suite)}" '
            f'aria-selected="{"true" if index == 0 else "false"}">{esc(suite)}'
            f'<span class="tab-count">{len(runs)}</span></button>'
        )

    first_suite = next(iter(suites), "")
    first_runs = suites.get(first_suite) or []
    prov = load_provenance(runs_dir, first_runs[-1].run_id) if first_runs else None
    facts = []
    if prov:
        facts = [
            f"<span><b>judge</b> {esc(prov.get('judge_model'))}</span>",
            f"<span><b>organization</b> <code>{esc(_target_fact(prov, 'organization')[:20])}</code></span>",
            f"<span><b>seed</b> {esc(prov.get('seed'))}</span>",
            f"<span><b>runs per evaluation</b> {esc(prov.get('epochs'))}</span>",
            f"<span><b>app</b> {esc(_target_fact(prov, 'base_url'))}</span>",
        ]

    return fill(
        TEMPLATE,
        # The tab's name carries the subject too: a browser showing six local pages is the
        # one place where the tool's name alone identifies nothing.
        page_title=esc(f"{BRAND} — {subject}{' (demo)' if demo else ''}"),
        subject=esc(subject),
        font_link=(
            ""
            if local_fonts
            else (
                '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
                'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
            )
        ),
        css=CSS,
        facts="".join(facts),
        demo_flag='<span class="pill warning demo-flag">Demo</span>' if demo else "",
        # The app bar stays put: the tabs and the baseline/candidate pair define what every
        # number below means, so they must not scroll away from the numbers they govern.
        tabs="".join(tabs),
        controls="".join(controls),
        panels="".join(panels),
        generated_at=esc(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        program=esc(program),
        demo_note=(
            " <b>Every number here is synthetic demo data</b> — it describes nothing about the agent."
            if demo
            else ""
        ),
        docks="".join(docks),
        payload=json.dumps({"suites": payloads}),
        js=PAGE_JS,
    )


def run_columns(
    conn: sqlite3.Connection,
    suite: str,
    runs_dir: Path,
    limit: int = 12,
    include_mock: bool = False,
) -> list[RunColumn]:
    """Evaluations for a suite, oldest first — the reading order of a hill.

    Mock evaluations are dropped by default: they never touched the app, so a column of
    synthetic scores beside real ones invites exactly the comparison that means nothing.
    """
    rows = list(
        conn.execute(
            "SELECT * FROM runs WHERE suite = ? ORDER BY created_at DESC LIMIT ?", (suite, limit)
        )
    )
    columns: list[RunColumn] = []
    for r in reversed(rows):
        prov = load_provenance(runs_dir, r["run_id"]) or {}
        env = prov.get("env") or {}
        columns.append(
            RunColumn(
                run_id=r["run_id"],
                suite=r["suite"],
                label=r["label"] or "",
                created_at=r["created_at"] or "",
                change_id=r["change_id"] or "",
                is_baseline=bool(r["is_baseline"]),
                epochs=int(r["epochs"] or 1),
                suite_sha=r["suite_sha"] or "",
                judge_model=r["judge_model"] or "",
                harness=str(env.get("harness_fingerprint") or env.get("harness_sha") or "?")[:12],
            )
        )
    if not include_mock:
        columns = [c for c in columns if not c.mock]

    per_run = {c.run_id: set(sample_values(conn, c.run_id, c.suite, PRIMARY)) for c in columns}
    universe: set[str] = set()
    for ids in per_run.values():
        universe |= ids
    for c in columns:
        ids = per_run[c.run_id] or set()
        c.graded, c.of = len(ids), len(universe)
        c.partial = bool(universe) and ids != universe
    # A column with nothing graded says nothing, so it is dropped — except when the caller
    # asked for mock runs. A mock run's judge is `none/offline`, so its PRIMARY scores are
    # all recorded `excluded` for want of credentials and `graded` is 0 by construction:
    # dropping those here made `--include-mock` silently do nothing for exactly the runs it
    # names. Keep them, flagged as mock, and let the page show them for what they are.
    return [c for c in columns if c.graded > 0 or (include_mock and c.mock)]


def suites_with_runs(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT suite FROM runs ORDER BY suite")]
