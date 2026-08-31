"""Suite loading: YAML on disk -> Inspect ``Sample`` objects.

The suite file is the contract for an eval. It is versioned, and its content hash goes
into every run record: if the suite changes, scores across the boundary are not
comparable and the report says so instead of quietly drawing a trend line.

Expectations are rendered with the *same* bindings as the prompt, so a rubric can refer
to ``${period}`` and get the exact random month the agent was asked about.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.dataset import MemoryDataset, Sample

from .templating import render, sample_seed


@dataclass
class ToolExpectations:
    required: list[str] = field(default_factory=list)
    required_any: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    max_calls: dict[str, int] = field(default_factory=dict)

    @property
    def specified(self) -> bool:
        return bool(self.required or self.required_any or self.forbidden or self.order or self.max_calls)


@dataclass
class AgentExpectations:
    """Which sub-agents the orchestrator should (and should not) delegate to."""

    required: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    @property
    def specified(self) -> bool:
        return bool(self.required or self.forbidden)


@dataclass
class Expectations:
    rubric: str = ""
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    # Numbers that must appear in the answer, within tolerance. Lets an eval assert a
    # computed total without breaking every time the seeded fixture data shifts.
    must_approx: list[dict[str, float]] = field(default_factory=list)
    # Figures to verify against the ledger rather than against the answer's own internal
    # consistency. Each entry names a figure, describes it for extraction, and carries
    # whatever the target needs to compute it independently.
    figures: list[dict] = field(default_factory=list)
    tools: ToolExpectations = field(default_factory=ToolExpectations)
    agents: AgentExpectations = field(default_factory=AgentExpectations)
    max_steps: int | None = None
    max_latency_ms: int | None = None
    # For deliberately underspecified prompts: the right behaviour is to ask, not guess.
    must_ask: bool | None = None
    numeric_tolerance: float = 0.01


@dataclass
class EvalCase:
    id: str
    prompt: str
    prompt_template: str
    bindings: dict[str, str]
    seed: int
    months_back: int
    expect: Expectations
    user: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    severity: str = "medium"
    notes: str = ""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _render_all(value: Any, seed: int, today: date, bindings: dict[str, str], months_back: int) -> Any:
    """Render templates inside strings, recursively, reusing the prompt's bindings."""
    if isinstance(value, str):
        return render(value, seed=seed, today=today, bindings=bindings, months_back=months_back).text
    if isinstance(value, list):
        return [_render_all(v, seed, today, bindings, months_back) for v in value]
    if isinstance(value, dict):
        return {k: _render_all(v, seed, today, bindings, months_back) for k, v in value.items()}
    return value


def _expectations(raw: dict[str, Any]) -> Expectations:
    tools = raw.get("tools") or {}
    agents = raw.get("agents") or {}
    return Expectations(
        rubric=str(raw.get("rubric", "")).strip(),
        must_contain=[str(x) for x in raw.get("must_contain", [])],
        must_not_contain=[str(x) for x in raw.get("must_not_contain", [])],
        figures=[dict(x) for x in (raw.get("figures") or []) if isinstance(x, dict)],
        must_approx=[
            {"value": float(x["value"]), "tolerance": float(x.get("tolerance", 0.01))}
            for x in raw.get("must_approx", [])
        ],
        tools=ToolExpectations(
            required=[str(x) for x in tools.get("required", [])],
            required_any=[str(x) for x in tools.get("required_any", [])],
            forbidden=[str(x) for x in tools.get("forbidden", [])],
            order=[str(x) for x in tools.get("order", [])],
            max_calls={str(k): int(v) for k, v in (tools.get("max_calls") or {}).items()},
        ),
        agents=AgentExpectations(
            required=[str(x) for x in agents.get("required", [])],
            forbidden=[str(x) for x in agents.get("forbidden", [])],
        ),
        max_steps=raw.get("max_steps"),
        max_latency_ms=raw.get("max_latency_ms"),
        must_ask=raw.get("must_ask"),
        numeric_tolerance=float(raw.get("numeric_tolerance", 0.01)),
    )


@dataclass
class Suite:
    name: str
    version: int
    path: Path
    content_sha: str
    cases: list[EvalCase]

    def filter(
        self,
        tags: list[str] | None = None,
        ids: list[str] | None = None,
        severity: str | None = None,
    ) -> Suite:
        cases = self.cases
        if tags:
            wanted = set(tags)
            cases = [c for c in cases if wanted & set(c.tags)]
        if ids:
            wanted_ids = set(ids)
            cases = [c for c in cases if c.id in wanted_ids]
        if severity:
            cases = [c for c in cases if c.severity == severity]
        return Suite(self.name, self.version, self.path, self.content_sha, cases)

    def to_dataset(self) -> MemoryDataset:
        samples = [
            Sample(
                input=case.prompt,
                id=case.id,
                target=case.expect.rubric or "see rubric",
                metadata={
                    "suite": self.name,
                    "suite_version": self.version,
                    "suite_sha": self.content_sha,
                    "prompt_template": case.prompt_template,
                    "bindings": case.bindings,
                    "seed": case.seed,
                    "months_back": case.months_back,
                    "tags": case.tags,
                    "severity": case.severity,
                    "notes": case.notes,
                    "expect": _expect_dict(case.expect),
                    "user": case.user,
                },
            )
            for case in self.cases
        ]
        return MemoryDataset(samples=samples, name=self.name)


def _expect_dict(e: Expectations) -> dict[str, Any]:
    return {
        "rubric": e.rubric,
        "must_contain": e.must_contain,
        "must_not_contain": e.must_not_contain,
        "must_approx": e.must_approx,
        "figures": e.figures,
        "tools": {
            "required": e.tools.required,
            "required_any": e.tools.required_any,
            "forbidden": e.tools.forbidden,
            "order": e.tools.order,
            "max_calls": e.tools.max_calls,
        },
        "agents": {"required": e.agents.required, "forbidden": e.agents.forbidden},
        "max_steps": e.max_steps,
        "max_latency_ms": e.max_latency_ms,
        "must_ask": e.must_ask,
        "numeric_tolerance": e.numeric_tolerance,
    }


def load_suite(path: Path, base_seed: int, today: date | None = None, months_back: int = 18) -> Suite:
    raw_text = path.read_text()
    data = yaml.safe_load(raw_text) or {}
    today = today or date.today()
    content_sha = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
    defaults = data.get("defaults") or {}
    name = str(data.get("suite") or path.stem)
    version = int(data.get("version", 1))

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for raw_case in data.get("tasks") or []:
        merged = _merge(defaults, raw_case)
        case_id = str(merged.get("id") or "")
        if not case_id:
            raise ValueError(f"{path}: every task needs a stable `id` (it is the join key across runs)")
        if case_id in seen:
            raise ValueError(f"{path}: duplicate task id {case_id!r}")
        seen.add(case_id)

        seed = sample_seed(base_seed, case_id)
        template = str(merged.get("prompt", "")).strip()
        case_months_back = int(merged.get("months_back") or months_back)
        rendered = render(template, seed=seed, today=today, months_back=case_months_back)
        expect_raw = _render_all(merged.get("expect") or {}, seed, today, rendered.bindings, case_months_back)
        user_raw = _render_all(merged.get("user") or {}, seed, today, rendered.bindings, case_months_back)

        cases.append(
            EvalCase(
                id=case_id,
                prompt=rendered.text,
                prompt_template=template,
                bindings=rendered.bindings,
                seed=seed,
                months_back=case_months_back,
                expect=_expectations(expect_raw),
                user=user_raw,
                tags=[str(t) for t in merged.get("tags", [])],
                severity=str(merged.get("severity", "medium")),
                notes=str(merged.get("notes", "")),
            )
        )
    return Suite(name=name, version=version, path=path, content_sha=content_sha, cases=cases)


def load_suites(
    paths: list[Path], base_seed: int, today: date | None = None, months_back: int = 18
) -> list[Suite]:
    return [load_suite(p, base_seed, today, months_back) for p in paths]


def discover(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))
