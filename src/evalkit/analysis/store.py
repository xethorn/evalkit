"""Persist run history in SQLite so improvements can be measured, not remembered.

Inspect's ``.eval`` logs are the source of truth and stay on disk untouched. This module
flattens them into three tables that answer the questions a hill climb actually asks:

* ``runs`` — what was under test (change_id, suite hash, judge, seed, label)
* ``sample_scores`` — one row per (run, sample, epoch, scorer), the join key for paired
  comparisons between a baseline and a candidate
* ``run_metrics`` — the headline numbers, for trend lines

Ingestion is idempotent: re-ingesting a log replaces its rows, so the store can be
rebuilt from ``logs/`` at any time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT NOT NULL,
    suite         TEXT NOT NULL,
    log_path      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    label         TEXT DEFAULT '',
    suite_version INTEGER DEFAULT 0,
    suite_sha     TEXT DEFAULT '',
    change_id     TEXT DEFAULT '',
    is_baseline   INTEGER DEFAULT 0,
    judge_model   TEXT DEFAULT '',
    base_seed     INTEGER DEFAULT 0,
    epochs        INTEGER DEFAULT 1,
    status        TEXT DEFAULT '',
    samples       INTEGER DEFAULT 0,
    provenance    TEXT DEFAULT '{}',
    PRIMARY KEY (run_id, suite)
);
CREATE TABLE IF NOT EXISTS sample_scores (
    run_id     TEXT NOT NULL,
    suite      TEXT NOT NULL,
    sample_id  TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    scorer     TEXT NOT NULL,
    value      REAL,
    answer     TEXT DEFAULT '',
    explanation TEXT DEFAULT '',
    excluded   INTEGER DEFAULT 0,
    metadata   TEXT DEFAULT '{}',
    PRIMARY KEY (run_id, suite, sample_id, epoch, scorer)
);
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT NOT NULL,
    suite  TEXT NOT NULL,
    scorer TEXT NOT NULL,
    metric TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, suite, scorer, metric)
);
CREATE INDEX IF NOT EXISTS idx_scores_sample ON sample_scores (sample_id, scorer);
CREATE INDEX IF NOT EXISTS idx_runs_change ON runs (change_id);
CREATE INDEX IF NOT EXISTS idx_runs_label ON runs (label);
"""

# One eval invocation can run several suites, and each produces its own log. Keying any
# of these tables on run_id alone made suite #2 overwrite suite #1 — sample rows were
# deleted before re-insert, and run_metrics collided on (run_id, scorer, metric). The
# result was a run that silently reported only its last suite. Hence `suite` in every
# primary key, and comparisons that pair suite-to-suite.
EXPECTED_COLUMNS = {"runs": "suite", "sample_scores": "suite", "run_metrics": "suite"}


def _needs_rebuild(conn: sqlite3.Connection) -> bool:
    """True when the database predates suite-scoped keys."""
    for table, column in EXPECTED_COLUMNS.items():
        rows = list(conn.execute(f"PRAGMA table_info({table})"))
        if not rows:
            continue
        if column not in {r[1] for r in rows}:
            return True
    return False


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (and if necessary rebuild) the results database.

    The eval logs in ``logs/`` are the source of truth, so an outdated schema is dropped
    and rebuilt rather than migrated — ``ingest`` restores every run from disk.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if _needs_rebuild(conn):
        conn.executescript(
            "DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS sample_scores; DROP TABLE IF EXISTS run_metrics;"
        )
        conn.commit()
    conn.executescript(SCHEMA)
    return conn


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass
class Ingested:
    run_id: str
    suite: str
    samples: int
    scores: int


def ingest_log(conn: sqlite3.Connection, log: EvalLog, log_path: str) -> Ingested:
    meta = log.eval.metadata or {}
    run_id = str(meta.get("run_id") or log.eval.run_id)
    suite = str(meta.get("suite") or log.eval.task)
    provenance = _load_provenance(meta)

    conn.execute("DELETE FROM sample_scores WHERE run_id = ? AND suite = ?", (run_id, suite))
    conn.execute("DELETE FROM run_metrics WHERE run_id = ? AND suite = ?", (run_id, suite))
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, suite, log_path, created_at, label, suite_version, suite_sha, change_id,
            is_baseline, judge_model, base_seed, epochs, status, samples, provenance)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            suite,
            log_path,
            log.eval.created or datetime.now(UTC).isoformat(),
            str(meta.get("label", "")),
            int(meta.get("suite_version", 0) or 0),
            str(meta.get("suite_sha", "")),
            str(meta.get("change_id", "")),
            1 if str(meta.get("is_baseline", "")).lower() in {"1", "true", "yes"} else 0,
            str(meta.get("judge_model", "")),
            int(meta.get("base_seed", 0) or 0),
            int(meta.get("epochs", 1) or 1),
            str(log.status),
            len(log.samples or []),
            json.dumps(provenance),
        ),
    )

    n_scores = 0
    for sample in log.samples or []:
        for scorer_name, score in (sample.scores or {}).items():
            value = _num(score.value)
            metadata = score.metadata or {}
            conn.execute(
                """INSERT OR REPLACE INTO sample_scores
                   (run_id, suite, sample_id, epoch, scorer, value, answer, explanation, excluded, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    suite,
                    str(sample.id),
                    int(sample.epoch or 1),
                    scorer_name,
                    value,
                    str(score.answer or "")[:500],
                    str(score.explanation or "")[:2000],
                    1 if metadata.get("excluded") else 0,
                    json.dumps(metadata, default=str)[:200000],
                ),
            )
            n_scores += 1

    for score in (log.results.scores if log.results else None) or []:
        for metric_name, metric in (score.metrics or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO run_metrics (run_id, suite, scorer, metric, value) VALUES (?,?,?,?,?)",
                (run_id, suite, score.name, metric_name, _num(metric.value)),
            )
    conn.commit()
    return Ingested(run_id=run_id, suite=suite, samples=len(log.samples or []), scores=n_scores)


def _load_provenance(meta: dict[str, Any]) -> dict[str, Any]:
    """Fold the run's provenance.json into the row, if the run recorded one."""
    path = meta.get("provenance_path")
    if path and Path(str(path)).exists():
        try:
            return json.loads(Path(str(path)).read_text())
        except Exception:
            pass
    return {k: v for k, v in meta.items() if k in {"change_id", "suite_sha", "base_url", "judge_model", "target"}}


def ingest_dir(conn: sqlite3.Connection, log_dir: Path) -> list[Ingested]:
    out: list[Ingested] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name, header_only=False)
        out.append(ingest_log(conn, log, info.name))
    return out


def latest_runs(conn: sqlite3.Connection, suite: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
    """Newest first, one row per (run, suite)."""
    sql = "SELECT * FROM runs"
    args: list[Any] = []
    if suite:
        sql += " WHERE suite = ?"
        args.append(suite)
    sql += " ORDER BY created_at DESC, suite ASC LIMIT ?"
    args.append(limit)
    return list(conn.execute(sql, args))


def _nominated_baseline() -> str | None:
    """Read the nominated baseline lazily, so the module has no import-time dependency on
    the repo layout and tests can point it elsewhere."""
    from .. import config as config_module
    from ..provenance import baseline_evaluation

    return baseline_evaluation(config_module.REPO_ROOT)


def run_row(conn: sqlite3.Connection, run_id: str, suite: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE run_id = ? AND suite = ?", (run_id, suite)).fetchone()


def suites_in(conn: sqlite3.Connection, run_id: str) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT suite FROM runs WHERE run_id = ? ORDER BY suite", (run_id,))]


def resolve_runs(conn: sqlite3.Connection, ref: str) -> list[sqlite3.Row]:
    """Resolve a run reference to one row per suite it covered.

    Tried in order: exact run id, exact label (most recent run bearing it), then the
    keywords ``latest`` and ``baseline``. Labels come first on purpose — a run *labelled*
    "baseline" is almost always what someone typing ``baseline`` meant, and falling
    through to "newest run whose code matched the pinned baseline" can quietly resolve to
    the candidate itself and compare a run against itself.
    """
    rows = list(conn.execute("SELECT * FROM runs WHERE run_id = ? ORDER BY suite", (ref,)))
    if rows:
        return rows

    latest_label = conn.execute(
        "SELECT run_id FROM runs WHERE label = ? ORDER BY created_at DESC LIMIT 1", (ref,)
    ).fetchone()
    if latest_label:
        return list(
            conn.execute("SELECT * FROM runs WHERE run_id = ? ORDER BY suite", (latest_label["run_id"],))
        )

    if ref == "latest":
        newest = conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    elif ref == "baseline":
        # A nomination beats inference. `baseline use` is the user naming which
        # *measurement* is the reference, which is a different question from "whose code
        # matched the pinned shas" — and it is the one a comparison should default to.
        nominated = _nominated_baseline()
        if nominated:
            rows = list(
                conn.execute("SELECT * FROM runs WHERE run_id = ? ORDER BY suite", (nominated,))
            )
            if rows:
                return rows
        newest = conn.execute(
            "SELECT run_id FROM runs WHERE is_baseline = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    else:
        return []
    if not newest:
        return []
    return list(conn.execute("SELECT * FROM runs WHERE run_id = ? ORDER BY suite", (newest["run_id"],)))


def resolve_run(conn: sqlite3.Connection, ref: str, suite: str | None = None) -> sqlite3.Row | None:
    """One row: the given suite, or the only suite if the run covered just one."""
    rows = resolve_runs(conn, ref)
    if not rows:
        return None
    if suite:
        return next((r for r in rows if r["suite"] == suite), None)
    return rows[0]


def sample_values(conn: sqlite3.Connection, run_id: str, suite: str, scorer: str) -> dict[str, list[float]]:
    """Per-sample lists of epoch values (excluded scores dropped)."""
    out: dict[str, list[float]] = {}
    for row in conn.execute(
        "SELECT sample_id, value, excluded FROM sample_scores WHERE run_id = ? AND suite = ? AND scorer = ?",
        (run_id, suite, scorer),
    ):
        if row["excluded"] or row["value"] is None:
            continue
        out.setdefault(row["sample_id"], []).append(float(row["value"]))
    return out


def scorers_for(conn: sqlite3.Connection, run_id: str, suite: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT scorer FROM sample_scores WHERE run_id = ? AND suite = ? ORDER BY scorer",
            (run_id, suite),
        )
    ]


def metric_value(
    conn: sqlite3.Connection, run_id: str, suite: str, scorer: str, metric: str
) -> float | None:
    row = conn.execute(
        "SELECT value FROM run_metrics WHERE run_id = ? AND suite = ? AND scorer = ? AND metric = ?",
        (run_id, suite, scorer, metric),
    ).fetchone()
    return None if row is None or row["value"] is None else float(row["value"])
