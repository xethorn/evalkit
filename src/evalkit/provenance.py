"""Run provenance: what exactly was under test.

A score is only useful next to the thing it scored. For every repo under test we
record the HEAD sha, the branch, the diff against the declared *baseline* sha, and the
uncommitted working-tree diff — then fingerprint all of it. Two runs with the same
``change_id`` tested the same code; a run whose ``change_id`` differs from the baseline's
is a hill-climb candidate.

Patches are written to ``runs/<run_id>/provenance/`` rather than into the eval log, so
the log stays small and the patch stays greppable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings, redact

BASELINE_FILE = "baseline.json"
# The nominated baseline *evaluation* is a run id from the local results database, so it
# means nothing on another machine. It lives in its own gitignored file: `baseline.json`
# carries the pinned commits, which are the team's shared reference and belong in git.
LOCAL_BASELINE_FILE = ".baseline-local.json"


def _git(repo: Path, *args: str, timeout: int = 30) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


@dataclass
class RepoState:
    name: str
    path: str
    exists: bool = False
    branch: str = ""
    head_sha: str = ""
    head_subject: str = ""
    baseline_sha: str = ""
    dirty: bool = False
    # Commits on HEAD that the baseline does not have (the "adjustments").
    commits_ahead: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    diff_stat: str = ""
    untracked: list[str] = field(default_factory=list)
    # sha256 over (baseline..HEAD diff + working-tree diff): identifies the adjustment.
    diff_sha: str = ""
    committed_patch_file: str = ""
    worktree_patch_file: str = ""

    @property
    def is_baseline(self) -> bool:
        return not self.commits_ahead and not self.dirty


@dataclass
class Provenance:
    run_id: str
    created_at: str
    label: str
    seed: int
    epochs: int
    suite: str
    judge_model: str
    # Which product was evaluated, and the configuration of it that decides whether two
    # evaluations are comparable at all (tenant, app URL, feature toggles). The framework
    # does not interpret these — the target supplies them and the dashboard diffs them.
    target: str = ""
    target_config: dict[str, Any] = field(default_factory=dict)
    repos: list[RepoState] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # Mirrored out of `target_config` for records written before the framework/harness
    # split, so old runs and new ones read the same way on the dashboard.
    base_url: str = ""
    organization: str = ""
    feature_toggles: dict[str, object] = field(default_factory=dict)
    # One id for "the state of the world under test": same id == comparable runs.
    change_id: str = ""
    is_baseline: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def load_baseline(root: Path) -> dict[str, str]:
    path = root / BASELINE_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: str(v) for k, v in data.get("repos", {}).items()}


def save_baseline(root: Path, repos: dict[str, str], note: str = "") -> Path:
    path = root / BASELINE_FILE
    payload: dict[str, Any] = {
        "note": note,
        "updated_at": datetime.now(UTC).isoformat(),
        "repos": repos,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def set_baseline_evaluation(root: Path, evaluation_id: str) -> Path:
    """Nominate the evaluation that every comparison defaults to.

    One file, read by both `compare` and the dashboard, so the terminal and the page
    cannot disagree about what "baseline" means — the most confusing failure mode a
    two-sources-of-truth design has. Local, because the id names a row in *this* machine's
    results database.
    """
    path = root / LOCAL_BASELINE_FILE
    path.write_text(
        json.dumps(
            {"baseline_evaluation": evaluation_id, "updated_at": datetime.now(UTC).isoformat()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def baseline_evaluation(root: Path) -> str | None:
    """The nominated baseline evaluation, if this machine has chosen one."""
    for path in (root / LOCAL_BASELINE_FILE, root / BASELINE_FILE):
        # `baseline.json` is read second only so an existing nomination keeps working; new
        # ones are never written there.
        if not path.exists():
            continue
        try:
            nominated = json.loads(path.read_text()).get("baseline_evaluation")
        except (OSError, json.JSONDecodeError):
            continue
        if nominated:
            return str(nominated)
    return None


def capture_repo(repo: Path, baseline_sha: str, out_dir: Path) -> RepoState:
    name = repo.name
    state = RepoState(name=name, path=str(repo))
    if not (repo / ".git").exists():
        return state
    state.exists = True
    state.head_sha = _git(repo, "rev-parse", "HEAD").strip()
    state.branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    state.head_subject = _git(repo, "log", "-1", "--format=%s").strip()
    state.dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").strip())
    state.untracked = [
        line for line in _git(repo, "ls-files", "--others", "--exclude-standard").splitlines() if line
    ]

    # Does git know the baseline commit? If not, treat HEAD as its own baseline and say so.
    state.baseline_sha = baseline_sha
    known = bool(baseline_sha) and _commit_exists(repo, baseline_sha)
    committed_patch = ""
    if known and baseline_sha != state.head_sha:
        state.commits_ahead = [
            line for line in _git(repo, "log", "--format=%h %s", f"{baseline_sha}..HEAD").splitlines() if line
        ]
        committed_patch = _git(repo, "diff", f"{baseline_sha}..HEAD", timeout=120)
        state.diff_stat = _git(repo, "diff", "--stat", f"{baseline_sha}..HEAD").strip()
    elif not known and baseline_sha:
        state.baseline_sha = f"{baseline_sha} (unknown to {name}; comparison skipped)"

    worktree_patch = _git(repo, "diff", "HEAD", timeout=120) if state.dirty else ""

    changed: set[str] = set()
    for rev in ([f"{baseline_sha}..HEAD"] if state.commits_ahead else []) + (["HEAD"] if state.dirty else []):
        changed.update(f for f in _git(repo, "diff", "--name-only", rev).splitlines() if f)
    state.files_changed = sorted(changed)

    state.diff_sha = hashlib.sha256((committed_patch + "\0" + worktree_patch).encode()).hexdigest()[:16]

    out_dir.mkdir(parents=True, exist_ok=True)
    if committed_patch:
        p = out_dir / f"{name}.baseline-to-head.patch"
        p.write_text(redact(committed_patch))
        state.committed_patch_file = str(p.relative_to(out_dir.parent))
    if worktree_patch:
        p = out_dir / f"{name}.worktree.patch"
        p.write_text(redact(worktree_patch))
        state.worktree_patch_file = str(p.relative_to(out_dir.parent))
    return state


def _commit_exists(repo: Path, rev: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{rev}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


# Directories whose content defines the measurement: the framework, the target harness,
# the scorers and the suites.
HARNESS_SOURCES = ("src", "evals")


def harness_fingerprint(root: Path) -> str:
    """A content hash of the measuring apparatus itself.

    The framework is the ruler: change a scorer or a rubric and scores move without the
    agent changing at all. A git sha alone does not cover that — this repo may be
    uncommitted or dirty, in which case every run would report the same "uncommitted" and
    the boundary would be invisible on the dashboard. Hashing the source closes that hole
    regardless of git state.
    """
    digest = hashlib.sha256()
    for directory in HARNESS_SOURCES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(str(path.relative_to(root)).encode())
            try:
                digest.update(path.read_bytes())
            except OSError:
                continue
    return digest.hexdigest()[:12]


def capture(
    run_id: str,
    suite: str,
    label: str,
    cfg: Settings,
    root: Path,
) -> Provenance:
    out_dir = cfg.run.runs_dir / run_id / "provenance"
    baselines = load_baseline(root)
    repos = [capture_repo(r, baselines.get(r.name, ""), out_dir) for r in cfg.repos_under_test]
    target = cfg.target
    fingerprint = dict(target.fingerprint())

    prov = Provenance(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        label=label,
        seed=cfg.run.seed,
        epochs=cfg.run.epochs,
        suite=suite,
        judge_model=cfg.judge.model,
        target=target.name,
        target_config=fingerprint,
        base_url=str(fingerprint.get("base_url", "")),
        organization=str(fingerprint.get("organization", "")),
        feature_toggles=dict(fingerprint.get("feature_toggles") or {}),
        repos=repos,
        env={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "harness_sha": _git(root, "rev-parse", "HEAD").strip() or "uncommitted",
            "harness_dirty": str(bool(_git(root, "status", "--porcelain").strip())),
            # Content hash of src/ and evals/ — the actual identity of the ruler.
            "harness_fingerprint": harness_fingerprint(root),
            "ci": os.environ.get("CI", ""),
        },
    )
    prov.change_id = hashlib.sha256(
        "|".join(f"{r.name}@{r.head_sha}+{r.diff_sha}" for r in repos).encode()
    ).hexdigest()[:16]
    prov.is_baseline = all(r.is_baseline for r in repos if r.exists)

    path = cfg.run.runs_dir / run_id / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prov.to_json() + "\n")
    return prov


def summarize(prov: Provenance) -> str:
    lines = [f"run {prov.run_id}  change {prov.change_id}  {'BASELINE' if prov.is_baseline else 'CANDIDATE'}"]
    for r in prov.repos:
        if not r.exists:
            lines.append(f"  {r.name}: not a git repo ({r.path})")
            continue
        marker = "dirty" if r.dirty else "clean"
        lines.append(
            f"  {r.name}: {r.branch}@{r.head_sha[:10]} ({marker}), "
            f"{len(r.commits_ahead)} commit(s) ahead of baseline, {len(r.files_changed)} file(s) changed"
        )
    return "\n".join(lines)
