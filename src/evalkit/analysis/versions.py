"""What changed between two runs, in code.

A score is only meaningful next to the change that produced it, so the dashboard has to
answer "what did we actually do between these two columns?" without the reader leaving the
page.

Three sources, in order of preference:

1. ``git diff <a.head_sha>..<b.head_sha>`` computed live — exact, and includes commits
   that landed between the two runs.
2. The patches each run saved at capture time (``runs/<id>/provenance/*.patch``) — needed
   because a run of *uncommitted* work cannot be reconstructed from git later.
3. Nothing, when the commits no longer exist (rebased, pruned). Said plainly rather than
   rendered as "no changes", which is the dangerous reading.

The distinction matters: a dirty working tree means the true delta between two runs is
*not* recoverable from git alone, and a dashboard that quietly shows the commit range
would be lying about what was tested.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_PATCH_LINES = 1200


def _git(repo: Path, *args: str, timeout: int = 60) -> str | None:
    """Run git, returning None on failure so callers can distinguish empty from broken."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _commit_exists(repo: Path, sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, check=False
        ).returncode
        == 0
    )


@dataclass
class RepoDiff:
    """The change in one repo between two runs."""

    repo: str
    from_sha: str = ""
    to_sha: str = ""
    commits: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    diff_stat: str = ""
    patch: str = ""
    patch_truncated: bool = False
    patch_file: str | None = None
    # Anything that makes the rendered diff an incomplete account of what was tested.
    caveats: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return not self.commits and not self.patch and not self.caveats

    @property
    def summary(self) -> str:
        if self.identical:
            return "no change"
        parts = []
        if self.commits:
            parts.append(f"{len(self.commits)} commit(s)")
        if self.files_changed:
            parts.append(f"{len(self.files_changed)} file(s)")
        return ", ".join(parts) or "uncommitted changes"


@dataclass
class VersionDiff:
    """The change between two runs, across every repo under test."""

    from_run: str
    to_run: str
    from_label: str = ""
    to_label: str = ""
    from_change_id: str = ""
    to_change_id: str = ""
    repos: list[RepoDiff] = field(default_factory=list)
    config_changes: list[str] = field(default_factory=list)

    @property
    def same_code(self) -> bool:
        return bool(self.from_change_id) and self.from_change_id == self.to_change_id

    @property
    def summary(self) -> str:
        if self.same_code:
            return "identical code — any delta here is run-to-run noise"
        changed = [r for r in self.repos if not r.identical]
        if not changed:
            return "no code difference found"
        return "; ".join(f"{r.repo}: {r.summary}" for r in changed)


def load_provenance(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    path = runs_dir / run_id / "provenance.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _truncate(patch: str) -> tuple[str, bool]:
    lines = patch.splitlines()
    if len(lines) <= MAX_PATCH_LINES:
        return patch, False
    kept = lines[:MAX_PATCH_LINES]
    return "\n".join(kept), True


def _stored_patch(runs_dir: Path, run_id: str, repo_state: dict[str, Any], key: str) -> tuple[str, str | None]:
    rel = repo_state.get(key)
    if not rel:
        return "", None
    path = runs_dir / run_id / rel
    if not path.is_file():
        return "", None
    try:
        return path.read_text(), str(path)
    except OSError:
        return "", None


def repo_diff(
    runs_dir: Path,
    from_run: str,
    to_run: str,
    from_state: dict[str, Any],
    to_state: dict[str, Any],
) -> RepoDiff:
    name = str(to_state.get("name") or from_state.get("name") or "?")
    repo_path = Path(str(to_state.get("path") or from_state.get("path") or ""))
    from_sha = str(from_state.get("head_sha") or "")
    to_sha = str(to_state.get("head_sha") or "")
    out = RepoDiff(repo=name, from_sha=from_sha, to_sha=to_sha)

    # A dirty tree on either side means git cannot tell the whole story. Name the files:
    # "uncommitted changes" alone cannot be acted on, because it does not distinguish the
    # variation under test from unrelated work in progress sitting in the same checkout.
    # A run dirty only in scripts/ or a scratch note tested the committed code after all;
    # a run dirty in the agent's own source did not, and the reader needs to see which.
    for state, run, side in ((from_state, from_run, "earlier"), (to_state, to_run, "later")):
        if state.get("dirty"):
            files = list(state.get("files_changed") or [])
            if files:
                listed = ", ".join(f"`{f}`" for f in files[:6])
                if len(files) > 6:
                    listed += f", and {len(files) - 6} more"
                where = f" in {listed}"
            else:
                where = ""
            # With no commits ahead of the baseline, the patch is not a supplement to the
            # commit range — it is the entire difference.
            whole = (
                " The commit range is empty, so that patch is the whole change."
                if not state.get("commits_ahead")
                else ""
            )
            out.caveats.append(
                f"the {side} run ({run}) had uncommitted changes{where}, so the commit range "
                f"below is not the full difference — its saved working-tree patch is shown "
                f"separately.{whole}"
            )

    have_repo = repo_path.is_dir() and (repo_path / ".git").exists()
    if have_repo and from_sha and to_sha and from_sha != to_sha:
        if not (_commit_exists(repo_path, from_sha) and _commit_exists(repo_path, to_sha)):
            out.caveats.append(
                f"one of {from_sha[:10]}..{to_sha[:10]} is no longer in {name} (rebased or pruned), "
                "so the exact diff cannot be recomputed"
            )
        else:
            rng = f"{from_sha}..{to_sha}"
            out.commits = [ln for ln in (_git(repo_path, "log", "--format=%h %s", rng) or "").splitlines() if ln]
            out.files_changed = [ln for ln in (_git(repo_path, "diff", "--name-only", rng) or "").splitlines() if ln]
            out.diff_stat = (_git(repo_path, "diff", "--stat", rng) or "").strip()
            out.patch, out.patch_truncated = _truncate(_git(repo_path, "diff", rng) or "")
    elif not have_repo:
        out.caveats.append(f"{name} is not available at {repo_path}, so only stored patches can be shown")

    # The later run's own uncommitted work, which is part of what it tested.
    if to_state.get("dirty"):
        stored, path = _stored_patch(runs_dir, to_run, to_state, "worktree_patch_file")
        if stored:
            trimmed, truncated = _truncate(stored)
            out.patch = (out.patch + "\n" if out.patch else "") + trimmed
            out.patch_truncated = out.patch_truncated or truncated
            out.patch_file = path
    return out


CONFIG_KEYS = (
    ("suite", "suite"),
    ("seed", "seed"),
    ("epochs", "epochs"),
    ("judge_model", "judge model"),
    ("organization", "organization"),
    ("base_url", "base url"),
)


def config_changes(from_prov: dict[str, Any], to_prov: dict[str, Any]) -> list[str]:
    """Config differences that make two runs less comparable. Surfaced, never silently averaged."""
    out = []
    for key, label in CONFIG_KEYS:
        before, after = from_prov.get(key), to_prov.get(key)
        if before != after:
            out.append(f"{label}: {before!r} → {after!r}")
    # The harness is part of the measurement: change a scorer and the metric changes, so a
    # score movement across that boundary is not the agent moving. This is easy to forget
    # precisely because the harness is "just the ruler".
    before_env = from_prov.get("env") or {}
    after_env = to_prov.get("env") or {}
    before = before_env.get("harness_fingerprint") or before_env.get("harness_sha")
    after = after_env.get("harness_fingerprint") or after_env.get("harness_sha")
    if before != after:
        out.append(
            f"harness changed: {str(before)[:12]} → {str(after)[:12]} — a scorer, rubric or suite "
            "edit moves scores without the agent changing, so this boundary is not a hill step"
        )

    before_toggles = from_prov.get("feature_toggles") or {}
    after_toggles = to_prov.get("feature_toggles") or {}
    if before_toggles != after_toggles:
        flipped = sorted(
            k for k in set(before_toggles) | set(after_toggles) if before_toggles.get(k) != after_toggles.get(k)
        )
        if flipped:
            out.append(f"feature toggles changed: {', '.join(flipped[:8])}")
    return out


def version_diff(runs_dir: Path, from_run: str, to_run: str) -> VersionDiff:
    """The full account of what changed between two runs."""
    from_prov = load_provenance(runs_dir, from_run) or {}
    to_prov = load_provenance(runs_dir, to_run) or {}
    diff = VersionDiff(
        from_run=from_run,
        to_run=to_run,
        from_label=str(from_prov.get("label") or ""),
        to_label=str(to_prov.get("label") or ""),
        from_change_id=str(from_prov.get("change_id") or ""),
        to_change_id=str(to_prov.get("change_id") or ""),
        config_changes=config_changes(from_prov, to_prov),
    )
    if not from_prov or not to_prov:
        missing = from_run if not from_prov else to_run
        diff.repos = [
            RepoDiff(repo="(unknown)", caveats=[f"no provenance recorded for {missing}, so its code state is unknown"])
        ]
        return diff

    from_repos = {r["name"]: r for r in from_prov.get("repos", [])}
    to_repos = {r["name"]: r for r in to_prov.get("repos", [])}
    for name in sorted(set(from_repos) | set(to_repos)):
        diff.repos.append(
            repo_diff(
                runs_dir,
                from_run,
                to_run,
                from_repos.get(name, {"name": name}),
                to_repos.get(name, {"name": name}),
            )
        )
    return diff


def adjacent_diffs(runs_dir: Path, run_ids: list[str]) -> list[VersionDiff]:
    """One diff per step along the hill: run[i] -> run[i+1]."""
    return [version_diff(runs_dir, run_ids[i], run_ids[i + 1]) for i in range(len(run_ids) - 1)]
