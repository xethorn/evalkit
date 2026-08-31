"""Run configuration for the framework.

Everything here is about *how an evaluation is shaped* — seed, repeats, judge, where
results go. How to reach the agent under test is the target's business and lives in the
target package.

Names are read from the environment and an optional ``.env`` under the ``EVAL_`` prefix.
``EVAL_ENV_PREFIXES`` may name further prefixes to fall back to, which is how a project
that used to spell these knobs some other way keeps its existing ``.env`` and CI working;
``doctor`` reports every name it had to read under a fallback prefix, so they get migrated
deliberately rather than discovered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _find_repo_root() -> Path:
    """The consuming project's root: where ``.env``, ``logs/`` and ``runs/`` live.

    The framework is installed as a dependency, so its own location on disk says nothing
    about the project being evaluated — deriving this from ``__file__`` would point into
    site-packages and quietly scatter results there. ``EVAL_REPO_ROOT`` wins; otherwise
    walk up from the working directory for a project marker.
    """
    if override := os.environ.get("EVAL_REPO_ROOT"):
        return Path(override).expanduser().resolve()
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root()
load_dotenv(REPO_ROOT / ".env", override=False)

# Provider keys the framework is willing to adopt from an external env file. Deliberately
# an allowlist: `EVAL_SECRETS_ENV` usually points at the .env of the service under test,
# which also holds database passwords and connection strings that have no business in
# this process.
BORROWABLE_SECRETS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")

# Prefixes to look under, in order. The first is the framework's own; any others are
# compatibility with whatever a project called these before. Read once, from the
# environment, because it decides how every other name is read.
ENV_PREFIXES: tuple[str, ...] = tuple(
    part.strip().rstrip("_").upper()
    for part in ("EVAL," + os.environ.get("EVAL_ENV_PREFIXES", "")).split(",")
    if part.strip()
)

# Names that had to be read under a fallback prefix, so `doctor` can say so.
LEGACY_ENV_USED: set[str] = set()


def env(name: str, default: str = "") -> str:
    """Read ``<prefix>_<name>`` under each configured prefix, in order."""
    for index, prefix in enumerate(ENV_PREFIXES):
        value = os.environ.get(f"{prefix}_{name}", "").strip()
        if value:
            if index:
                LEGACY_ENV_USED.add(f"{prefix}_{name}")
            return value
    return default


def env_int(name: str, default: int) -> int:
    raw = env(name)
    return int(raw) if raw else default


def env_bool(name: str, default: bool) -> bool:
    raw = env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _borrow_secrets() -> list[str]:
    """Adopt judge credentials from another repo's .env instead of duplicating them here.

    A secret copied into a second file is a secret that outlives its rotation. Returns the
    names adopted, for `doctor` to report.
    """
    source = env("SECRETS_ENV")
    if not source:
        return []
    path = Path(source).expanduser()
    if not path.is_file():
        return []
    adopted: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if name not in BORROWABLE_SECRETS or os.environ.get(name):
            continue  # never override a key already set in this environment
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[name] = value
            adopted.append(name)
    return adopted


BORROWED_SECRETS = _borrow_secrets()
#: Where they came from, for `doctor`.
BORROWED_FROM = env("SECRETS_ENV")

# Secrets are read by name and never echoed into a run record. `redact()` scrubs their
# values out of anything we are about to persist. Targets add their own via
# `register_secret_env` — a credential the framework has never heard of still must not
# reach a stored transcript.
SECRET_ENV_VARS: list[str] = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
]


def register_secret_env(*names: str) -> None:
    """Declare environment variables whose values must never appear in a record."""
    for name in names:
        if name not in SECRET_ENV_VARS:
            SECRET_ENV_VARS.append(name)


@dataclass
class JudgeConfig:
    """The grader. Pinned by default: an unpinned judge silently redefines the metric."""

    model: str = field(default_factory=lambda: env("JUDGE_MODEL", "anthropic/claude-opus-5"))
    # `None` means "send no temperature at all", which is the only thing that works on
    # current OpenAI reasoning models: gpt-5.x rejects any value but its default with a
    # 400, and the client retries it, so a run appears to hang rather than fail.
    #
    # So judge stability cannot come from temperature. It comes from two other places:
    # grading criterion-by-criterion (a tally is far more reproducible than a free-form
    # 0-10 score), and averaging over `epochs`. Set EVAL_JUDGE_TEMPERATURE only for a
    # model you know accepts it.
    temperature: float | None = field(
        default_factory=lambda: float(env("JUDGE_TEMPERATURE")) if env("JUDGE_TEMPERATURE") else None
    )
    # A judge that cannot answer in this long is misconfigured, not slow.
    timeout_s: int = field(default_factory=lambda: env_int("JUDGE_TIMEOUT_S", 120))
    # A second, different judge on a labelled slice keeps self-preference honest.
    calibration_model: str = field(default_factory=lambda: env("JUDGE_CALIBRATION_MODEL", ""))


@dataclass
class RunConfig:
    """Knobs that define the shape of a run (recorded into the log header)."""

    seed: int = field(default_factory=lambda: env_int("SEED", 20260828))
    epochs: int = field(default_factory=lambda: env_int("EPOCHS", 3))
    max_samples: int = field(default_factory=lambda: env_int("MAX_SAMPLES", 2))
    max_user_turns: int = field(default_factory=lambda: env_int("MAX_USER_TURNS", 6))
    # How far back randomized prompts may reach, in months. Should match the tenant's real
    # data window, or prompts will ask about months with nothing in them.
    data_months_back: int = field(default_factory=lambda: env_int("DATA_MONTHS_BACK", 18))
    log_dir: Path = field(default_factory=lambda: Path(env("LOG_DIR", str(REPO_ROOT / "logs"))))
    runs_dir: Path = field(default_factory=lambda: Path(env("RUNS_DIR", str(REPO_ROOT / "runs"))))
    db_path: Path = field(default_factory=lambda: Path(env("DB", str(REPO_ROOT / "runs" / "results.db"))))
    # Repos whose git state is captured as run provenance. Empty means "ask the target",
    # which knows where its own code lives — see `Settings.repos_under_test`.
    repos: tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            Path(p.strip()).expanduser() for p in env("REPOS").split(",") if p.strip()
        )
    )


@dataclass
class Settings:
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def target(self):  # -> evalkit.target.Target
        """The product under test, resolved from ``EVAL_TARGET``.

        Imported lazily and on the instance, so nothing in the framework imports a target
        at module load — that is the boundary this split exists to keep.
        """
        from .registry import load_target

        return load_target()

    @property
    def repos_under_test(self) -> tuple[Path, ...]:
        return self.run.repos or self.target.default_repos()


def settings() -> Settings:
    """Fresh settings from the current environment (cheap; not cached, so tests can patch env)."""
    return Settings()


def redact(text: str) -> str:
    """Remove known secret values from text bound for a stored record."""
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value and len(value) > 6:
            text = text.replace(value, f"<redacted:{name}>")
    return text
