"""The framework must declare what it imports, and locate the consuming repo root."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import evalkit

FRAMEWORK = Path(evalkit.__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]


def imported_names(path: Path) -> set[str]:
    """Every module named by an import in this file, including inside functions."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


# Distribution name -> the module it provides, where they differ.
IMPORT_NAMES = {"inspect-ai": "inspect_ai", "pyyaml": "yaml", "python-dotenv": "dotenv"}


def declared(section: str) -> set[str]:
    """Requirement names from pyproject: 'core', or the name of an extra."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_bytes().decode())
    project = data["project"]
    reqs = project["dependencies"] if section == "core" else project["optional-dependencies"][section]
    names = {r.split(">")[0].split("=")[0].split("[")[0].strip().lower() for r in reqs}
    return {IMPORT_NAMES.get(n, n.replace("-", "_")) for n in names}


def third_party_imports(root: Path) -> dict[str, set[str]]:
    """Every non-stdlib, non-first-party module imported under ``root``, and by whom."""
    found: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for name in imported_names(path):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top == "evalkit":
                continue
            found.setdefault(top, set()).add(path.name)
    return found


def test_the_framework_imports_only_what_it_declares() -> None:
    """A dependency the framework imports but does not declare works on this machine and
    fails on a fresh install; one it declares but never imports is a download for nothing.
    Both drift silently, so both are checked."""
    imports = third_party_imports(FRAMEWORK)
    core = declared("core")

    undeclared = {mod: sorted(where) for mod, where in imports.items() if mod not in core}
    assert not undeclared, f"imported by evalkit but not in [project.dependencies]: {undeclared}"

    unused = core - set(imports)
    assert not unused, f"declared as a core dependency but never imported by evalkit: {sorted(unused)}"


def test_the_repo_root_is_the_consumers_not_the_installs(tmp_path, monkeypatch) -> None:
    """Installed as a dependency, the framework must locate the *project's* root. Deriving
    it from the package's own __file__ would scatter runs/ and logs/ into site-packages."""
    from evalkit.config import _find_repo_root

    project = tmp_path / "someproject"
    (project / "nested").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")

    monkeypatch.delenv("EVAL_REPO_ROOT", raising=False)
    monkeypatch.chdir(project / "nested")
    assert _find_repo_root() == project.resolve()

    monkeypatch.setenv("EVAL_REPO_ROOT", str(tmp_path))
    assert _find_repo_root() == tmp_path.resolve()
