"""Resolving a target by name, without the framework importing one.

``EVAL_TARGET`` names the target as ``module:attribute``. It is imported on first use and
cached. No target name is written down anywhere in ``evalkit`` — not in an import, not in
a default, not in a comment — so the framework cannot quietly grow a favourite.

To evaluate a product, write a package exposing a :class:`~evalkit.target.Target` and set
``EVAL_TARGET=my_package:my_target``. Nothing in the framework changes.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .target import Target

#: The only short name the framework knows, because it is the framework's own: the offline
#: target behind ``run --mock``. Every other target is named in full by
#: ``EVAL_TARGET``, so nothing here has to know that any particular product exists.
ALIASES = {"mock": "evalkit.mock:mock_target"}

_cache: dict[str, Target] = {}


def target_ref() -> str:
    return os.environ.get("EVAL_TARGET", "").strip()


def load_target(ref: str | None = None) -> Target:
    """Import and return the target named by ``ref`` (default: ``EVAL_TARGET``).

    ``ref`` is ``module:attr``; the attribute may be a target instance or a zero-argument
    callable returning one. Resolution is cached per ref — a target holds a browser
    launcher and a config, not per-run state.
    """
    ref = (ref or target_ref()).strip()
    if not ref:
        raise ValueError(
            "EVAL_TARGET is not set, so there is no agent to evaluate. Set it to "
            "'module:attribute' in your .env (see .env.example), or to 'mock' for the "
            "framework's offline self-test target."
        )
    ref = ALIASES.get(ref, ref)
    if ref in _cache:
        return _cache[ref]
    if ":" not in ref:
        raise ValueError(
            f"EVAL_TARGET={ref!r} is not resolvable. Use 'module:attribute' "
            f"or one of {', '.join(sorted(ALIASES))}."
        )
    module_name, _, attr = ref.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"EVAL_TARGET={ref!r}: cannot import {module_name!r} ({exc})") from exc
    try:
        obj: Any = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"EVAL_TARGET={ref!r}: {module_name!r} has no attribute {attr!r}") from exc
    resolved = obj() if isinstance(obj, type) else obj
    _cache[ref] = resolved
    return resolved


def reset_cache() -> None:
    """Forget resolved targets (tests that patch ``EVAL_TARGET``)."""
    _cache.clear()
