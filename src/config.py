"""Configuration loading and hashing.

Every artefact the pipeline writes carries the hash of the config that produced it.
The hash is taken over a canonical JSON serialisation, so key order in the YAML file
cannot change it and a whitespace edit cannot invalidate a cache.

Hydra is specified in ``CLAUDE.md`` §4 and is deliberately not used here — see decision
D9 in ``research/decisions/2026-08-09-scope-decisions.md``. One config, no sweeps, and
Hydra's working-directory rewriting fights the determinism requirement.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "desk.yaml"


class Config:
    """Read-only view over the config tree, addressed by dotted path."""

    def __init__(self, tree: dict[str, Any], source: Path):
        self._tree = tree
        self.source = source

    def __repr__(self) -> str:
        return f"Config(source={self.source.name!r}, hash={self.hash[:12]!r})"

    def get(self, dotted: str, default: Any = ...) -> Any:
        """``cfg.get("bars.time_seconds")``. Raises on a missing key unless given a default."""
        node: Any = self._tree
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(f"{dotted!r} not in {self.source.name}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.get(name))

    @property
    def tree(self) -> dict[str, Any]:
        """A mutable deep copy. Not a JSON round-trip — that would turn dates into strings."""
        return copy.deepcopy(self._tree)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self._tree, sort_keys=True, separators=(",", ":"), default=_encode)

    @property
    def hash(self) -> str:
        """SHA-256 of the canonical serialisation. Stamped into every artefact."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def with_overrides(self, **dotted: Any) -> Config:
        """A copy with values replaced, e.g. ``cfg.with_overrides(**{"data.root": tmp})``.

        The hash changes, which is the point: a run against a different data root is a
        different run and must not reuse the original's artefacts.
        """
        tree = self.tree
        for path, value in dotted.items():
            node = tree
            *parents, leaf = path.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return Config(tree, self.source)

    @property
    def is_synthetic(self) -> bool:
        """True when the data version marks generated data rather than a real snapshot."""
        return str(self.get("data.version")).startswith("synthetic")


def _encode(obj: Any) -> str:
    if isinstance(obj, (date, time)):
        return obj.isoformat()
    raise TypeError(f"cannot serialise {type(obj).__name__} into the config hash")


def load(path: Path | str | None = None) -> Config:
    source = Path(path) if path else DEFAULT_CONFIG
    tree = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(tree, dict):
        raise ValueError(f"{source} did not parse to a mapping")
    return Config(tree, source)
