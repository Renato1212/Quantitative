"""Configuration and its hash.

A maintenance plan is a spending decision. Six months later somebody will ask why truck
BL-0147 was pulled off a pharma run on a Tuesday, and the only defensible answer names the
prices that were in force when the plan was made. So every artefact carries the hash of the
config that produced it, and the hash is taken over a canonical serialisation — reordering
the YAML does not change it, changing a price does.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "bayline.yaml"


@dataclass(frozen=True)
class Component:
    """One failure mode, with the physics and the prices that go with it."""

    id: str
    name: str
    driver: str
    weibull_shape: float
    characteristic_life: float
    immobilising: bool
    safety_critical: bool
    repair_eur: float
    planned_eur: float
    bay_hours: float
    sensor: str

    @property
    def deferrable(self) -> bool:
        """Safety-critical work is never traded against money. It is not an input."""
        return not self.safety_critical


class Config:
    """Read-only view over the config tree, addressed by dotted path."""

    def __init__(self, tree: dict[str, Any], source: Path):
        self._tree = tree
        self.source = source

    def __repr__(self) -> str:
        return f"Config({self.source.name!r}, hash={self.hash[:12]!r})"

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._tree
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(f"{dotted!r} not in {self.source.name}")
                return default
            node = node[part]
        return node

    @property
    def tree(self) -> dict[str, Any]:
        """A mutable deep copy. Not a JSON round-trip, which would stringify the dates."""
        return copy.deepcopy(self._tree)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self._tree, sort_keys=True, separators=(",", ":"), default=str)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def with_overrides(self, **dotted: Any) -> Config:
        """A copy with values replaced. The hash changes, which is the point."""
        tree = self.tree
        for path, value in dotted.items():
            node = tree
            *parents, leaf = path.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return Config(tree, self.source)

    # ------------------------------------------------------------------ typed views

    @property
    def components(self) -> list[Component]:
        return [Component(**entry) for entry in self.get("components")]

    def component(self, component_id: str) -> Component:
        for component in self.components:
            if component.id == component_id:
                return component
        raise KeyError(f"unknown component {component_id!r}")

    @property
    def depots(self) -> list[dict]:
        return list(self.get("fleet.depots"))

    @property
    def vehicle_classes(self) -> list[dict]:
        return list(self.get("fleet.classes"))

    @property
    def contracts(self) -> list[dict]:
        return list(self.get("economics.contracts"))

    @property
    def end_date(self) -> date:
        value = self.get("history.end_date")
        return value if isinstance(value, date) else date.fromisoformat(str(value))

    @property
    def warehouse(self) -> Path:
        root = Path(self.get("storage.warehouse"))
        return root if root.is_absolute() else REPO_ROOT / root

    @property
    def app_dir(self) -> Path:
        root = Path(self.get("storage.app_dir"))
        return root if root.is_absolute() else REPO_ROOT / root

    @property
    def seed(self) -> int:
        return int(self.get("determinism.seed"))


def load(path: Path | str | None = None) -> Config:
    source = Path(path) if path else DEFAULT_CONFIG
    tree = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(tree, dict):
        raise ValueError(f"{source} did not parse to a mapping")
    _validate(tree)
    return Config(tree, source)


def _validate(tree: dict) -> None:
    """Fail at load, not three modules downstream.

    Shares that do not sum to one silently reweight the whole simulated fleet, and the
    resulting bias looks like a finding rather than a typo.
    """
    for path in ("fleet.classes", "economics.contracts"):
        node: Any = tree
        for part in path.split("."):
            node = node[part]
        total = sum(entry["share"] for entry in node)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{path} shares sum to {total:.4f}, not 1.0")

    ids = [entry["id"] for entry in tree["components"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate component ids")

    for entry in tree["components"]:
        if entry["planned_eur"] >= entry["repair_eur"]:
            raise ValueError(
                f"component {entry['id']}: planned cost is not below the unplanned repair "
                "cost, so there is no case for ever scheduling it"
            )
