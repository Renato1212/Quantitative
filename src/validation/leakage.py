"""Leakage detection.

The premise: a feature that only reads through ``PointInTimeView`` cannot tell whether
the data after its cutoff exists. Change the future and its answer must not move. So the
detector does exactly that, twice, and compares.

*Truncation.* Rebuild the data root containing only rows knowable by the cutoff, and
recompute. Catches anything that reads rows it should not be able to see.

*Perturbation.* Rebuild the data root with post-cutoff numeric values scrambled and
pre-cutoff values untouched, and recompute. Catches a feature that reads the future but
tolerates its absence — one whose fallback happens to return the same number.

The two are complementary, not ordered. Perturbation touches float columns only, so a
feature reading future *timestamps* slips past it and is caught by truncation; a future
value entering through a clamp that usually binds slips past truncation and is caught by
perturbation. ``tests/test_gate.py`` pins one example of each direction, so neither probe
can be dropped as redundant.

Both swap the whole data root rather than handing the feature a different object,
because that is how the leak actually happens in practice — a direct Parquet read, a
frame cached at import, a helper that takes the path instead of the view. Substituting
the view would only catch the leaks that already went through it.

``CLAUDE.md`` §6 asks the Phase 1 gate for one canary that peeks a bar ahead. One canary
proves the detector catches the leak you already thought of. ``canaries.py`` defines one
per mechanism from the spec review's ranked list, and the gate requires all of them.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.data.point_in_time import PointInTimeView
from src.data.store import Store

Feature = Callable[[PointInTimeView], float]

# Features are floats built from prices; equality is up to floating-point reassociation,
# not bit equality. A leak moves a value far more than this.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Finding:
    t0: datetime
    probe: str
    baseline: float
    probed: float

    @property
    def gap(self) -> float:
        if math.isnan(self.baseline) and math.isnan(self.probed):
            return 0.0
        if math.isnan(self.baseline) or math.isnan(self.probed):
            return math.inf
        return abs(self.baseline - self.probed)

    def __str__(self) -> str:
        return (
            f"{self.probe} @ {self.t0.isoformat()}: "
            f"{self.baseline!r} -> {self.probed!r} (gap {self.gap:.6g})"
        )


@dataclass(frozen=True)
class AuditResult:
    feature: str
    findings: tuple[Finding, ...]
    anchors_tested: int

    @property
    def leaks(self) -> bool:
        return bool(self.findings)

    def summary(self) -> str:
        if not self.leaks:
            return f"{self.feature}: clean over {self.anchors_tested} anchors"
        probes = sorted({f.probe for f in self.findings})
        return (
            f"{self.feature}: LEAKS ({len(self.findings)}/{self.anchors_tested} anchors, "
            f"probes: {', '.join(probes)}) — first: {self.findings[0]}"
        )


def _evaluate(feature: Feature, cfg: Config, root: Path, t0: datetime, *, strict: bool) -> float:
    # The config the feature sees points at the probed root too. A canary that reaches
    # for a path instead of the view must reach for the *probed* path, or the probe
    # proves nothing.
    scoped = cfg.with_overrides(**{"data.root": str(root)})
    store = Store.from_config(scoped, root=root)
    try:
        view = PointInTimeView(store, t0, scoped)
        return float(feature(view))
    except Exception:
        if strict:
            # On complete data a feature must run. Failing here is a bug in the feature,
            # not evidence about lookahead, and swallowing it hides both.
            raise
        # Under a probe, blowing up is an answer: the feature needed the future.
        return math.nan
    finally:
        store.close()


def audit_all(
    features: dict[str, Feature],
    t0s: Sequence[datetime],
    cfg: Config,
    data_root: Path,
    *,
    probes: Sequence[str] = ("truncation", "perturbation"),
) -> dict[str, AuditResult]:
    """Audit every feature at every anchor.

    Probe roots are built once per anchor and shared across features. Rebuilding them
    per feature would multiply the cost by the size of the feature set for no extra
    information — the probed data does not depend on who reads it.
    """
    findings: dict[str, list[Finding]] = {name: [] for name in features}
    seed = cfg.get("determinism.seed")
    source = Store.from_config(cfg, root=data_root)

    try:
        for t0 in t0s:
            cutoff = PointInTimeView(source, t0, cfg).cutoff
            baselines = {
                n: _evaluate(f, cfg, data_root, t0, strict=True) for n, f in features.items()
            }
            for probe in probes:
                with tempfile.TemporaryDirectory() as tmp:
                    target = Path(tmp) / "root"
                    if probe == "truncation":
                        source.truncated_copy(cutoff, target).close()
                    elif probe == "perturbation":
                        source.perturbed_copy(cutoff, target, seed).close()
                    else:
                        raise ValueError(f"unknown probe {probe!r}")
                    for name, feature in features.items():
                        probed = _evaluate(feature, cfg, target, t0, strict=False)
                        finding = Finding(t0, probe, baselines[name], probed)
                        if finding.gap > TOLERANCE:
                            findings[name].append(finding)
    finally:
        source.close()

    return {
        name: AuditResult(name, tuple(found), len(t0s))
        for name, found in sorted(findings.items())
    }


def audit(
    feature: Feature,
    t0s: Sequence[datetime],
    cfg: Config,
    data_root: Path,
    *,
    name: str | None = None,
    probes: Sequence[str] = ("truncation", "perturbation"),
) -> AuditResult:
    """Audit a single feature. Convenience wrapper over :func:`audit_all`."""
    label = name or getattr(feature, "__name__", repr(feature))
    return audit_all({label: feature}, t0s, cfg, data_root, probes=probes)[label]
