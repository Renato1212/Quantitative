"""The experiment log (C5).

Every run writes a row. The row is what makes a p-value mean anything later: the count
of rows in a phase is the number of tests we ran, and that count feeds both the
Benjamini–Hochberg correction and any deflated Sharpe calculation. Selective reporting
is the failure mode this exists to prevent, so the log is append-only and there is no
delete method.

Note the two different counts, which ``CLAUDE.md`` §C5 and §C6 conflate:

- ``trial_count`` — every run ever executed, including the ones that went nowhere. This
  is the honest denominator for scepticism and for DSR.
- the size of a pre-registered hypothesis family — the denominator for BH within that
  family.

Reporting one as the other flatters the result. Both are available here and the
reporting layer prints both.

A plain SQLite file rather than MLflow: one artefact, diffable, no server, and the trial
count is a single ``SELECT COUNT(*)``. ``CLAUDE.md`` §4 permits either.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    started_utc  TEXT NOT NULL,
    phase        TEXT NOT NULL,
    hypothesis   TEXT,
    config_hash  TEXT NOT NULL,
    git_sha      TEXT NOT NULL,
    data_version TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL,
    seed         INTEGER NOT NULL,
    result       TEXT NOT NULL,
    notes        TEXT
);
CREATE INDEX IF NOT EXISTS runs_phase ON runs (phase);
CREATE INDEX IF NOT EXISTS runs_hypothesis ON runs (hypothesis);
"""


def git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or default
    except (subprocess.SubprocessError, OSError):
        return default


class ExperimentLog:
    """Append-only run ledger."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else REPO_ROOT / "research" / "experiments.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path)
        self._con.executescript(SCHEMA)
        self._con.commit()

    def __enter__(self) -> ExperimentLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def record(
        self,
        cfg: Config,
        *,
        phase: str,
        result: Mapping[str, Any],
        hypothesis: str | None = None,
        notes: str = "",
    ) -> str:
        """Write one run. Returns its id."""
        run_id = uuid.uuid4().hex
        self._con.execute(
            "INSERT INTO runs (run_id, started_utc, phase, hypothesis, config_hash, git_sha,"
            " data_version, is_synthetic, seed, result, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                phase,
                hypothesis,
                cfg.hash,
                git_sha(),
                str(cfg.get("data.version")),
                int(cfg.is_synthetic),
                int(cfg.get("determinism.seed")),
                json.dumps(dict(result), sort_keys=True, default=str),
                notes,
            ),
        )
        self._con.commit()
        return run_id

    def trial_count(self, *, phase: str | None = None, exclude_synthetic: bool = False) -> int:
        """Runs executed. The denominator for how sceptical to be about a p-value."""
        clauses, params = [], []
        if phase:
            clauses.append("phase = ?")
            params.append(phase)
        if exclude_synthetic:
            clauses.append("is_synthetic = 0")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return int(self._con.execute(f"SELECT COUNT(*) FROM runs{where}", params).fetchone()[0])

    def family_size(self, hypothesis_family: str) -> int:
        """Distinct hypotheses tested in a family — the BH denominator."""
        return int(
            self._con.execute(
                "SELECT COUNT(DISTINCT hypothesis) FROM runs WHERE phase = ? AND hypothesis IS NOT NULL",
                (hypothesis_family,),
            ).fetchone()[0]
        )

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self._con.execute(
            "SELECT * FROM runs ORDER BY started_utc DESC, run_id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._con.close()
