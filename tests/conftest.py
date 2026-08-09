"""Shared fixtures.

The built data root is session-scoped and small. Every test that needs price data uses
the same one, so the cost is paid once and the tests stay honest about determinism —
they read the same bytes the gate reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config as config_module
from src.ingest.calendar import SessionCalendar
from src.pipeline import build

BUILD_SESSIONS = 25
BUILD_TICKS = 1200


@pytest.fixture(scope="session")
def cfg():
    return config_module.load()


@pytest.fixture(scope="session")
def calendar(cfg):
    return SessionCalendar(cfg)


@pytest.fixture(scope="session")
def window(calendar):
    days = [s.session_date for s in calendar.sessions]
    return days[0], days[BUILD_SESSIONS - 1]


@pytest.fixture(scope="session")
def build_args(window):
    start, end = window
    return {"start": start, "end": end, "ticks_per_session": BUILD_TICKS}


@pytest.fixture(scope="session")
def data_root(tmp_path_factory, cfg, build_args) -> Path:
    root = tmp_path_factory.mktemp("data")
    build(cfg, root, **build_args)
    return root
