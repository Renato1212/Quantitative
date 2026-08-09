"""The committed ``site/`` must match what the generator produces.

Vercel serves ``site/`` verbatim with no build step, so a stale directory silently
publishes yesterday's text. This test is the thing that stops that, and it doubles
as the C7 determinism check for the reporting layer.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

from src.reporting.build_site import DEFAULT_OUT, build, main


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_committed_site_matches_sources() -> None:
    assert main(["--check"]) == 0, "run: python -m src.reporting.build_site"


def test_build_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    build(first)
    build(second)

    assert _tree(first) == _tree(second)
    match, mismatch, errors = filecmp.cmpfiles(
        first, second, sorted(_tree(first)), shallow=False
    )
    assert not mismatch and not errors
    assert match


@pytest.mark.parametrize("expected", ["index.html", "style.css", "spec/index.html"])
def test_expected_artefacts_exist(expected: str) -> None:
    assert (DEFAULT_OUT / expected).is_file()


def test_no_absolute_repo_paths_leak_into_output() -> None:
    """Determinism means output cannot depend on where the repo is checked out."""
    repo = str(Path(__file__).resolve().parents[1])
    for page in DEFAULT_OUT.rglob("*.html"):
        assert repo not in page.read_text(encoding="utf-8"), page
