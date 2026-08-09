"""The deploy configuration must actually publish the site.

This file exists because it once did not. `.vercelignore` held `*` followed by
`!site/**`, which reads like an allowlist and is not one: gitignore semantics exclude the
`site` directory at the first pattern, and a file whose parent directory is excluded
cannot be re-included. The upload contained nothing, the build could not find its output
directory, and the published URL served a 404.

Nothing in the repository caught that, because nothing here had ever asserted anything
about deployment. These tests do.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERCEL_JSON = REPO_ROOT / "vercel.json"
VERCEL_IGNORE = REPO_ROOT / ".vercelignore"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(VERCEL_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ignore_patterns() -> list[str]:
    if not VERCEL_IGNORE.exists():
        return []
    return [
        line.strip()
        for line in VERCEL_IGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _matches(pattern: str, path: str, is_dir: bool) -> bool:
    """Approximate gitignore matching, enough to catch an excluded ancestor."""
    body = pattern.rstrip("/")
    anchored = body.startswith("/")
    body = body.lstrip("/")
    if pattern.endswith("/") and not is_dir:
        return False

    if anchored or "/" in body:
        return fnmatch.fnmatch(path, body) or fnmatch.fnmatch(path, f"{body}/*")
    # Unanchored patterns match any path component.
    return any(fnmatch.fnmatch(part, body) for part in path.split("/"))


def is_ignored(path: str, patterns: list[str], *, is_dir: bool = False) -> bool:
    """Whether ``path`` would be excluded, honouring the excluded-parent rule.

    Gitignore cannot re-include a file inside an excluded directory, so every ancestor is
    resolved first and an excluded one is final. Getting this wrong is exactly the bug
    this module is here to prevent.
    """
    parts = path.split("/")
    for depth in range(1, len(parts) + 1):
        candidate = "/".join(parts[:depth])
        directory = is_dir or depth < len(parts)
        excluded = False
        for pattern in patterns:
            negated = pattern.startswith("!")
            body = pattern[1:] if negated else pattern
            if _matches(body, candidate, directory):
                excluded = not negated
        if excluded:
            return True  # an excluded ancestor cannot be rescued further down
    return False


# --------------------------------------------------------------------------- output


def test_output_directory_exists_and_has_an_index(config):
    output = REPO_ROOT / config["outputDirectory"]
    assert output.is_dir(), f"{output} is missing — run: python -m src.reporting.build_site"
    assert (output / "index.html").is_file()
    assert (output / "style.css").is_file()


def test_every_published_file_survives_vercelignore(config, ignore_patterns):
    """The regression test. Every file Vercel must serve has to reach the build."""
    output = REPO_ROOT / config["outputDirectory"]
    published = sorted(p.relative_to(REPO_ROOT).as_posix() for p in output.rglob("*") if p.is_file())
    assert published, "nothing to publish"

    blocked = [p for p in published if is_ignored(p, ignore_patterns)]
    assert not blocked, f".vercelignore excludes files the deploy needs: {blocked[:5]}"


def test_vercel_json_itself_is_not_ignored(ignore_patterns):
    assert not is_ignored("vercel.json", ignore_patterns)


def test_the_original_broken_pattern_would_be_caught():
    """A guard on the guard: the checker must flag the configuration that broke the site.

    Note which half worked. `!vercel.json` re-included the config, because nothing
    excluded its parent — so Vercel read the settings, looked for `outputDirectory`, and
    found no `site/` at all. Configuration present, content missing.
    """
    broken = ["*", "!site/**", "!vercel.json"]
    assert is_ignored("site/index.html", broken), "the checker must catch the excluded parent"
    assert not is_ignored("vercel.json", broken), "top-level negation is legal and did apply"


def test_data_and_ledger_stay_out_of_the_deploy(ignore_patterns):
    """Market data never leaves the machine, and the run ledger is not a web asset."""
    assert is_ignored("data/raw/ticks.parquet", ignore_patterns)
    assert is_ignored("research/experiments.sqlite", ignore_patterns)


# --------------------------------------------------------------------------- build


def test_build_and_install_are_explicitly_empty(config):
    """`""` means "run nothing". `null` means "detect something", which with a
    requirements.txt at the root is an invitation to attempt a Python build."""
    assert config["installCommand"] == ""
    assert config["buildCommand"] == ""
    assert config["framework"] is None


def test_no_clean_urls_with_directory_indexes(config):
    """`cleanUrls` rewrites `/x.html` to `/x`, which fights `trailingSlash` on a site
    built entirely from `dir/index.html`. The site does not need it."""
    assert not config.get("cleanUrls", False)
    assert config["trailingSlash"] is True


def test_security_headers_cover_every_path(config):
    catch_all = next(h for h in config["headers"] if h["source"] == "/(.*)")
    keys = {entry["key"] for entry in catch_all["headers"]}
    assert {"Content-Security-Policy", "X-Robots-Tag", "X-Frame-Options"} <= keys


def test_stylesheet_is_reachable_under_the_content_security_policy(config):
    """The CSP allows `style-src 'self'`, so the stylesheet must be same-origin."""
    index = (REPO_ROOT / config["outputDirectory"] / "index.html").read_text(encoding="utf-8")
    assert 'href="style.css"' in index or 'href="/style.css"' in index
    assert "https://" not in index.split("<body")[0].replace("http-equiv", "")
