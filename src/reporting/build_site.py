"""Render the desk's committed markdown artefacts into a static HTML site.

This is a reporting surface, not part of the research pipeline. It reads markdown
that already exists in the repository and writes HTML into ``site/``. It computes
nothing about the market, reads no data under ``data/``, and has no network access.

The output is committed so that hosting is a pure file serve with no build step —
see ``research/reviews/2026-08-09-spec-review.md`` §A16 for why that constraint
exists. ``tests/test_site_up_to_date.py`` fails if ``site/`` drifts from source.

Determinism (C7): inputs are traversed in sorted order and nothing derived from the
build environment — wall-clock time, absolute paths, hostnames — enters the output.
Running this twice on the same sources yields byte-identical files.

Usage:
    python -m src.reporting.build_site [--check]

``--check`` renders to a temporary directory and diffs against ``site/`` instead of
writing, exiting non-zero on any difference.
"""

from __future__ import annotations

import argparse
import filecmp
import html
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_OUT = REPO_ROOT / "site"

SITE_TITLE = "Quantitative Research Desk"
SITE_TAGLINE = (
    "Measurement apparatus for a discretionary intraday futures desk. "
    "Null results are the primary product."
)


@dataclass(frozen=True)
class Page:
    """One markdown source rendered to one output directory."""

    source: Path  # absolute path to the markdown file
    slug: str  # URL path, no leading or trailing slash
    section: str  # grouping shown on the landing page
    blurb: str  # one line of context on the landing page
    nav_title: str = ""  # short label for the sidebar; falls back to the H1

    @property
    def out_path(self) -> Path:
        return Path(self.slug) / "index.html"

    @property
    def href(self) -> str:
        return f"/{self.slug}/"

    @property
    def depth(self) -> int:
        return len(Path(self.slug).parts)


def discover_pages() -> list[Page]:
    """Collect the markdown artefacts to publish, in a stable order."""
    pages = [
        Page(
            source=REPO_ROOT / "CLAUDE.md",
            slug="spec",
            section="Specification",
            blurb="The desk specification: constraints, phase gates, statistical standards.",
            nav_title="Desk specification",
        )
    ]
    for directory, section, blurb in (
        ("research/reviews", "Reviews", "Methodology review."),
        ("research/reports", "Reports", "Generated research output."),
        ("research/hypotheses", "Hypotheses", "Pre-registered hypothesis."),
    ):
        for md in sorted((REPO_ROOT / directory).glob("*.md")):
            pages.append(
                Page(
                    source=md,
                    slug=f"{directory.split('/')[-1]}/{md.stem}",
                    section=section,
                    blurb=blurb,
                )
            )
    return pages


# --------------------------------------------------------------------------- markdown


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _heading_text(tokens, i: int) -> str:
    inline = tokens[i + 1]
    return "".join(c.content for c in (inline.children or [])) or inline.content


def render_markdown(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return ``(title, body_html, toc)``.

    ``title`` is the first level-1 heading, which is dropped from the body because
    the page template renders it. ``toc`` lists ``(anchor, text)`` for level-2
    headings.
    """
    md = MarkdownIt("commonmark", {"typographer": True}).enable(["table", "strikethrough"])
    tokens = md.parse(text)

    title = SITE_TITLE
    toc: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    drop: set[int] = set()

    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        heading = _heading_text(tokens, i)
        if tok.tag == "h1" and not drop and title == SITE_TITLE:
            title = heading
            drop.update({i, i + 1, i + 2})
            continue
        anchor = _slugify(heading)
        if anchor in seen:
            seen[anchor] += 1
            anchor = f"{anchor}-{seen[anchor]}"
        else:
            seen[anchor] = 0
        tok.attrSet("id", anchor)
        if tok.tag == "h2":
            toc.append((anchor, heading))

    body = md.renderer.render([t for i, t in enumerate(tokens) if i not in drop], md.options, {})
    return title, body, toc


# --------------------------------------------------------------------------- template


def _nav(pages: list[Page], current: str | None) -> str:
    out = []
    last_section = None
    for page in pages:
        title = page.nav_title or read_title(page)
        if page.section != last_section:
            out.append(f'<p class="nav-section">{html.escape(page.section)}</p>')
            last_section = page.section
        cls = ' class="current"' if page.slug == current else ""
        out.append(f'<a href="{page.href}"{cls}>{html.escape(title)}</a>')
    return "\n".join(out)


_TITLE_CACHE: dict[Path, str] = {}


def read_title(page: Page) -> str:
    if page.source not in _TITLE_CACHE:
        for line in page.source.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                _TITLE_CACHE[page.source] = line[2:].strip()
                break
        else:
            _TITLE_CACHE[page.source] = page.source.stem
    return _TITLE_CACHE[page.source]


def shell(*, title: str, nav: str, main: str, depth: int) -> str:
    """Wrap rendered content in the page chrome. ``depth`` sets asset paths."""
    prefix = "../" * depth
    full_title = title if SITE_TITLE in title else f"{title} — {SITE_TITLE}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(full_title)}</title>
<meta name="description" content="{html.escape(SITE_TAGLINE)}">
<meta name="robots" content="noindex">
<link rel="stylesheet" href="{prefix}style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="layout">
<aside class="sidebar">
<a class="brand" href="{prefix or "/"}">{html.escape(SITE_TITLE)}</a>
<nav>
{nav}
</nav>
<p class="phase">Phase 0 &middot; no pipeline code</p>
</aside>
<main id="main">
{main}
</main>
</div>
</body>
</html>
"""


def landing_html(pages: list[Page]) -> str:
    cards = []
    for page in pages:
        cards.append(
            "<li>"
            f'<a href="{page.href}"><h3>{html.escape(page.nav_title or read_title(page))}</h3>'
            f"<p>{html.escape(page.blurb)}</p></a>"
            "</li>"
        )
    return f"""<header class="hero">
<h1>{html.escape(SITE_TITLE)}</h1>
<p class="tagline">{html.escape(SITE_TAGLINE)}</p>
</header>
<section>
<h2>Artefacts</h2>
<ul class="cards">
{chr(10).join(cards)}
</ul>
</section>
<section class="note">
<h2>What this site is</h2>
<p>A read-only rendering of markdown committed to the repository. It is generated
locally and served as static files. No research computation, data storage, or
artefact generation happens here &mdash; that would breach the specification's
local-and-reproducible constraint.</p>
</section>
"""


def page_html(page: Page) -> tuple[str, str, list[tuple[str, str]]]:
    title, body, toc = render_markdown(page.source.read_text(encoding="utf-8"))
    toc_html = ""
    if len(toc) > 2:
        items = "\n".join(
            f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc
        )
        toc_html = f'<nav class="toc"><p>On this page</p><ul>{items}</ul></nav>'
    main = (
        f'<header class="page-head"><p class="eyebrow">{html.escape(page.section)}</p>'
        f"<h1>{html.escape(title)}</h1></header>"
        f'{toc_html}<article class="prose">{body}</article>'
    )
    return title, main, toc


# --------------------------------------------------------------------------- build


def build(out_dir: Path) -> None:
    pages = discover_pages()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copyfile(ASSETS_DIR / "style.css", out_dir / "style.css")

    (out_dir / "index.html").write_text(
        shell(title=SITE_TITLE, nav=_nav(pages, None), main=landing_html(pages), depth=0),
        encoding="utf-8",
    )

    for page in pages:
        title, main, _ = page_html(page)
        target = out_dir / page.out_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            shell(title=title, nav=_nav(pages, page.slug), main=main, depth=page.depth),
            encoding="utf-8",
        )


def _differences(left: Path, right: Path, rel: str = "") -> list[str]:
    cmp = filecmp.dircmp(left, right)
    diffs = [f"{rel}{n} (only in generated)" for n in sorted(cmp.left_only)]
    diffs += [f"{rel}{n} (only in committed)" for n in sorted(cmp.right_only)]
    diffs += [f"{rel}{n} (content differs)" for n in sorted(cmp.diff_files)]
    for sub in sorted(cmp.common_dirs):
        diffs += _differences(left / sub, right / sub, f"{rel}{sub}/")
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed site instead of writing",
    )
    args = parser.parse_args(argv)

    if not args.check:
        build(args.out)
        print(f"wrote {args.out.relative_to(REPO_ROOT)}/")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "site"
        build(generated)
        if not args.out.exists():
            print(f"{args.out} does not exist; run without --check", file=sys.stderr)
            return 1
        diffs = _differences(generated, args.out)
        if diffs:
            print("site/ is stale:", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            print("run: python -m src.reporting.build_site", file=sys.stderr)
            return 1
    print("site/ is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
