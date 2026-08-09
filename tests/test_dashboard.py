"""The dashboard and its chart layer.

Two failure modes matter here and neither is a crash. A chart can render cleanly and
still mislead — a missing value drawn as zero, an interval clipped without saying so. And
the prose can drift out of step with the numbers it describes, which is worse than no
prose at all, because it reads as authoritative.

These tests pin the honesty properties: unmeasurable values never render as numbers, the
synthetic banner cannot be dropped, the verdict is computed rather than written, and every
chart has a table beside it.
"""

from __future__ import annotations

import json
import re

import pytest

from src.reporting import charts, dashboard

MINIMUM_EFFECT = 1.5


@pytest.fixture(scope="module")
def summary() -> dict:
    data = dashboard.load_summary()
    if data is None:
        pytest.skip("no committed run; produce one with `python -m src.pipeline all`")
    return data


@pytest.fixture(scope="module")
def pages():
    from src.reporting.build_site import discover_pages

    return discover_pages()


@pytest.fixture(scope="module")
def page(summary, pages) -> str:
    return dashboard.render(summary, pages)


# --------------------------------------------------------------------------- charts


def test_every_chart_is_labelled_for_assistive_technology(page):
    """An SVG without a title and desc is an image with no alt text."""
    svgs = re.findall(r"<svg\b[^>]*>", page)
    assert len(svgs) >= 5
    for tag in svgs:
        assert 'role="img"' in tag
        assert "aria-labelledby=" in tag
    assert page.count("<title id=") == len(svgs)
    assert page.count("<desc id=") == len(svgs)


def test_chart_ids_are_unique(page):
    """Duplicate ids would make aria-labelledby point at the wrong chart."""
    ids = re.findall(r'<(?:title|desc) id="([^"]+)"', page)
    assert len(ids) == len(set(ids))


def test_charts_are_responsive_not_fixed_width(page):
    """A fixed pixel width would overflow the column on a phone."""
    for tag in re.findall(r"<svg\b[^>]*>", page):
        assert "viewBox=" in tag
        assert not re.search(r'\swidth="\d', tag)


def test_unmeasurable_values_never_render_as_a_number():
    """JSON has no NaN, so an undefined effect arrives as None. It must not become 0."""
    rows = [
        {"id": "H1", "feature": "f", "effect": None, "ci_low": None, "ci_high": None,
         "p_value": None, "n": 100, "supported": False, "note": "undefined: no exceedances"},
    ]
    svg = charts.hypothesis_forest(rows, minimum_effect=MINIMUM_EFFECT)
    assert "undefined" in svg
    assert "<circle" not in svg, "an undefined effect must not be plotted as a point"


def test_forest_marks_a_clipped_interval_with_an_arrow():
    """An interval running past the axis must say so, not end in a tidy cap."""
    rows = [
        {"id": "H1", "feature": "wide", "effect": 2.0, "ci_low": 0.5, "ci_high": 99.0,
         "p_value": 0.4, "n": 100, "supported": False, "note": ""},
        {"id": "H2", "feature": "narrow", "effect": 1.1, "ci_low": 0.9, "ci_high": 1.3,
         "p_value": 0.8, "n": 100, "supported": False, "note": ""},
    ]
    svg = charts.hypothesis_forest(rows, minimum_effect=MINIMUM_EFFECT)
    assert "viz-arrow" in svg
    assert "99" in svg, "the true upper bound stays in the tooltip"


def test_forest_reference_lines_never_overlap():
    """1.0 and the minimum-effect bar sit close together; their labels must not collide."""
    rows = [{"id": "H1", "feature": "f", "effect": 1.2, "ci_low": 0.8, "ci_high": 1.9,
             "p_value": 0.3, "n": 100, "supported": False, "note": ""}]
    svg = charts.hypothesis_forest(rows, minimum_effect=MINIMUM_EFFECT)
    labels = re.findall(r'<text class="viz-annot" x="([\d.]+)" y="([\d.]+)"', svg)
    positions = [(float(x), float(y)) for x, y in labels]
    for i, (x1, y1) in enumerate(positions):
        for x2, y2 in positions[i + 1:]:
            assert abs(x1 - x2) > 40 or abs(y1 - y2) > 6, "reference labels overlap"


def test_empty_inputs_render_nothing_rather_than_crashing():
    assert charts.excursion_histogram([], emphasis_from=2.0, median=0.0) == ""
    assert charts.tail_curve([]) == ""
    assert charts.hypothesis_forest([], minimum_effect=1.5) == ""
    assert charts.ranked_bars([], value_key="v", label_key="l") == ""


def test_histogram_keeps_a_one_event_bin_visible():
    """A tail bin of one event is the study's whole subject; it cannot be a zero-height bar."""
    bins = [{"lo": 0.0, "hi": 0.25, "n": 400}, {"lo": 2.0, "hi": 2.25, "n": 1}]
    svg = charts.excursion_histogram(bins, emphasis_from=2.0, median=0.4)
    heights = [float(h) for h in re.findall(r'<rect class="viz-mark[^"]*"[^>]*height="([\d.]+)"', svg)]
    assert min(heights) >= 2


def test_charts_are_deterministic():
    """C7 reaches the site: the same summary must produce the same bytes."""
    rows = [{"threshold": 2.0, "p": 0.02, "ci_low": 0.0, "ci_high": 0.05,
             "n": 292, "n_exceeding": 6, "n_blocks": 46}]
    assert charts.tail_curve(rows) == charts.tail_curve(rows)


# --------------------------------------------------------------------------- page


def test_every_section_is_present(page):
    for anchor in ("edge", "size", "power", "states", "trust", "detail"):
        assert f'id="{anchor}"' in page


def test_every_chart_has_a_table_beside_it(page):
    """Nothing may be reachable only by pointing at it."""
    assert page.count("table-view") >= 4
    assert page.count("<dl class=\"reading\">") >= 4


def test_each_section_states_its_conclusion_before_its_chart(page):
    """The reader who stops after the headline still has the finding."""
    for section in re.findall(r'<section class="finding"[^>]*>.*?</section>', page, re.S):
        if "<svg" not in section:
            continue
        answer = section.index('class="answer"')
        assert answer < section.index("<svg"), "a chart appears before its conclusion"


def test_synthetic_runs_cannot_lose_their_warning(summary, page):
    if summary["is_synthetic"]:
        assert "Synthetic data." in page
        assert "say nothing about ES" in page


def test_no_python_none_or_nan_leaks_into_the_page(page):
    for token in (">None<", ">nan<", ">NaN<", "None×", "nan×"):
        assert token not in page, f"{token} rendered as text"


def test_the_verdict_is_computed_from_the_data(summary, pages):
    """A hand-written headline survives the run that contradicts it. This one cannot."""
    null_run = json.loads(json.dumps(summary))
    null_run["is_synthetic"] = False
    null_run["verdict"] = {
        "state": "finding", "headline": "2 of 6 hypotheses supported", "detail": "provisional",
    }
    for row in null_run["hypotheses"][:2]:
        row["supported"] = True
    rendered = dashboard.render(null_run, pages)
    assert "2 of 6" in rendered
    assert "hold up" in rendered
    assert "None of the" not in rendered


def test_underpowered_setups_are_named_not_just_counted(summary, page):
    if summary["underpowered"]:
        assert "fire too rarely to study" in page


def test_placeholder_says_what_to_do(pages):
    rendered = dashboard.placeholder()
    assert "python -m src.pipeline all" in rendered
    assert "No run has been committed" in rendered


def test_render_is_deterministic(summary, pages):
    assert dashboard.render(summary, pages) == dashboard.render(summary, pages)


# --------------------------------------------------------------------------- summary


def test_summary_has_no_non_finite_numbers(summary):
    """JSON cannot hold NaN or Infinity; anything unmeasurable must already be null."""
    text = json.dumps(summary)
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text


def test_summary_carries_its_provenance(summary):
    for key in ("config_hash", "data_version", "is_synthetic", "instrument", "period"):
        assert key in summary


def test_events_per_year_is_annualised_from_sessions(summary):
    """A six-week sample is not a year; dividing by calendar years touched understates it."""
    scale = 252 / summary["annualised_from_sessions"]
    for family in summary["families"]:
        assert family["per_year"] == pytest.approx(family["n"] * scale, rel=1e-6)


def test_tail_probabilities_are_pooled_over_every_family(summary):
    """The dashboard leads with one curve; each threshold must appear exactly once."""
    thresholds = [t["threshold"] for t in summary["tails"]]
    assert len(thresholds) == len(set(thresholds))
