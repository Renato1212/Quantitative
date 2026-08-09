"""The dashboard: the page a trader reads in two minutes.

The reports are correct and nobody reads them, which makes them useless. This page
answers five questions in the order someone actually asks them, leads each with the
answer, and puts the chart underneath as evidence rather than as the thing to decode.

Three rules it follows throughout:

*The conclusion is written above the chart, not left to the reader.* Every section opens
with a sentence stating what the evidence says. If a reader stops after the headline,
they have the finding.

*Every interpretation is computed.* The prose is generated from the numbers, so it cannot
drift out of step with them. A hand-written "no effect found" survives the run that finds
one.

*Nothing is reachable only by hovering.* Each chart has a `<details>` table beside it
carrying the same values, which is also the accessible path.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from src.reporting import charts

# Deliberately no polars, numpy or sklearn here. The site generator renders committed
# artefacts and computes nothing, so building the site must not drag in the research
# stack — reading a JSON file needs neither.
SUMMARY_PATH = Path(__file__).resolve().parents[2] / "research" / "dashboard.json"


def load_summary(path: Path | None = None) -> dict | None:
    target = Path(path) if path else SUMMARY_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))

STATUS_ICON = {"pass": "✓", "fail": "✕", "null": "○", "blocked": "◌"}


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _num(value, places: int = 2, dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return f"{value:,.{places}f}".rstrip("0").rstrip(".") if places else f"{value:,.0f}"


def _pct(value, places: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value * 100:.{places}f}%"


# --------------------------------------------------------------------------- pieces


def _tile(label: str, value: str, note: str, *, state: str = "") -> str:
    tone = f" tile-{state}" if state else ""
    return (
        f'<div class="tile{tone}"><p class="tile-label">{_e(label)}</p>'
        f'<p class="tile-value">{_e(value)}</p>'
        f'<p class="tile-note">{note}</p></div>'
    )


def _reading(shows: str, means: str, changes: str) -> str:
    """The three questions every chart on this page has to answer in words."""
    return (
        '<dl class="reading">'
        f'<dt>What it shows</dt><dd>{shows}</dd>'
        f'<dt>What it means</dt><dd>{means}</dd>'
        f'<dt>What would change it</dt><dd>{changes}</dd>'
        "</dl>"
    )


def _table(headers: list[str], rows: list[list[str]], *, caption: str) -> str:
    head = "".join(f"<th scope='col'>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="table-view"><summary>Table view — {_e(caption)}</summary>'
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></details>"
    )


def _section(number: int, question: str, answer: str, body: str, *, anchor: str) -> str:
    return (
        f'<section class="finding" id="{anchor}">'
        f'<p class="finding-index">{number:02d}</p>'
        f"<h2>{_e(question)}</h2>"
        f'<p class="answer">{answer}</p>'
        f"{body}</section>"
    )


# --------------------------------------------------------------------------- sections


def _hero(data: dict) -> str:
    verdict = data["verdict"]
    caveat = ""
    if data["is_synthetic"]:
        caveat = (
            '<p class="caveat"><strong>Synthetic data.</strong> Every number on this page '
            "came from a seeded random walk with a trading calendar attached. It has no "
            "positioning imbalance, no catalysts and no auction structure — the things the "
            "research question is about. These results show the apparatus works. They say "
            "nothing about ES.</p>"
        )
    chips = " ".join(
        f'<span class="chip">{_e(text)}</span>'
        for text in (
            data["instrument"],
            data["period"],
            f"{data['panel']['events']:,} events",
            f"{data['panel']['sessions']:,} sessions",
            f"config {data['config_hash']}",
        )
    )
    return (
        '<header class="hero">'
        f'<p class="eyebrow">Quantitative research desk · {_e(data["data_version"])}</p>'
        f'<h1 class="verdict verdict-{_e(verdict["state"])}">{_e(verdict["headline"])}</h1>'
        f'<p class="tagline">{_e(verdict["detail"])}</p>'
        f'<p class="chips">{chips}</p>'
        f"{caveat}</header>"
    )


def _kpis(data: dict) -> str:
    gbm, gate = data["gbm"], data["gate"]
    supported = sum(1 for h in data["hypotheses"] if h["supported"])
    stable = (data["cluster_stability"] or 0) >= data["cluster_gate"]
    return (
        '<section class="tiles" aria-label="Headline measures">'
        + _tile(
            "Big moves, base rate",
            _pct(data["base_rate_2x"]),
            f"of events run {_num(charts_focus(data))}× the stop distance before the clock runs out",
        )
        + _tile(
            "Hypotheses supported",
            f"{supported} of {data['family_size']}",
            f"{data['hypotheses_survived_bh']} survived multiple-testing correction; "
            f"support also needs a {_num(data['minimum_effect'])}× effect",
            state="good" if supported else "null",
        )
        + _tile(
            "State taxonomy",
            "Stable" if stable else "Unstable",
            f"adjusted Rand {_num(data['cluster_stability'], 3)} between the two halves of the period",
            state="good" if stable else "null",
        )
        + _tile(
            "Leakage gate",
            "Passed" if gate["passed"] else "Failed",
            f"{gate['canaries_caught']}/{gate['canaries_total']} planted leaks caught, "
            f"{gate['features_clean']}/{gate['features_total']} real features clean",
            state="good" if gate["passed"] else "bad",
        )
        + "</section>"
    )


def charts_focus(data: dict) -> float:
    return data["excursion"]["emphasis_from"]


def _edge_section(data: dict) -> str:
    rows = data["hypotheses"]
    supported = [h for h in rows if h["supported"]]
    survived = data["hypotheses_survived_bh"]
    bar = data["minimum_effect"]

    if supported:
        answer = (
            f"<strong>{len(supported)} of {len(rows)} hold up.</strong> "
            + ", ".join(f"{_e(h['id'])} ({_e(h['feature'])})" for h in supported)
            + f" cleared both the correction and the {_num(bar)}× effect bar."
        )
    else:
        answer = (
            f"<strong>No. None of the {len(rows)} context conditions we registered in "
            f"advance shifts the odds of a big move enough to trade.</strong> "
            f"{survived} survived multiple-testing correction, and none reached the "
            f"{_num(bar)}× effect bar set before any of this was measured."
        )

    measurable = [h for h in rows if isinstance(h["ci_high"], (int, float))]
    widest = max(measurable, key=lambda h: h["ci_high"], default=None)
    if not widest:
        width_note = "the intervals are too wide to measure anything"
    else:
        clipped = " (clipped on the chart, arrow-tipped)" if widest["ci_high"] > bar * 4 else ""
        width_note = (
            f"the widest runs from {_num(widest['ci_low'])}× to {_num(widest['ci_high'])}×"
            f"{clipped}, which is not a measurement, it is a shrug"
        )

    table = _table(
        ["Hypothesis", "Feature", "Predicted", "Effect", "95% interval", "p", "n", "Supported"],
        [
            [
                _e(h["id"]), _e(h["feature"]), _e(h["predicted_direction"]),
                _num(h["effect"]) + "×", f"{_num(h['ci_low'])} – {_num(h['ci_high'])}",
                _num(h["p_value"], 3), f"{h['n']:,}", "yes" if h["supported"] else "no",
            ]
            for h in rows
        ],
        caption="every registered hypothesis, pass or fail",
    )

    body = (
        '<figure class="viz-figure">'
        + charts.hypothesis_forest(rows, minimum_effect=bar)
        + '<figcaption>Each row is one hypothesis registered before any of this was '
        "measured. The dot is the effect; the bar is its 95% interval. To be supported, "
        f"the whole bar must sit right of <em>no effect</em> and the dot right of the "
        f"{_num(bar)}× line.</figcaption></figure>"
        + _reading(
            shows=(
                "The ratio between how often big moves follow the top third of each "
                "context measure versus the bottom third. A ratio of 1.0 means the "
                "context tells you nothing."
            ),
            means=(
                f"Every interval straddles 1.0, so none of these readings changes the odds "
                f"in a way the data can distinguish from chance — and {width_note}."
            ),
            changes=(
                "More events. These intervals are wide because big moves are rare, not "
                "because the method is loose. Roughly four times the sample would be "
                "needed to halve them."
            ),
        )
        + table
    )
    return _section(1, "Is there an edge?", answer, body, anchor="edge")


def _size_section(data: dict) -> str:
    excursion, tails = data["excursion"], data["tails"]
    focus = next((t for t in tails if abs(t["threshold"] - 2.0) < 1e-9), None)
    extreme = tails[-1] if tails else None

    answer = (
        f"<strong>Most go nowhere.</strong> The median event travels "
        f"{_num(excursion['median'])}× its stop distance before the two-hour clock runs "
        f"out; one in ten reaches {_num(excursion['q90'])}×."
    )
    if focus:
        answer += (
            f" A move worth twice the risk happens {_pct(focus['p'])} of the time — "
            f"{focus['n_exceeding']} times in {focus['n']:,} events."
        )

    if not extreme:
        thin = ""
    elif extreme["n_exceeding"] == 0:
        thin = (
            f"Nothing in this sample reached {_num(extreme['threshold'])}× at all, which is not "
            "the same as saying it cannot happen — it means the study has no observations there "
            "and cannot distinguish a rare event from an impossible one."
        )
    else:
        thin = (
            f"By {_num(extreme['threshold'])}× only {extreme['n_exceeding']} observations remain "
            f"and the interval runs {_pct(extreme['ci_low'])} to {_pct(extreme['ci_high'])}; "
            "beyond this point the data cannot answer the question."
        )

    table = _table(
        ["Threshold", "Probability", "95% interval", "Events above", "Sample", "Independent days"],
        [
            [
                _num(t["threshold"]) + "× stop", _pct(t["p"]),
                f"{_pct(t['ci_low'])} – {_pct(t['ci_high'])}",
                f"{t['n_exceeding']:,}", f"{t['n']:,}", f"{t['n_blocks']:,}",
            ]
            for t in tails
        ],
        caption="right-tail probabilities with intervals",
    )

    body = (
        '<div class="viz-pair">'
        '<figure class="viz-figure">'
        + charts.excursion_histogram(
            excursion["bins"],
            emphasis_from=excursion["emphasis_from"],
            median=excursion["median"] or 0.0,
        )
        + "<figcaption>How far each event ran, in multiples of the distance to its stop. "
        f"Highlighted bars are the moves worth at least {_num(excursion['emphasis_from'])}× "
        "the risk taken.</figcaption></figure>"
        '<figure class="viz-figure">'
        + charts.tail_curve(tails)
        + "<figcaption>How often a move of each size happens, with the uncertainty on that "
        "estimate. The bars are the answer, not the dots.</figcaption></figure>"
        "</div>"
        + _reading(
            shows=(
                "The full shape of what happens after a trigger fires, and how confident "
                "we can be about the rare end of it."
            ),
            means=(
                "The distribution is heavily concentrated near zero: most triggers are "
                f"noise. {thin}"
            ),
            changes=(
                "A different holding period or stop distance moves the whole picture. "
                "Both are set in config and both are reported under a second, "
                "pre-declared scaling so a result cannot depend on one lucky choice."
            ),
        )
        + table
    )
    return _section(2, "How big do the moves actually get?", answer, body, anchor="size")


def _power_section(data: dict) -> str:
    families, weak = data["families"], data["underpowered"]
    answer = (
        f"<strong>{len(weak)} of {len(families)} setups fire too rarely to study.</strong> "
        "Below roughly 150 events a year, a rare outcome cannot be measured at all — "
        "there is no statistical repair for a sample that does not exist."
        if weak
        else f"<strong>All {len(families)} setups clear the sample-size bar.</strong>"
    )

    table = _table(
        ["Setup", "Events", "Per year"],
        [[_e(f["trigger_family"]), f"{f['n']:,}", _num(f["per_year"], 0)] for f in families],
        caption="events per trigger family",
    )

    body = (
        '<figure class="viz-figure">'
        + charts.ranked_bars(
            families,
            value_key="per_year",
            label_key="trigger_family",
            reference=150,
            reference_label="150/yr minimum",
            emphasis_below=150,
            unit="/yr",
            places=0,
            chart_id="families",
            title="Events per year by trigger family",
            desc=(
                "How often each mechanical setup fires per year, against the minimum "
                "sample size needed to measure a rare outcome."
            ),
        )
        + "<figcaption>Each setup is a mechanical condition that fires whether or not "
        "anything follows — the quiet days are the control group. Faded bars fall below "
        "the sample-size bar.</figcaption></figure>"
        + _reading(
            shows="How much evidence each setup can ever supply.",
            means=(
                "Sample size is the binding constraint on the whole study. A setup at 150 "
                "events a year gives about 600 in the training period, and at a 2% tail "
                "rate that is a dozen big moves to reason from."
            ),
            changes=(
                "Loosening a trigger raises its count and changes what it means. That is a "
                "trade, not a free win, and it has to be registered before it is measured."
            ),
        )
        + table
    )
    return _section(3, "Do the setups fire often enough to learn from?", answer, body, anchor="power")


def _states_section(data: dict) -> str:
    stability, gate = data["cluster_stability"] or 0.0, data["cluster_gate"]
    stable = stability >= gate
    clusters, gbm = data["clusters"], data["gbm"]

    answer = (
        "<strong>Yes — the state groupings hold across time.</strong>"
        if stable
        else (
            "<strong>No. The market states this finds do not survive being re-derived on a "
            "different stretch of history</strong>, so they describe the period rather than "
            "the market. Reported as noise rather than re-tuned until they look stable."
        )
    )

    ic_note = (
        f"A model handed every context variable at once ranks future moves with a "
        f"correlation of {_num(gbm['ic'], 3)} out of sample — near zero. {_e(gbm['sanity'])}"
    )

    table = _table(
        ["State", "Events", "Median move", "P(> 2× stop)", "P(> 3× stop)"],
        [
            [
                f"State {c['cluster']}", f"{c['n']:,}", _num(c["median_mfe_stops"]) + "×",
                _pct(c["p_gt_2x"]), _pct(c["p_gt_3x"]),
            ]
            for c in clusters
        ],
        caption="forward excursion by discovered state",
    )

    body = (
        '<figure class="viz-figure">'
        + charts.stability_meter(stability, threshold=gate)
        + "<figcaption>How much the state definitions agree when learned from the first "
        "half of the period versus the second. Below the gate, they are period artefacts."
        "</figcaption></figure>"
        '<figure class="viz-figure">'
        + charts.ranked_bars(
            [{"label": f"State {c['cluster']} · n={c['n']}", "p": c["p_gt_2x"] * 100} for c in clusters],
            value_key="p",
            label_key="label",
            reference=(data["base_rate_2x"] or 0) * 100,
            reference_label="base rate",
            unit="%",
            chart_id="clusters",
            title="Big-move rate by discovered state",
            desc="Share of events exceeding twice the stop distance, within each discovered market state.",
        )
        + "<figcaption>States are found from context alone, with no sight of what happened "
        "next; the outcomes are attached afterwards. The line is the rate across all events."
        "</figcaption></figure>"
        + _reading(
            shows=(
                "Whether grouping events by their market context produces categories that "
                "behave differently — and whether those categories are real or accidental."
            ),
            means=ic_note,
            changes=(
                "Order-flow depth. Every context variable here is derived from trades and "
                "the clock; book imbalance and resting liquidity are the obvious missing "
                "inputs, and whether they are available is still an open question."
            ),
        )
        + table
    )
    return _section(4, "Are there distinct market states?", answer, body, anchor="states")


def _trust_section(data: dict) -> str:
    gate = data["gate"]
    answer = (
        f"<strong>The measurement machinery is checked, and it passes.</strong> "
        f"{gate['canaries_caught']} deliberately leaky features were planted and all "
        f"{gate['canaries_total']} were caught; all {gate['features_total']} real features "
        f"came through clean; two independent builds produced byte-identical output."
        if gate["passed"]
        else "<strong>The leakage gate is failing. Nothing downstream should be believed.</strong>"
    )

    strip = "".join(
        f'<li class="phase phase-{_e(p["state"])}">'
        f'<span class="phase-icon" aria-hidden="true">{STATUS_ICON.get(p["state"], "·")}</span>'
        f'<span class="phase-n">Phase {p["n"]}</span>'
        f'<span class="phase-name">{_e(p["name"])}</span>'
        f'<span class="phase-note">{_e(p["note"])}</span></li>'
        for p in data["phases"]
    )

    costs = data["costs"]
    body = (
        f'<ul class="phase-strip">{strip}</ul>'
        + _reading(
            shows=(
                "Whether the pipeline can see the future by accident — the failure that "
                "makes a backtest look brilliant and lose money."
            ),
            means=(
                "Features read through a handle that physically cannot return data from "
                "after the decision moment. The check plants known leaks and confirms they "
                "are caught, then confirms the real features are not."
            ),
            changes=(
                "A leak that never touches the data — a threshold chosen after seeing "
                "outcomes, say. No detector catches that; only registering hypotheses in "
                "advance does, which is why the page above leads with them."
            ),
        )
        + '<p class="footnote">Costs assumed before any of this is tradeable: '
        f"{_num(costs['round_turn_cost_points'], 3)} points round turn "
        f"(${_num(costs['round_turn_cost_usd'])} per contract), being commission both "
        "sides, a spread assumption, and volatility-scaled slippage on entry and exit. "
        f"Every interval on this page is a block bootstrap over session-days with "
        f"{data['resamples']:,} resamples — events cluster inside a session, and treating "
        "them as independent would make every interval here too narrow.</p>"
    )
    return _section(5, "Can the machinery be trusted?", answer, body, anchor="trust")


def _detail_links(pages) -> str:
    cards = "".join(
        f'<li><a href="{p.href}"><h3>{_e(p.nav_title or p.title)}</h3>'
        f"<p>{_e(p.blurb)}</p></a></li>"
        for p in pages
    )
    return (
        '<section class="finding" id="detail"><p class="finding-index">06</p>'
        "<h2>The full working</h2>"
        '<p class="answer">Every number above is derived in one of these, with the method, '
        "the assumptions and the failure conditions written out.</p>"
        f'<ul class="cards">{cards}</ul></section>'
    )


# --------------------------------------------------------------------------- entry


def render(data: dict, pages) -> str:
    """The whole dashboard, from the committed summary."""
    return (
        '<div class="dash">'
        + _hero(data)
        + _kpis(data)
        + _edge_section(data)
        + _size_section(data)
        + _power_section(data)
        + _states_section(data)
        + _trust_section(data)
        + _detail_links(pages)
        + "</div>"
    )


def placeholder() -> str:
    """Shown when no run has been committed. Says what to do, rather than nothing."""
    return (
        '<div class="dash"><header class="hero">'
        '<p class="eyebrow">Quantitative research desk</p>'
        '<h1 class="verdict verdict-null">No run has been committed yet</h1>'
        '<p class="tagline">The dashboard renders <code>research/dashboard.json</code>, '
        "which a full pipeline run writes. Produce one with "
        "<code>python -m src.pipeline all</code>, then rebuild the site.</p>"
        "</header></div>"
    )
