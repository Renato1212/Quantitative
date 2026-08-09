"""Inline SVG charts, generated at build time.

No JavaScript and no chart library. The site ships under a strict Content Security
Policy (`default-src 'none'`), it must stay byte-identical across builds (C7), and a
research artefact that needs a CDN to render is not an artefact. Every chart here is
computed in Python and emitted as SVG markup.

Interaction within those limits: each mark carries a native `<title>`, which browsers
surface as a tooltip and screen readers announce, and CSS `:hover` lifts the mark. Every
chart is paired with a `<details>` table in `dashboard.py`, so no value is reachable only
by pointing at it.

Colour follows the job, not the mood:

- **Emphasis** is the default form. One hue for the marks that carry the story, the
  de-emphasis gray for context. Most of these charts have one series, so they carry no
  legend — the title names what is plotted.
- **Status** hues (good / warning / critical) appear only where a colour *means* a state,
  always beside an icon and a word, never alone.
- Text never wears a data colour. Marks are coloured; labels use ink tokens.

Palette slots are validated against this site's own surfaces (`#ffffff` light, `#1d1d22`
dark), not against the reference defaults: series blue 4.4:1 light / 4.6:1 dark,
de-emphasis gray 3.7:1 / 4.6:1, worst adjacent CVD ΔE 19.8. Values live in `style.css`
as custom properties so a theme switch moves them in one place.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

# Geometry, fixed so every chart on the page shares a rhythm.
BAR_MAX = 24  # marks never fill their band; the leftover is air
GAP = 2  # surface gap between touching marks
RADIUS = 4  # rounded data-end
DOT = 4.5  # marker radius, >= 8px diameter
# A single-event bin in the tail is the study's whole subject. Rounded to zero height it
# would disappear exactly where the reader is meant to look.
MIN_BAR = 2


def _ok(value) -> bool:
    """A usable number. JSON has no NaN, so an unmeasurable value arrives as ``None``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: float, places: int = 2) -> str:
    if not _ok(value):
        return "n/a"
    return f"{value:,.{places}f}".rstrip("0").rstrip(".") if places else f"{value:,.0f}"


def _pct(value: float, places: int = 1) -> str:
    if not _ok(value):
        return "n/a"
    return f"{value * 100:.{places}f}%"


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


@dataclass(frozen=True)
class Frame:
    """Plot area inside the SVG canvas."""

    width: int
    height: int
    left: int = 44
    right: int = 16
    top: int = 12
    bottom: int = 34

    @property
    def x0(self) -> float:
        return self.left

    @property
    def x1(self) -> float:
        return self.width - self.right

    @property
    def y0(self) -> float:
        return self.top

    @property
    def y1(self) -> float:
        return self.height - self.bottom

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def _canvas(frame: Frame, body: str, *, title: str, desc: str, chart_id: str) -> str:
    """Wrap marks in an accessible, responsive SVG root."""
    return (
        f'<svg class="viz" viewBox="0 0 {frame.width} {frame.height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-labelledby="{chart_id}-t {chart_id}-d">'
        f'<title id="{chart_id}-t">{_e(title)}</title>'
        f'<desc id="{chart_id}-d">{_e(desc)}</desc>'
        f"{body}</svg>"
    )


def _grid_y(frame: Frame, ticks: list[tuple[float, str]]) -> str:
    """Horizontal hairlines and their labels. Solid, one step off surface, recessive."""
    parts = []
    for y, label in ticks:
        parts.append(f'<line class="viz-grid" x1="{frame.x0}" y1="{y:.1f}" x2="{frame.x1}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="viz-tick" x="{frame.x0 - 8}" y="{y + 3.5:.1f}" text-anchor="end">{_e(label)}</text>'
        )
    return "".join(parts)


def _baseline(frame: Frame) -> str:
    return (
        f'<line class="viz-axis" x1="{frame.x0}" y1="{frame.y1}" '
        f'x2="{frame.x1}" y2="{frame.y1}"/>'
    )


def _nice_ceiling(value: float) -> float:
    """Round an axis maximum up to something a reader can do arithmetic on."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    # A fine ladder matters: rounding 6 up to 10 would compress every mark into the left
    # half of the plot and push the reference lines back on top of each other.
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


# --------------------------------------------------------------------------- histogram


def excursion_histogram(
    bins: list[dict], *, emphasis_from: float, median: float, chart_id: str = "hist"
) -> str:
    """Distribution of forward excursion, in multiples of the stop distance.

    Emphasis, not categorical: the bins at or beyond ``emphasis_from`` carry the series
    hue because they are the study's entire subject, and everything below them is
    context in the de-emphasis gray. Colouring all bins by height would double-encode
    the bar length and burn the only free channel.
    """
    frame = Frame(width=560, height=250)
    if not bins:
        return ""

    peak = max((b["n"] for b in bins), default=0) or 1
    top = _nice_ceiling(peak)
    band = frame.w / len(bins)
    bar_w = min(BAR_MAX, band - GAP)

    ticks = [(frame.y1 - frame.h * f, f"{int(top * f):,}") for f in (0, 0.5, 1.0)]
    marks = []
    for i, b in enumerate(bins):
        height = frame.h * (b["n"] / top)
        x = frame.x0 + band * i + (band - bar_w) / 2
        y = frame.y1 - height
        tail = b["lo"] >= emphasis_from
        cls = "viz-mark" if tail else "viz-mark-muted"
        label = f"{_fmt(b['lo'])}–{_fmt(b['hi'])}× stop: {b['n']:,} events"
        marks.append(
            f'<g class="viz-hit"><title>{_e(label)}</title>'
            f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(height, MIN_BAR):.1f}" rx="{min(RADIUS, bar_w / 2):.1f}"/></g>'
        )

    # Axis labels only where they can be read: first, last, and the emphasis boundary.
    x_labels = []
    for i, b in enumerate(bins):
        show = i == 0 or i == len(bins) - 1 or abs(b["lo"] - emphasis_from) < 1e-9
        if show:
            cx = frame.x0 + band * i + band / 2
            x_labels.append(
                f'<text class="viz-tick" x="{cx:.1f}" y="{frame.y1 + 16:.1f}" '
                f'text-anchor="middle">{_fmt(b["lo"])}×</text>'
            )

    median_x = frame.x0 + frame.w * min(median / (bins[-1]["hi"] or 1), 1.0)
    rule = (
        f'<line class="viz-rule" x1="{median_x:.1f}" y1="{frame.y0}" '
        f'x2="{median_x:.1f}" y2="{frame.y1}"/>'
        f'<text class="viz-annot" x="{median_x + 6:.1f}" y="{frame.y0 + 12}">'
        f"median {_fmt(median)}×</text>"
    )

    body = _grid_y(frame, ticks) + "".join(marks) + rule + _baseline(frame) + "".join(x_labels)
    return _canvas(
        frame,
        body,
        title="Forward excursion distribution",
        desc=(
            f"Histogram of maximum favourable excursion in multiples of the stop distance. "
            f"Median {_fmt(median)}×. Bins at or above {_fmt(emphasis_from)}× are highlighted."
        ),
        chart_id=chart_id,
    )


# --------------------------------------------------------------------------- tail curve


def tail_curve(rows: list[dict], *, chart_id: str = "tail") -> str:
    """``P(MFE > k × stop)`` with its confidence interval at each threshold.

    The whiskers are the message. A point estimate here would imply a precision the
    sample does not have, so the interval is drawn at full weight and the dot is small.
    """
    frame = Frame(width=560, height=250, left=52)
    if not rows:
        return ""

    top = _nice_ceiling(max((r["ci_high"] for r in rows if _ok(r["ci_high"])), default=0.01) or 0.01)
    band = frame.w / len(rows)
    ticks = [(frame.y1 - frame.h * f, _pct(top * f, 0)) for f in (0, 0.5, 1.0)]

    marks = []
    for i, r in enumerate(rows):
        cx = frame.x0 + band * i + band / 2
        y_of = lambda v: frame.y1 - frame.h * min(v / top, 1.0)  # noqa: E731
        lo, hi, p = y_of(r["ci_low"]), y_of(r["ci_high"]), y_of(r["p"])
        label = (
            f"P(MFE > {_fmt(r['threshold'])}× stop) = {_pct(r['p'])}  "
            f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]  "
            f"{r['n_exceeding']} of {r['n']} events"
        )
        marks.append(
            f'<g class="viz-hit"><title>{_e(label)}</title>'
            f'<line class="viz-whisker" x1="{cx:.1f}" y1="{hi:.1f}" x2="{cx:.1f}" y2="{lo:.1f}"/>'
            f'<line class="viz-cap" x1="{cx - 6:.1f}" y1="{hi:.1f}" x2="{cx + 6:.1f}" y2="{hi:.1f}"/>'
            f'<line class="viz-cap" x1="{cx - 6:.1f}" y1="{lo:.1f}" x2="{cx + 6:.1f}" y2="{lo:.1f}"/>'
            f'<circle class="viz-dot" cx="{cx:.1f}" cy="{p:.1f}" r="{DOT}"/></g>'
        )
        marks.append(
            f'<text class="viz-tick" x="{cx:.1f}" y="{frame.y1 + 16:.1f}" '
            f'text-anchor="middle">{_fmt(r["threshold"])}×</text>'
        )
        # Label the count, not the probability: n is what the reader cannot infer.
        marks.append(
            f'<text class="viz-annot" x="{cx:.1f}" y="{min(hi - 8, frame.y1 - 6):.1f}" '
            f'text-anchor="middle">n={r["n_exceeding"]}</text>'
        )

    body = _grid_y(frame, ticks) + "".join(marks) + _baseline(frame)
    return _canvas(
        frame,
        body,
        title="Right-tail probability with 95% intervals",
        desc=(
            "Probability that maximum favourable excursion exceeds each multiple of the "
            "stop distance, with block-bootstrapped 95% confidence intervals and the "
            "number of events above each threshold."
        ),
        chart_id=chart_id,
    )


# --------------------------------------------------------------------------- forest


def hypothesis_forest(
    rows: list[dict], *, minimum_effect: float, chart_id: str = "forest"
) -> str:
    """Effect size and interval per pre-registered hypothesis, against two reference lines.

    A forest plot is the right form because the reader's job is to compare each interval
    to a fixed baseline — 1.0 for "no effect" and the pre-declared minimum for "large
    enough to matter". Both lines are drawn; the reader does no arithmetic.
    """
    if not rows:
        return ""
    row_h = 30
    # The gutter is sized from the longest label, not guessed: a collision between a row
    # label and the plot is the failure this chart is most prone to.
    longest = max((len(f"{r['id']} · {r['feature']}") for r in rows), default=10)
    gutter = min(230, max(120, int(longest * 6.4) + 14))
    frame = Frame(width=620 + gutter - 150, height=58 + row_h * len(rows),
                  left=gutter, bottom=26, top=10)

    finite = [r for r in rows if _ok(r.get("ci_high"))]
    # Cap the axis so the two reference lines stay apart. One wide interval would
    # otherwise compress every other row into the left margin and push "no effect" and
    # the minimum-effect line into the same pixel. Overflowing bars get an arrow and keep
    # their true value in the tooltip and the table.
    widest = max((r["ci_high"] for r in finite), default=minimum_effect * 2)
    top = _nice_ceiling(min(widest, minimum_effect * 4))
    x_of = lambda v: frame.x0 + frame.w * min(max(v, 0.0) / top, 1.0)  # noqa: E731

    # Reference labels ride the top of the plot, staggered when they crowd each other.
    refs = [(1.0, "no effect"), (minimum_effect, f"{_fmt(minimum_effect)}× minimum")]
    rules = ""
    previous_x = None
    for i, (value, label) in enumerate(refs):
        x = x_of(value)
        crowded = previous_x is not None and abs(x - previous_x) < 78
        y = frame.y0 - 1 + (11 if crowded else 0)
        cls = "viz-rule" if i == 0 else "viz-rule-strong"
        rules += (
            f'<line class="{cls}" x1="{x:.1f}" y1="{frame.y0 + 12}" x2="{x:.1f}" y2="{frame.y1}"/>'
            f'<text class="viz-annot" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">{_e(label)}</text>'
        )
        previous_x = x

    marks = []
    for i, r in enumerate(rows):
        cy = frame.y0 + row_h * i + row_h / 2 + 10
        marks.append(
            f'<text class="viz-rowlabel" x="{frame.x0 - 10}" y="{cy + 3.5:.1f}" '
            f'text-anchor="end">{_e(r["id"])} · {_e(r["feature"])}</text>'
        )
        if not (_ok(r.get("effect")) and _ok(r.get("ci_low")) and _ok(r.get("ci_high"))):
            marks.append(
                f'<g class="viz-hit"><title>{_e(r["id"] + ": " + (r.get("note") or "undefined"))}</title>'
                f'<text class="viz-annot" x="{x_of(1.0) + 10:.1f}" y="{cy + 3.5:.1f}">'
                f"undefined — no exceedances to compare</text></g>"
            )
            continue

        lo, point = x_of(r["ci_low"]), x_of(r["effect"])
        clipped = r["ci_high"] > top
        hi = x_of(min(r["ci_high"], top))
        label = (
            f"{r['id']} {r['feature']}: effect {_fmt(r['effect'])}× "
            f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}], p={_fmt(r['p_value'])}, "
            f"n={r['n']} — {'supported' if r['supported'] else 'not supported'}"
        )
        cls = "viz-dot-good" if r["supported"] else "viz-dot"
        end = (
            f'<path class="viz-arrow" d="M{hi - 5:.1f},{cy - 5:.1f} L{hi:.1f},{cy:.1f} '
            f'L{hi - 5:.1f},{cy + 5:.1f}"/>'
            if clipped
            else f'<line class="viz-cap" x1="{hi:.1f}" y1="{cy - 5:.1f}" x2="{hi:.1f}" y2="{cy + 5:.1f}"/>'
        )
        marks.append(
            f'<g class="viz-hit"><title>{_e(label)}</title>'
            f'<line class="viz-whisker" x1="{lo:.1f}" y1="{cy:.1f}" x2="{hi:.1f}" y2="{cy:.1f}"/>'
            f'<line class="viz-cap" x1="{lo:.1f}" y1="{cy - 5:.1f}" x2="{lo:.1f}" y2="{cy + 5:.1f}"/>'
            f"{end}"
            f'<circle class="{cls}" cx="{point:.1f}" cy="{cy:.1f}" r="{DOT}"/></g>'
        )

    x_ticks = "".join(
        f'<text class="viz-tick" x="{x_of(v):.1f}" y="{frame.y1 + 18:.1f}" text-anchor="middle">{_fmt(v)}×</text>'
        for v in (0, top)
    )
    body = rules + "".join(marks) + x_ticks
    return _canvas(
        frame,
        body,
        title="Pre-registered hypotheses: effect size with 95% intervals",
        desc=(
            "One row per registered hypothesis. The dot is the ratio of tail probabilities "
            "between the predicted-high and predicted-low terciles; the bar is its "
            f"confidence interval. A hypothesis is supported only if its interval clears "
            f"1.0 and its effect clears {_fmt(minimum_effect)}×."
        ),
        chart_id=chart_id,
    )


# --------------------------------------------------------------------------- bars


def ranked_bars(
    rows: list[dict],
    *,
    value_key: str,
    label_key: str,
    reference: float | None = None,
    reference_label: str = "",
    emphasis_below: float | None = None,
    unit: str = "",
    places: int = 2,
    chart_id: str = "bars",
    title: str = "",
    desc: str = "",
) -> str:
    """Horizontal bars, one hue, optionally with a threshold line and emphasis.

    Horizontal because the category names are long. Emphasis marks the rows that fail a
    stated bar, which is the only thing the reader needs to spot.
    """
    if not rows:
        return ""
    row_h = 26
    frame = Frame(width=560, height=30 + row_h * len(rows), left=150, bottom=26, top=8)
    top = _nice_ceiling(max([r[value_key] for r in rows] + ([reference] if reference else [])))
    x_of = lambda v: frame.x0 + frame.w * (v / top)  # noqa: E731

    marks = []
    for i, r in enumerate(rows):
        cy = frame.y0 + row_h * i + row_h / 2
        bar_h = min(BAR_MAX - 8, row_h - GAP * 3)
        value = r[value_key]
        weak = emphasis_below is not None and value < emphasis_below
        cls = "viz-mark-muted" if weak else "viz-mark"
        width = max(x_of(value) - frame.x0, 1)
        marks.append(
            f'<text class="viz-rowlabel" x="{frame.x0 - 10}" y="{cy + 3.5:.1f}" '
            f'text-anchor="end">{_e(r[label_key])}</text>'
            f'<g class="viz-hit"><title>{_e(f"{r[label_key]}: {_fmt(value, places)}{unit}")}</title>'
            f'<rect class="{cls}" x="{frame.x0}" y="{cy - bar_h / 2:.1f}" width="{width:.1f}" '
            f'height="{bar_h}" rx="{RADIUS}"/></g>'
            f'<text class="viz-value" x="{frame.x0 + width + 8:.1f}" y="{cy + 3.5:.1f}">'
            f"{_fmt(value, places)}{_e(unit)}</text>"
        )

    rule = ""
    if reference:
        x = x_of(reference)
        rule = (
            f'<line class="viz-rule-strong" x1="{x:.1f}" y1="{frame.y0}" x2="{x:.1f}" y2="{frame.y1}"/>'
            f'<text class="viz-annot" x="{x:.1f}" y="{frame.y1 + 16:.1f}" text-anchor="middle">'
            f"{_e(reference_label)}</text>"
        )

    return _canvas(frame, rule + "".join(marks), title=title, desc=desc, chart_id=chart_id)


# --------------------------------------------------------------------------- meter


def stability_meter(value: float, *, threshold: float, chart_id: str = "ari") -> str:
    """A single ratio against a limit. A meter, not a one-bar bar chart.

    The unfilled track is a lighter step of the fill's own ramp so the state reads across
    the whole bar rather than only where the fill ends.
    """
    frame = Frame(width=560, height=64, left=8, right=8, top=8, bottom=28)
    pos = frame.x0 + frame.w * min(max(value, 0.0), 1.0)
    gate = frame.x0 + frame.w * threshold
    good = value >= threshold
    body = (
        f'<rect class="viz-track" x="{frame.x0}" y="{frame.y0}" width="{frame.w:.1f}" height="14" rx="7"/>'
        f'<g class="viz-hit"><title>Adjusted Rand index {_fmt(value, 3)} against a {_fmt(threshold, 2)} gate</title>'
        f'<rect class="{"viz-meter-good" if good else "viz-meter-bad"}" x="{frame.x0}" '
        f'y="{frame.y0}" width="{max(pos - frame.x0, 2):.1f}" height="14" rx="7"/></g>'
        f'<line class="viz-rule-strong" x1="{gate:.1f}" y1="{frame.y0 - 4}" x2="{gate:.1f}" y2="{frame.y0 + 18}"/>'
        f'<text class="viz-annot" x="{gate:.1f}" y="{frame.y0 + 34}" text-anchor="middle">'
        f"stable from {_fmt(threshold, 2)}</text>"
        f'<text class="viz-value" x="{frame.x0}" y="{frame.y0 + 34}">ARI {_fmt(value, 3)}</text>'
    )
    return _canvas(
        frame,
        body,
        title="Cluster stability",
        desc=(
            f"Adjusted Rand index of {_fmt(value, 3)} between cluster assignments fitted on "
            f"the two halves of the period, against a {_fmt(threshold, 2)} stability gate."
        ),
        chart_id=chart_id,
    )
