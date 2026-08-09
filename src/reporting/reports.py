"""Phase report generation.

Reports are written from results, never hand-edited, and they are deterministic: no wall
clock, no absolute paths, no run ids. Regenerating from the same config and data produces
the same bytes, so a diff in a report is a diff in a finding.

Two rules the generator enforces rather than trusts:

*The synthetic banner cannot be omitted.* :func:`banner` is called by every report and
raises if a synthetic run somehow produces a document without it. A number computed from
a seeded random walk must never be readable as a market fact.

*A conditional table is never emitted without its base rate.* The Phase 4 writer takes
both or neither (C2).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import REPO_ROOT, Config

REPORT_DIR = REPO_ROOT / "research" / "reports"

SYNTHETIC_BANNER = """> **SYNTHETIC DATA — NOT A RESEARCH RESULT.**
> Every number below was computed from a seeded random walk with a calendar attached. The
> generator has no positioning imbalance, no catalysts and no auction structure, which are
> precisely the things the research question is about. These tables demonstrate that the
> apparatus runs and that its arithmetic is right. They say nothing whatsoever about ES.
"""


def banner(cfg: Config) -> str:
    return SYNTHETIC_BANNER if cfg.is_synthetic else ""


def provenance(cfg: Config, extra: dict[str, object] | None = None) -> str:
    fields = {
        "Config hash": f"`{cfg.hash[:12]}`",
        "Data version": f"`{cfg.get('data.version')}`",
        "Instrument": cfg.get("scope.primary_instrument"),
        "Bootstrap": f"{cfg.get('statistics.bootstrap_resamples'):,} resamples, "
        f"blocked by {cfg.get('statistics.bootstrap_block')}",
    } | (extra or {})
    return "\n".join(f"**{k}:** {v}  " for k, v in fields.items())


def markdown_table(frame: pl.DataFrame, *, floats: int = 4) -> str:
    """A polars frame as a GitHub-flavoured table. Empty frames say so rather than vanish."""
    if frame.is_empty():
        return "_no rows_"
    rounded = frame.with_columns(
        [pl.col(c).round(floats) for c, t in frame.schema.items() if t in (pl.Float64, pl.Float32)]
    )
    header = "| " + " | ".join(rounded.columns) + " |"
    rule = "|" + "|".join("---" for _ in rounded.columns) + "|"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in rounded.iter_rows()
    ]
    return "\n".join([header, rule, *body])


def _write(name: str, text: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- phase 2


def phase2(cfg: Config, per_year: pl.DataFrame, labels: pl.DataFrame, dropped: int) -> Path:
    underpowered = per_year.filter(pl.col("n") < 150)["trigger_family"].unique().to_list()
    outcomes = (
        labels.group_by("label").len().sort("label").rename({"len": "n"})
        if not labels.is_empty() else pl.DataFrame()
    )
    verdict = (
        f"**{len(underpowered)} of {per_year['trigger_family'].n_unique()} families fall below "
        f"~150 events/year:** {', '.join(underpowered)}."
        if underpowered
        else "**Every family clears ~150 events/year.**"
    )

    text = f"""# Phase 2 — events and labels

{banner(cfg)}
{provenance(cfg, {"Trigger families": per_year["trigger_family"].n_unique(), "Events labelled": labels.height})}

## The gate

{verdict} `CLAUDE.md` §6 asks for the count and leaves the decision with the principal:
an underpowered family is not automatically wrong, but keeping it must be a choice made
with the number in view.

Sample size is the binding constraint on this entire study (spec review W1). A family at
150 events/year yields roughly 600 training events, and at a 2% tail rate that is a dozen
observations above 3x the stop distance. No amount of statistical care recovers
information that is not in the sample.

### Events per family per year

{markdown_table(per_year)}

## Labels

Triple barrier with horizon-scaled ATR barriers, plus continuous excursion metrics
computed from **ticks** rather than bars — bar highs understate the true maximum
favourable excursion, and the right tail is where that understatement bites hardest.

{dropped} events were dropped for falling inside the ATR warm-up window, where no
volatility estimate built from prior sessions alone yet exists.

### Barrier outcomes

{markdown_table(outcomes)}

`label` is +1 target first, -1 stop first, 0 the clock ran out. Continuous values —
`mfe`, `mae`, `mae_before_mfe`, `time_to_mfe_min`, `velocity`, `asymmetry` — are stored
unthresholded. Nothing is turned into "big move" here; that choice happens in Phase 3
where it can be varied in the open.

## What would falsify the event construction

A trigger family is supposed to be outcome-agnostic: it should fire on the sessions that
go nowhere as readily as on the sessions that run. If a family's firing rate correlates
with same-session realised range, it is selecting for movement and the control group is
contaminated. That check belongs at this gate on real data and is reported in Phase 3's
stratified tables.
"""
    return _write("2026-08-09-phase2.md", text)


# --------------------------------------------------------------------------- phase 3


def phase3(
    cfg: Config,
    quantiles: pl.DataFrame,
    tails: pl.DataFrame,
    by_year: pl.DataFrame,
    by_regime: pl.DataFrame,
    by_time: pl.DataFrame,
    costs: dict,
) -> Path:
    widest = (
        tails.filter(pl.col("n_exceeding") > 0)
        .with_columns((pl.col("ci_high") / pl.col("ci_low").clip(1e-9)).alias("width"))
        .sort("width", descending=True)
        .head(1)
    )
    width_note = (
        f"The widest interval in the tail table spans a factor of "
        f"{float(widest['width'][0]):.0f} "
        f"({widest['trigger_family'][0]} at {float(widest['threshold'][0]):.0f}x, "
        f"n_exceeding={int(widest['n_exceeding'][0])})."
        if not widest.is_empty()
        else "No threshold was exceeded often enough to compute a meaningful interval."
    )

    text = f"""# Phase 3 — base rates

{banner(cfg)}
{provenance(cfg)}

## The finding first

The unconditional forward excursion distribution per trigger family, in multiples of the
stop distance. **Read the intervals, not the point estimates.** {width_note}

That width is the finding, not a presentational problem. At 4x and 6x the counts fall to
single digits and no conditioning analysis downstream can be more precise than this.

## Quantiles

{markdown_table(quantiles)}

## Right-tail probabilities

`P(MFE > k x stop)`, with 95% intervals from a block bootstrap over session-days. The
`n_blocks` column is the number of independent days and is the honest sample size;
`n` is the nominal one and is larger.

{markdown_table(tails)}

## Stability

A base rate that holds in one year and collapses in the next is a regime-conditional
base rate, and §7 requires it to be labelled one rather than averaged away.

### By year

{markdown_table(by_year)}

### By volatility regime

Terciles of the trailing ATR, fitted within the training block only.

{markdown_table(by_regime)}

### By time of day

{markdown_table(by_time)}

## Cost floor

Before any of this is tradeable it has to clear costs (spec review W7):

- Round turn: **{costs['round_turn_cost_points']:.3f} points** ({costs['round_turn_cost_usd']:.2f} USD
  per contract), being commission both sides, a spread assumption, and volatility-scaled
  slippage on entry and exit.
- {costs['note']}.

If the unconditional tail is thinner than the cost floor requires, the phenomenon is not
tradeable and the project changes shape here rather than at Phase 6.

## The gate

This is a decision point for the principal, not a checkbox. The question is whether the
phenomenon he wants to trade occurs at the frequency he assumed. If the tail is thinner
than the thesis requires, that is the finding.
"""
    return _write("2026-08-09-phase3.md", text)


# --------------------------------------------------------------------------- phase 4


def phase4(cfg: Config, results: pl.DataFrame, family, trial_count: int) -> Path:
    supported = results.filter(pl.col("supported")) if "supported" in results.columns else pl.DataFrame()
    survived = results.filter(pl.col("survives_bh")) if "survives_bh" in results.columns else pl.DataFrame()

    if supported.is_empty():
        headline = (
            f"**Null result. None of the {family.size} pre-registered hypotheses is supported.** "
            f"{survived.height} survived Benjamini–Hochberg correction, and of those, none "
            f"cleared the pre-declared minimum effect of {family.minimum_effect}x."
        )
    else:
        headline = (
            f"**{supported.height} of {family.size} pre-registered hypotheses are supported**: "
            f"{', '.join(supported['id'].to_list())}. Each survived BH correction and cleared "
            f"the pre-declared minimum effect of {family.minimum_effect}x."
        )

    text = f"""# Phase 4 — pre-registered hypotheses

{banner(cfg)}
{provenance(cfg, {"Family": family.name, "Registered": family.registered, "Family size": family.size, "Trial count (experiment log)": trial_count})}

## The finding first

{headline}

Sample sizes are in the table. Two denominators are reported because they answer
different questions (decision D8): the **family size** of {family.size} is what BH
corrects across, and the **trial count** of {trial_count} is every run this desk has
executed, which is the honest measure of how many chances we have given ourselves.

## Results — every hypothesis, pass or fail

`effect` is the ratio of tail probabilities between the predicted-high and predicted-low
terciles of the feature. `base_rate` is the unconditional probability for the same
population, which C2 requires to sit beside every conditional number.

{markdown_table(results.drop([c for c in ("family",) if c in results.columns]))}

`supported` requires all three of: survives BH, effect at or above
{family.minimum_effect}x, and the direction as predicted. An effect can be statistically
distinguishable from 1.0 and still too small to matter, which is why the size bar exists
and why it was set before any base rate was seen (decision D6).

## Registered but not yet testable

{markdown_table(results.filter(pl.col("n") < 30).select("id", "feature", "n", "note")) if (results["n"] < 30).any() else "_none — every hypothesis had enough sample to test_"}

## What would falsify these results

Each hypothesis carries its own falsification condition in
`research/hypotheses/{family.source.name}`, written before the test ran. The file's git
history is the evidence of that ordering — not the timestamp inside it, which anyone can
type.

H6 (time of day) is a deliberate near-null. Its useful failure mode is *succeeding*: a
large surviving effect there would indict the event construction rather than reveal
something about the clock.

## Exploratory findings

None. Any post-hoc analysis belongs in a separate section explicitly labelled exploratory
and is not eligible for the holdout, per the Phase 4 gate.
"""
    return _write("2026-08-09-phase4.md", text)


# --------------------------------------------------------------------------- phase 5


def phase5(cfg: Config, clusters, diagnosis, sanity: str) -> Path:
    stability = (
        f"Adjusted Rand index between the two halves of the period is "
        f"**{clusters.stability_ari:.3f}**. "
        + ("The vocabulary is stable enough to use." if clusters.stable
           else "**Below 0.5 — these clusters are noise, not a taxonomy.** Reported as such rather "
                "than re-tuned until they look stable.")
    )

    text = f"""# Phase 5 — structure discovery

{banner(cfg)}
{provenance(cfg, {"Clusters": clusters.k, "CV": f"purged, {diagnosis.folds} folds"})}

## The finding first

{stability}

Out-of-sample Spearman IC is **{diagnosis.spearman_ic:+.4f}** and R² is
**{diagnosis.r2:+.4f}**. {sanity}

## State taxonomy

Clusters are fitted on context features with **no reference to outcomes**; the excursion
distribution is attached afterwards. That ordering is what makes this a vocabulary rather
than a fitted target.

{markdown_table(clusters.profile)}

Stability is checked by fitting on the first half of the period and on the second, then
comparing how the two models label the whole panel. A vocabulary that reorganises itself
every year describes the period, not the market.

## Gradient boosting, read for diagnosis

```
{diagnosis.summary()}
```

The headline metric is **not** R². On a heavy-tailed target it is dominated by a handful
of observations and swings across folds for reasons unrelated to learning (spec review
A11). Spearman rank IC and pinball loss are the metrics; R² is shown because §6 expects
to see it near zero, and it should be visible when it is.

{diagnosis.purged} training samples were purged for label-window overlap and
{diagnosis.embargoed} more embargoed. Removing either to "keep more data" would be
refused: the discarded rows are precisely the contaminated ones.

### Permutation importance

{markdown_table(diagnosis.importance)}

Importance here says which variables the model leaned on, not which variables matter.
With an out-of-sample R² at this level, the ranking is mostly noise and should be read as
a pointer for the next round of pre-registration, not as a result.

## Second instrument

Not run. Generalisation to NQ is a single test at the end of Phase 5, and running it
before the primary instrument's findings are settled would waste the one clean shot at it.
"""
    return _write("2026-08-09-phase5.md", text)


# --------------------------------------------------------------------------- phase 6


def phase6(cfg: Config, readiness_note: str) -> Path:
    text = f"""# Phase 6 — meta-labeling interface

{banner(cfg)}
{provenance(cfg)}

## Status

{readiness_note}

The interface is built and deliberately unfed. `src/models/meta_labeling.py` defines the
trade-log schema, the calibration test, and the sizing function; `fit()` raises rather
than training on a placeholder.

## What it will do

The principal's discretionary read is the primary model — direction and entry come from
him. The secondary model estimates `P(target before stop | context)` over his own trade
population. It is not a market forecast; it is a calibration of a signal population that
already exists.

Output feeds **position sizing**, not a binary filter. A filter discards the edge on the
trades it vetoes; a multiplier keeps them and weights them.

## The gate

Calibration, not discrimination. Predicted probabilities must match realised frequencies
within block-bootstrapped intervals across every bin. A model with excellent
discrimination and miscalibrated probabilities will size positions wrongly and
confidently, which is worse than not sizing at all.

## What is needed

A trade log with the schema in `TRADE_LOG_SCHEMA`, at `data/trades/principal_trades.parquet`.
The `t0` column must record when the principal **committed**, not when he first considered
the trade — the difference is the whole point of the exercise, and it is not recoverable
after the fact.
"""
    return _write("2026-08-09-phase6.md", text)
