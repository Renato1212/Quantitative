# Spec review

**Date:** 2026-08-09
**Author:** research engineer
**Status:** Phase 0. No code written, per spec §10.
**Requires:** principal's response before Phase 1 begins.

---

## 1. The research question, restated

Given an anchor timestamp `t0` selected by a mechanical, outcome-agnostic trigger, estimate the
conditional distribution of forward price excursion — with emphasis on the right tail — as a
function of context observable at or before `t0`, and quantify how far each context condition
moves that distribution relative to the unconditional base rate for the same trigger family.

The output is descriptive, not predictive: a vocabulary of market states, each with an attached
excursion distribution and an honest confidence interval. Success is a well-measured map,
including the large regions of it that turn out to be flat.

Two things follow that are worth making explicit, because they constrain everything downstream:

- The unit of analysis is a **distribution**, not a point forecast. The right statistic is a
  distributional shift, not an R² or a hit rate.
- The unit of *evidence* is the **base rate contrast**. A conditional tail probability is
  meaningless without the unconditional one and the sampling error on both.

### Where the question is weakest

**W1 — The right tail is where the whole thesis lives, and it is where you have almost no data.**
This is the binding constraint on the entire programme and it deserves a decision before Phase 1,
not a discovery in Phase 4. Rough arithmetic on the spec's own numbers: a family firing ~250
events/year over a ~4-year training block gives ~1,000 events. If P(MFE > 4× stop) is somewhere
around 2–4%, that is 20–40 tail events unconditionally. Split the sample three ways on a context
variable and the tail cells hold single digits. The 95% CI on a proportion estimated from 8 events
spans roughly a factor of three. You will not be able to distinguish "this condition doubles the
tail" from "no effect" at 4× and 6×.

Consequences, and what I would do about them:

- Run a power calculation *now*, before ingest, with the principal's assumed base rates as input.
  State the minimum detectable effect at each tail threshold for the planned `n`. If the honest
  answer is "we can only detect a doubling of P(MFE > 3×)", the principal should know that before
  six months are spent.
- Prefer statistics that use the whole distribution over binary tail-threshold comparisons. A
  peaks-over-threshold / generalised Pareto fit above a pre-registered threshold, or a
  tail-weighted distributional distance, or quantile regression at pre-declared quantiles, all
  extract far more from the same events than `P(MFE > 4×)` does. The threshold choice must be
  pre-registered, because it is otherwise a free parameter with enormous influence.
- Accept fewer conditioning variables. Every additional split is paid for out of tail counts.

**W2 — Event dependence will make every confidence interval in the report too narrow.**
The spec's purge and embargo (C4) handle label-window overlap. They do not handle the fact that
events cluster: several triggers fire on the same session, driven by one shared shock; adjacent
sessions share a volatility regime; a CPI print moves every family at once. The nominal `n` is not
the effective `n`. If bootstrap CIs are drawn i.i.d. over events, they will be materially too
narrow, and that error propagates into the BH correction and any deflated-Sharpe calculation
downstream — meaning findings will "survive correction" that should not.

This is my single biggest methodological objection to the spec as written. Fix: **block bootstrap
by session-day (or by week) everywhere**, and report both the nominal and effective sample size in
every table. I would like this written into §7 as a hard standard alongside the existing ones.

**W3 — The economic thesis is currently unfalsifiable.** "The pre-conditions are partially
observable" is satisfied by any effect size strictly greater than zero, which no finite sample can
rule out. Before Phase 4 there needs to be a number. Something of the shape: *the thesis is
supported only if at least one pre-registered context variable shifts the pre-declared tail
statistic by a factor ≥ X, with a BH-adjusted, block-bootstrapped CI excluding 1.0 on training, and
reproduces in direction with a shift ≥ Y on the sealed holdout.* The principal should set X and Y,
and should set them before seeing any base rates. Otherwise Phase 4 terminates in a shrug that both
parties can read however they prefer.

**W4 — "Large and fast" is undefined, and the definition does most of the work.** Scaling barriers
to ATR makes the metric scale-free, which is right, but the ATR lookback then silently decides what
is being measured. A short lookback makes the study largely a rediscovery of volatility clustering
and vol-of-vol persistence — real, well-documented, and *not* the positioning-imbalance thesis. A
long lookback measures something closer to "unusual relative to the regime". These are different
research questions. Pre-register the lookback, and run every headline result under a second,
pre-declared scaling as a robustness axis — declared in advance so it cannot become a rescue.

The "fast" half of "large and fast" is currently carried only by the time barrier and
`time_to_MFE`. If speed is central to the thesis, it should have a first-class metric — e.g.
excursion velocity in stop-units per unit time, or MFE conditional on being reached within the
first third of the horizon — chosen now rather than reverse-engineered later.

**W5 — Trigger conditioning is itself a selection effect.** Triggers do not fire uniformly across
market states; an initial-balance-extension family fires more often on high-range days almost by
construction. Everything measured is therefore conditional on the trigger having fired, and a
context variable can appear to shift outcomes purely by shifting the *composition* of the events
that fired. This is not fatal — it is why the spec is right to keep families separate and never
pool them — but it means each conditional table must also report the **trigger firing rate** per
context bucket, otherwise a composition effect reads as an outcome effect.

**W6 — The taxonomy, which is the stated deliverable, is never holdout-validated.** Clustering in
Phase 5 is exploratory by construction, and §6 correctly rules exploratory findings ineligible for
the holdout. So the primary deliverable is, by design, a descriptive artefact with no
out-of-sample confirmation. That is acceptable — but it should be stated in the write-up in those
words, not left for a reader to notice. If the principal wants more, the affordable version is to
derive the taxonomy on training and spend *one* pre-registered holdout test on a narrow claim:
cluster assignment stability, plus the rank-ordering of cluster tail probabilities being preserved.

**W7 — There is no stopping rule tied to economics.** A real distributional shift can still be
untradeable after costs, latency, and the principal's own discretion. The spec rightly forbids an
execution layer, but Phase 3 should include a **cost floor**: given commission, spread, and
vol-scaled slippage, what tail probability is required for the phenomenon to clear costs at all?
If the unconditional base rate is already below that floor, the project should change shape at the
Phase 3 gate rather than at Phase 6.

---

## 2. Ambiguities and contradictions to resolve before Phase 1

**A1 — §3 is entirely unfilled.** Instrument, second instrument, study period, session scope, and
data granularity are all placeholders. Nothing can be designed. These are the top blocking items.

**A2 — The holdout definition collides with Phases 6 and 7.** The holdout is "the final 25% of the
period, chronological, sealed". But Phase 6 needs the principal's logged trades, which are being
generated *now* and into the future — i.e. inside and beyond the sealed window — and Phase 7's
nightly refresh consumes new data continuously, which dissolves the seal the moment it runs. Decide
now: I propose the price-data holdout is a fixed calendar window, the trade log is a separate
dataset with its own chronological split, and everything after the study period end becomes a
rolling second holdout that is opened on a fixed cadence and never re-opened.

**A3 — C3 and C4 describe two different validation designs and the relationship is unstated.**
C3 gives train / validation / holdout; C4 gives PurgedKFold. My assumption is that purged CV runs
*inside* the training block only. That leaves the validation block's role undefined. It needs one
paragraph stating exactly which decisions may consume the validation block (hyperparameters?
thresholds? feature selection?) and how many times it may be touched — because each touch is a
trial and belongs in the C5 log.

**A4 — C6's "hypothesis family" is undefined.** BH across 8 hypotheses and BH across 80 are
different studies. Is a family a trigger family, a feature group, a phase, or the whole
pre-registration? Related: C5's trial count (every run ever executed) and C6's family size (the
pre-registered set) are different numbers serving different purposes — the former feeds deflated
Sharpe and general scepticism, the latter feeds BH. Both should be reported, labelled distinctly.

**A5 — The Phase 3 gate is itself a researcher degree of freedom.** Having the principal review
base rates and then decide the project's shape is correct and unavoidable — but it conditions every
later choice on outcome data. It should be logged as such: the decisions taken at that gate get
written into the experiment log, because they inflate the effective trial count even though no
p-value was computed.

**A6 — §5's feature contract has the leakage test backwards.** "No column may correlate with the
label above a stated threshold" will not work. A genuinely useful feature in this domain might
correlate with the label at 0.05; leakage typically shows up at 0.4+. Any threshold low enough to
fire will flag real signal, and any threshold high enough to avoid that will never fire on anything
except catastrophic leaks that the structural checks already catch. Leakage detection should be
structural, not correlational. I propose replacing that clause with:
  - a **future-shift test**: recompute each feature with the point-in-time handle advanced by one
    bar and assert the value changes only where the feature's declared information window says it
    should;
  - a **block label-permutation test**: shuffle labels within time blocks; every effect size must
    collapse to null. If it does not, the pipeline leaks.
  - retain the correlation scan, but as a *report line* that triggers a human look, not a gate.

**A7 — Bar-type ambiguity.** §4 lists time / volume / dollar / imbalance bars; §5's "no gaps
unexplained by the session calendar" is only meaningful for time bars. More importantly, it is not
stated which bar type is canonical for features versus for labels. MFE and MAE computed on bars
understate the true excursion at bar granularity; they should be computed from the finest data
available (tick or 1s), with the barrier-touch fill assumption stated explicitly. I would also drop
imbalance bars from Phase 1 entirely — nothing has asked for them yet (§8, "smallest thing that
answers the question").

**A8 — Roll handling breaks levels-based triggers.** "Levels features use raw, return features use
adjusted" is right in principle, but a prior-day-high trigger spanning a roll compares a level in
contract A to price in contract B. Pick one rule and pre-register it: either suppress levels-based
triggers for the first N sessions of a new front month, or map prior levels across the roll using
the roll-day spread. Then test sensitivity to the choice. The spec already asks whether findings
concentrate in roll weeks — good — but that test only means something once the mapping rule is
fixed.

**A9 — "Outcome-agnostic" needs an operational test, not an assertion.** Proposed definition and
Phase 2 gate addition: a trigger family is outcome-agnostic to the degree its firing rate is
independent of same-session realised range decile. Most of the suggested families will fail this
to some extent. That is not disqualifying, but it must be *measured and reported* at the Phase 2
gate, because trigger construction is the main channel through which outcome information leaks
into an otherwise clean design.

**A10 — The Phase 2 power gate is too lenient given the Phase 3 tail focus.** 150 events/year over
a ~4-year training block is ~600 events, which for 3×/4×/6× tail estimation is already thin — see
W1. I would raise the flag threshold, or better, replace the fixed count with the power calculation
from W1: report, per family, the minimum detectable tail shift at the realised `n`, and let the
principal judge against that rather than against a round number.

**A11 — Phase 5's "out-of-sample R²" is the wrong metric for a heavy-tailed target.** R² on
excursion magnitude will be dominated by a handful of extreme observations and will swing wildly
across folds — it will be near zero or negative most of the time for reasons that have nothing to
do with whether the model learned anything. Choose the metric now, before seeing results: I
recommend out-of-sample log-likelihood improvement over the unconditional distribution, plus
Spearman rank IC, plus pinball loss at the pre-declared quantiles. Keep R² only as a secondary line
so the number the spec expects to be near zero is still visible.

**A12 — Deflated Sharpe Ratio may not apply.** §7 requires DSR "for any strategy-level performance
claim", but this project's deliverable is explicitly not a strategy. Unless and until P&L is
discussed, the analogue for distributional claims is BH plus a stated trial count. Keep DSR in
`src/validation/` but note it is dormant until Phase 6.

**A13 — C7 determinism has unstated preconditions.** Byte-identical output requires: pinned BLAS
thread counts, `PYTHONHASHSEED` fixed, explicit `ORDER BY` on every DuckDB query that feeds an
artefact (SQL row order is not guaranteed), stable Parquet writer settings including compression
level, and no timestamps or absolute paths embedded in artefacts. Worth stating that "byte
identical" applies to a declared artefact manifest rather than to every intermediate file.

**A14 — MLflow *or* SQLite.** "Or" leaves the config-hash and trial-count contract ambiguous. I
recommend the SQLite log: one file, diffable, no server, satisfies §4's no-cloud/no-server rule,
and the trial count is a single `SELECT COUNT(*)`.

**A15 — `asymmetry = MFE / MAE_before` is undefined when `MAE_before = 0`.** At fine granularity
this is common — price moves in your favour from the first tick. The ratio's right tail will
otherwise be a pure artefact of the denominator floor. Pre-register the floor (one tick, or
`MAE_before + half spread`), or use a difference in stop-units instead of a ratio.

**A16 — Deployment target conflicts with §4.** A request has been raised to make this deployable on
Vercel. §4 says "do not introduce a cloud dependency. Local, file-based, reproducible." These are
reconcilable only in one direction: a **static, read-only report site** generated from artefacts
that were produced locally and committed — the hosting holds no data-generating logic and nothing
in the research pipeline depends on it. Any design where research computation, data storage, or
artefact generation happens on Vercel violates C7 and §4 and I would refuse it. See §7, open
question Q8.

---

## 3. Data properties to confirm before ingest can be designed

Nothing below is optional; each one changes the ingest design or invalidates a planned feature.

**Feed content**
1. What Rithmic actually delivers on this account: full order-by-order depth, MBP-10, or Level 1
   trades plus BBO only? This decides whether Phase 5's "order flow features" are possible at all.
   With trades-plus-aggressor only, cumulative delta is available and book imbalance is not.
2. Is the aggressor/side flag exchange-provided or inferred? For CME it should be exchange-provided
   — confirm, because inferred sides carry a systematic error that concentrates exactly in fast
   moves.
3. Are implied/spread-leg trades included in the tape? Calendar-spread-generated prints distort
   volume-bar construction and delta, and they cluster around rolls.

**Timestamps**
4. Which timestamp is stored: exchange gateway time, Rithmic receive time, or local capture time?
   Precision (ns/ms)? Monotonic? Any documented clock offset? Only the exchange timestamp is
   defensible for `t0`; if all we have is local receive time, a latency buffer must be added before
   `t0` and that buffer becomes a config parameter.
5. Storage timezone (UTC assumed) and DST handling. RTH is defined in Chicago time and the UTC
   offset shifts twice a year, on different dates from Europe.

**History and completeness**
6. How far back does the archive actually reach for this instrument at this granularity? Rithmic
   tick history is often shallower than people assume; a 2021 start may simply not be available.
7. Known outages, gaps, resends, and corrections. Critically: **can a bar we ingested last month
   change?** If the vendor issues corrections, determinism requires versioning and checksumming the
   raw snapshot (the spec already does this) and treating each snapshot as a distinct data version.

**Calendars and contract structure**
8. Product calendar: CME holidays, half days, the daily maintenance break, Sunday open. Source?
9. Per-contract or continuous series? Do we have both front and next contract around roll dates —
   required for a volume-based roll rule and for spread-based level mapping (A8)?
10. Official **settlement** prices, which are not the last tape print, and their publication time.
    Prior-day settlement is a legitimate feature only if it was actually published before `t0`.

**Auxiliary data**
11. Economic release calendar: source, and is it **point-in-time**? A calendar file current as of
    today may contain reschedules that were not known at `t0`. Needed for the post-release trigger
    family and as a control everywhere else.
12. Volume-bar threshold policy: ES volume has trended over five years, so a fixed contract-count
    threshold produces very different bar counts in 2021 and 2026. The adaptation rule (e.g.
    threshold from a trailing median of prior sessions' volume) must be strictly backward-looking —
    it is itself a leakage surface.

---

## 4. Phase 1 task breakdown

Ordered by dependency. Each ends in tests.

| # | Task | Notes |
|---|------|-------|
| 1 | Repo skeleton, pinned environment (lockfile), Hydra config, SQLite experiment log | Log records config hash, git SHA, data version, seeds |
| 2 | Raw ingest: Rithmic → partitioned Parquet, immutable, SHA-256 manifest | Pandera schema on write; exchange timestamp in UTC ns as canonical |
| 3 | Session calendar: holidays, half days, maintenance break, RTH/ETH boundaries | Materialised table of session windows in UTC; DST via `zoneinfo`, never manual offsets |
| 4 | Roll handling: front-month determination, roll date table, back-adjusted builder, roll-week flags | Publication timing of the roll decision matters — see L7 |
| 5 | Bar builders: time, volume, dollar | Imbalance bars deferred until something needs them |
| 6 | `PointInTimeView` | The core safety primitive — see L1 |
| 7 | Leakage detector + canary suite | Multiple canaries, not one — see below |
| 8 | Determinism harness | Two full runs, compare artefact manifest checksums |
| 9 | Test suite | Every function that takes `t0` gets a test |

**On the Phase 1 gate:** the spec requires one canary — a feature that peeks one bar ahead. I would
argue that is not sufficient. A detector that catches only the obvious peek gives false confidence
in exactly the cases that matter. I propose the gate requires **one canary per leak mechanism**
below (L1, L3, L4, L6 at minimum), each independently caught.

### Where the leakage risk concentrates, ranked

**L1 — Bar boundary semantics: `<=` versus `<`.** This is the highest-probability leak in the whole
project and it is a one-character bug. A bar "at t0" contains information up to its *close*. If
bars are timestamped by open, then the bar labelled `t0` contains the future. Rule: every bar
carries both `open_ts` and `close_ts`; point-in-time truncation filters on `close_ts <= t0`; and
`t0` must itself be a bar close. Assert this in the `PointInTimeView` constructor.

**L2 — Decision latency.** Even with perfectly correct truncation, a feature computed *at* `t0` is
available to a human at `t0 + δ`. Results that are knife-edge on instantaneous reaction are not
results. Add an explicit `decision_lag` config applied uniformly, and report headline findings at
lag 0 and at one realistic lag.

**L3 — Global normalisers.** Any z-score, ATR scaling, quantile bucketing, or volatility regime
label computed over the full study period leaks global information into every scaled feature *and*
into the ATR-scaled barriers themselves. This one is insidious because it corrupts the labels, not
just the features. All normalisers must be trailing-window and point-in-time.

**L4 — Session-level aggregates.** Session VWAP, value area, initial balance, day type. Every one
of these is naturally written as a full-session computation and must instead be an anytime /
streaming computation evaluated as-of `t0`. Test: compute the feature at `t0` from truncated data
and from the full session, and assert they differ wherever they should.

**L5 — Rolling and EWM without an explicit shift, and centred library windows.** Named in the spec.
The structural test — recompute against a truncated store and assert equality — catches these
generically, which is better than grep-based review.

**L6 — Publication timing of daily levels.** Prior-day settlement is published with a delay; using
it in a 09:35 feature is only valid if it was actually out before 09:35. Same class of error: any
"prior day" quantity whose official value is determined after the session ends.

**L7 — Roll-date determination.** A volume-based roll rule identifies the crossover day using
end-of-day volume, so a feature asserting "we are now in the new front month" must respect when
that was knowable. Intraday on the crossover day, it was not.

**L8 — Label code and feature code sharing a reader.** Labels legitimately look forward; features
must not. Enforce this structurally: labels use a separate reader class, and the features package
cannot import it. A lint rule is cheap insurance.

**L9 — Point-in-time-ness of the economic calendar.** See data question 11.

**L10 — Vendor corrections in raw data.** If a corrected print arrives later and we ingest the
corrected file, we are quietly using post-hoc-cleaned data. Usually unavoidable; version the raw
snapshot and record it as a known, accepted, small leak rather than pretending it is absent.

---

## 5. What I am not doing yet

No code, per §10. No instrument assumptions. No bar type chosen. No trigger families implemented.

---

## 6. Summary of my disagreements with the spec

Stated plainly, since silent compliance is the expensive failure mode:

1. **Bootstrap must be block bootstrap by session-day.** i.i.d. event resampling will produce CIs
   that are too narrow and findings that falsely survive BH correction. (W2)
2. **The correlation-threshold leakage check in §5 does not work** and should be replaced with
   structural tests. (A6)
3. **The Phase 5 R² metric is wrong for a heavy-tailed target** and should be chosen now, before
   results exist. (A11)
4. **The Phase 2 power gate (150/year) is too lenient** for the tail estimation Phase 3 requires.
   (A10)
5. **The thesis needs a pre-declared falsification threshold** or Phase 4 cannot conclude anything.
   (W3)
6. **One leakage canary is not enough** for the Phase 1 gate. (§4 above)
7. **A cloud deployment target conflicts with §4** except in the narrow static-report form. (A16)

---

## 7. Open questions for the principal — blocking Phase 1

- **Q1.** §3 placeholders: primary instrument, secondary instrument, study period, session scope,
  and the granularity Rithmic actually provides on this account.
- **Q2.** Falsification threshold for the thesis — the X and Y in W3. Set before base rates are
  seen.
- **Q3.** ATR lookback for barrier scaling, plus the second pre-registered scaling for robustness.
  And: is "fast" to get a first-class metric, or is `time_to_MFE` sufficient?
- **Q4.** Definition of "hypothesis family" for BH correction, and confirmation that the trial
  count and the family size are reported as two separate numbers.
- **Q5.** The validation block's role: which decisions may consume it, and how many times.
- **Q6.** Holdout policy under Phase 6 and 7 — the rolling-second-holdout proposal in A2.
- **Q7.** Roll rule for levels-based triggers: suppress N sessions, or spread-map. (A8)
- **Q8.** Vercel: confirm this means a static, read-only report site built from locally generated,
  committed artefacts. Anything that runs research computation or stores data in the cloud is a
  refusal under §4 and C7.
