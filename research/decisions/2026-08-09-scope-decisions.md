# Scope decisions

**Date:** 2026-08-09
**Decided by:** research engineer, under instruction from the principal to make the calls
**Status:** in force, and reversible

The principal delegated the open questions in
[`research/reviews/2026-08-09-spec-review.md`](../reviews/2026-08-09-spec-review.md) §7. Each
decision below fills a `<<< >>>` placeholder in `CLAUDE.md` §3 or answers one of Q1–Q8.

Every one is written into `config/desk.yaml` and hashed into every artefact, so overturning a
decision invalidates the artefacts built under it rather than silently changing their meaning.
**Nothing here is a finding.** These are choices made to unblock Phase 1, and any of them can be
reversed on a sentence from the principal.

---

## D1 — Primary instrument: ES

The spec's own example, the deepest equity index book, and the instrument an auction-market
intraday trader is most likely to be reading. Depth matters more than usual here: the research
question is about the right tail of excursion, and a thinner book contaminates tail measurement
with liquidity artefacts that have nothing to do with positioning imbalance.

*Falsified by:* the principal actually trading NQ or CL as his primary. Say so and this changes.

## D2 — Secondary instrument: NQ

Phase 5 only, one run. NQ is the awkward choice on purpose. It is correlated enough with ES that
a genuine structural effect should carry across, and different enough in volatility and
participant mix that a fitted artefact should not. A less correlated instrument (CL, ZN) would
make failure to generalise uninformative — too many reasons for it to fail.

## D3 — Study period: 2021-01-01 .. 2026-06-30

Five and a half years. The lower bound is a compromise: starting in 2021 excludes the March 2020
dislocation, which would dominate every tail statistic and is not a regime the desk expects to
trade again soon. Starting later would cost sample the tail cannot spare.

The period spans the 2021 low-volatility grind, the 2022 bear trend, and whatever 2023–2026 turn
out to be, which is the minimum needed for the by-regime reporting §7 requires.

*Contingent on D10:* if Rithmic history does not reach 2021 at tick granularity, the period
starts where the data starts and every power calculation is redone.

## D4 — Splits: 60 / 15 / 25 by session count, chronological

Holdout is the final 25% as the spec requires. The remaining 75% divides 60/15 so there is a
validation block distinct from the training block, which A3 in the review said was undefined.

Split by **session count**, not calendar days — trading-day counts per quarter vary enough that
a calendar split leaves blocks with materially different power.

On the current calendar (1,378 sessions) this gives:

| block | sessions | range |
|---|---|---|
| train | 826 | 2021-01-04 .. 2024-04-17 |
| validation | 206 | 2024-04-18 .. 2025-02-11 |
| holdout | 346 | 2025-02-12 .. 2026-06-30 |

## D5 — Session scope: RTH anchors, ETH context

Anchors (`t0`) may only fall in the regular session. Feature lookback windows may reach back
through the overnight session.

The spec suggested "RTH only". Taken literally that would forbid overnight range, gap size, and
Globex value area as context — and in an auction-market framework those are among the most
informative pre-conditions available at the open. Restricting *anchors* to RTH keeps the event
population clean and liquid; restricting *lookback* would discard the context the thesis is about.

## D6 — Answer to Q2: the falsification threshold

The thesis is supported only if, on the training block, at least one pre-registered context
variable shifts the pre-declared tail statistic by a factor **≥ 1.5**, with a Benjamini–Hochberg
adjusted, session-day block-bootstrapped 95% CI excluding 1.0 — **and** the same variable
reproduces in the same direction with a shift **≥ 1.2** on the sealed holdout.

Both numbers are set before any base rate has been computed, which is the only time they can
honestly be set. 1.5 is roughly the smallest effect the sample can distinguish from noise at the
3× tail (see W1); 1.2 on the holdout allows for the shrinkage a real effect suffers out of sample
without allowing a null to pass.

If nothing clears this, the finding is that the pre-conditions are not observable at the
resolution available to us. That is a publishable negative result and the project changes shape.

## D7 — Answer to Q3: barrier scaling and the "fast" metric

ATR over **14 sessions** for the primary scaling, **60 sessions** as the pre-declared robustness
axis. Both are in config. Every headline result runs under both, and both are reported — a result
that survives only one scaling is a scaling-conditional result and gets labelled one.

Short and long are chosen to separate the two things W4 warned are conflated: a 14-session ATR
tracks the current volatility regime, so excursions scaled by it measure "large relative to now";
a 60-session ATR measures "large relative to the year". If findings only appear under the short
scaling, we have rediscovered volatility clustering.

"Fast" gets a first-class metric: **excursion velocity**, in stop-units per minute to the MFE,
alongside `time_to_MFE`. Deferred to Phase 2 with the rest of the label machinery.

## D8 — Answer to Q4: two counts, both reported

- **Hypothesis family** = one pre-registered YAML file's worth of hypotheses, scoped to a single
  trigger family. BH is applied within that.
- **Trial count** = every row in the experiment log. This is the honest denominator for
  scepticism, and it is always larger.

`ExperimentLog.family_size()` and `ExperimentLog.trial_count()` return them separately, and the
reporting layer prints both side by side. Reporting the smaller as though it were the larger is
the specific dishonesty this split exists to prevent.

## D9 — Answer to Q5, and a deviation from the spec's stack

*The validation block* may be consumed by: choosing hyperparameters, choosing thresholds, and
choosing which exploratory findings are worth pre-registering for a future study. Each touch
writes a row to the experiment log. It may **not** be used to re-select features after seeing
training results — that is what makes it validation rather than a second training set.

*Hydra is not used*, against `CLAUDE.md` §4. Hydra earns its complexity through multirun sweeps
and config composition; there is one config here and no sweeps. Its working-directory rewriting
also fights C7. A plain YAML file with a canonical-JSON SHA-256 gives the property that actually
matters — a hash in every artefact — in forty lines. Reversible if sweeps ever arrive.

*MLflow is not used*, per the spec's own "or a plain SQLite experiment log". One file, diffable,
no server, and the trial count is one `SELECT COUNT(*)`.

## D10 — Data properties still unconfirmed

Ingest is written against **assumed** Rithmic properties. §3 of the spec review lists twelve
questions; these four block the design rather than merely colouring it:

1. Granularity actually available — full depth, MBP-10, or trades plus BBO. Decides whether
   Phase 5's order-flow features exist at all.
2. Which timestamp is stored, and its precision. Only an exchange timestamp is defensible for
   `t0`; anything else needs a latency buffer added to `decision_lag_seconds`.
3. How far back the archive reaches at that granularity. Directly determines D3.
4. Whether the tape includes implied and spread-leg prints. They distort volume bars and delta,
   and they cluster around rolls.

Until these are answered the pipeline runs on synthetic data and produces no findings.

## D11 — Answer to Q7: levels across the roll

Levels-based triggers are **suppressed for the first two sessions** of a new front month, and the
roll gap is available as a mapping offset for anything that must carry a level across. Suppression
over spread-mapping because a prior-day high is a *behavioural* level — participants remember
where price was rejected in the contract they were trading — and mapping it arithmetically into a
new contract asserts a continuity that the order book does not have.

Cost: about eight sessions a year per trigger family. Sensitivity to this choice is reported.

## D12 — Answer to Q8: the deployment stays static

The report site renders committed markdown and is served as static files. No research
computation, no data, no runtime. Anything else breaches §4 and C7. Recorded in the README.

---

## What is still blocked

D10 is the live blocker. Everything below Phase 1 — trigger families, labels, base rates — needs
real data, and real data needs those four answers. Phase 1 is built and gated on synthetic data
precisely so that the answer to D10 unblocks ingest and nothing else.
