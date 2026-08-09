# Quantitative Research Desk

Measurement apparatus for a discretionary intraday futures desk. The research question,
constraints, phase gates, and statistical standards live in [`CLAUDE.md`](CLAUDE.md).

**Status: Phases 1–6 built and running end to end. No research result exists.**

The whole chain works — session calendar, contract roll, bars, point-in-time reads,
trigger families, triple-barrier labels, base rates, pre-registered hypothesis testing,
purged cross-validation, state clustering, GBM diagnosis, and the meta-labeling interface.
It has never seen a real tick. Every number in `research/reports/` came from a seeded
random walk and is banner-marked as such.

The blocker is not code. It is four unanswered questions about what Rithmic actually
provides — [decision D10](research/decisions/2026-08-09-scope-decisions.md).

Read in this order:

1. [Phase 1 report](research/reports/2026-08-09-phase1.md) — foundation and the leakage gate.
2. [Scope decisions](research/decisions/2026-08-09-scope-decisions.md) — instrument, period,
   splits, falsification threshold, and what is still blocked.
3. [Phase 2](research/reports/2026-08-09-phase2.md) → [3](research/reports/2026-08-09-phase3.md)
   → [4](research/reports/2026-08-09-phase4.md) → [5](research/reports/2026-08-09-phase5.md)
   → [6](research/reports/2026-08-09-phase6.md).
4. [Spec review](research/reviews/2026-08-09-spec-review.md) — the methodology critique the
   rest is built on.

## Quick start

```sh
pip install -r requirements.txt

python -m src.pipeline all        # build, gate, then every phase — about two minutes
python -m src.pipeline gate       # leakage canaries + byte-identical rerun
python -m src.pipeline phase3     # any single phase
python -m src.pipeline status     # config hash, data version, artefact checksums
pytest                            # 157 tests
```

`build` writes into `data/`, which is gitignored — market data never enters the repository.

Phases 3 onward read the **training block only**. The holdout is opened once, at the end,
by a human who has decided to open it — never as a side effect of running the pipeline.

## What the synthetic run says

On a tape with no structure in it, the apparatus finds nothing:

| Phase | Result |
|---|---|
| 1 | Gate PASS — 5 canaries caught, 12 clean features unflagged, builds byte-identical |
| 2 | 292 labelled events across 8 directional families |
| 3 | Base rates with intervals spanning a factor of several at 3× and beyond |
| 4 | **0 of 6 pre-registered hypotheses supported**; 0 survived BH correction |
| 5 | Clusters **unstable** (ARI 0.16); out-of-sample IC +0.09, R² near zero |
| 6 | Unfed — no trade log |

That is the correct answer for a random walk, and it is the strongest evidence available
that the machinery is not manufacturing findings. A pipeline that produced a taxonomy from
this tape would be broken.

## The Phase 1 gate

`CLAUDE.md` §6 asks for one deliberately leaky feature to be caught and for builds to
reproduce byte-identically. Both hold, and the bar is set higher than asked: one canary per
leak mechanism, clean features that must come through unflagged, and the **production
feature set audited by the same probes**.

```
Leakage canaries (6 anchors each)
  caught   leak_centered_moving_average     centred window
  caught   leak_full_session_vwap           whole-session aggregate
  caught   leak_global_zscore               full-period normaliser
  caught   leak_open_ts_filter              open_ts instead of close_ts
  caught   leak_peek_next_bar               one bar ahead

Clean and production features (must not be flagged)
  passed   × 12   (3 clean canaries + all 9 context features)

Determinism (build): byte-identical
Determinism (events, ATR, labels): identical
GATE: PASS
```

Two probes, and neither dominates. Truncation removes post-cutoff rows; perturbation
replaces post-cutoff numbers. A feature reading a future *timestamp* walks past
perturbation; a future value entering through a clamp walks past truncation.
`tests/test_gate.py` pins one example of each direction so neither can be dropped as
redundant.

## Synthetic data

`src/ingest/synthetic.py` generates a seeded tape so the machinery can be exercised and
gated before real data exists. It has session boundaries, a volume roll, an intraday
volatility shape and jumps — and no positioning imbalance, no catalysts, no auction
structure, which are precisely the things the research question is about.

Three things keep that from being forgotten: `data.version` starts with `synthetic` and is
hashed into every artefact; every report carries a banner the generator will not omit; and
`build()` raises `NotImplementedError` if the config claims a real snapshot.

## Layout

```
CLAUDE.md              the specification
config/                desk.yaml + the CME calendar table; hashed into every artefact
research/reports/      generated phase reports — never hand-edited
research/decisions/    scope decisions and their reasoning
research/reviews/      methodology reviews
research/hypotheses/   pre-registered hypotheses, immutable once committed
src/ingest/            calendar, contract roll, schemas, synthetic tape
src/bars/              time, volume, dollar bar construction
src/data/              Store and PointInTimeView — the only read path for features
src/events/            trigger families
src/features/          context features, all through the view
src/labels/            trailing ATR, triple barrier, excursion metrics
src/validation/        leakage probes, canaries, PurgedKFold
src/stats/             block bootstrap, base rates, hypothesis testing
src/models/            clustering, GBM diagnosis, meta-labeling interface
src/reporting/         report generation and the static site
site/                  generated HTML, committed — this is what gets served
tests/                 pytest
```

## Where this departs from the spec

Each is argued in the decisions record or a phase report, not done quietly:

- **Block bootstrap by session-day, everywhere.** Events cluster within a session; i.i.d.
  resampling makes every interval too narrow and lets findings survive BH that should not.
  This is the single most consequential addition here.
- **Hydra is not used** (D9). One config, no sweeps, and its directory rewriting fights C7.
- **The Phase 1 gate is stricter** than §6 asks — a canary per mechanism, plus the real
  feature set.
- **Barriers scale to the label horizon**, not the session. A 1×-session-ATR stop over a
  120-minute window is unreachable, and every event then times out carrying no information.
- **Phase 5's headline metric is not R²** — it is unstable on a heavy-tailed target. Spearman
  IC and pinball loss lead; R² is reported as a secondary line.
- **The decision lag is charged once**, on the entry fill. The information window runs to
  `t0` because the bar closing at `t0` is what made the trigger observable.

## The report site

`site/` is a static rendering of the markdown in this repository, **generated locally and
committed**, so hosting is a pure file serve with no build step and no runtime.

```sh
python -m src.reporting.build_site           # regenerate site/
python -m src.reporting.build_site --check   # fail if site/ is stale
```

Regenerate and commit `site/` whenever the markdown changes; `tests/test_site_up_to_date.py`
fails the build if you forget. To preview: `python -m http.server -d site`.

## Deploying to Vercel

`vercel.json` configures a static deploy: no install step, no build command, output
directory `site/`.

```sh
npx vercel        # preview deployment
npx vercel --prod # production
```

Or import the repository at vercel.com and accept the settings in `vercel.json`.

### Why the deployment is static, and stays that way

`CLAUDE.md` §4 requires the research pipeline to be local, file-based, and reproducible, and
C7 requires byte-identical output from the same config and data. Hosting a *rendering* of
committed artefacts is compatible with both: the deployment holds no data, runs no
computation, and nothing in the pipeline depends on it being up.

Moving research computation, data storage, or artefact generation onto Vercel would breach
those constraints. If the desk later wants an interactive dashboard, the honest version is a
local app; the hosted surface stays read-only.

The site is served with `X-Robots-Tag: noindex` and a restrictive CSP. It is unlisted, not
private — anyone with the URL can read it, and this repository is public.
