# Quantitative Research Desk

Measurement apparatus for a discretionary intraday futures desk. The research question,
constraints, phase gates, and statistical standards live in [`CLAUDE.md`](CLAUDE.md).

**Status: Phase 1 complete, gate met on synthetic data. No research result exists.**

The foundation is built and tested — session calendar, contract roll, bar construction,
the point-in-time read layer, schema contracts, the leakage detector, the experiment log.
It has never seen a real tick. Phase 2 is blocked on four unanswered questions about what
Rithmic actually provides; see
[decision D10](research/decisions/2026-08-09-scope-decisions.md).

Read in this order:

1. [Phase 1 report](research/reports/2026-08-09-phase1.md) — what was built, the gate
   result, what would falsify it.
2. [Scope decisions](research/decisions/2026-08-09-scope-decisions.md) — instrument,
   period, splits, falsification threshold, and what is still blocked.
3. [Spec review](research/reviews/2026-08-09-spec-review.md) — the methodology critique
   the rest of this is built on.

## Quick start

```sh
pip install -r requirements.txt

python -m src.pipeline build     # synthetic ticks -> bars, sessions, roll schedule
python -m src.pipeline gate      # leakage canaries + byte-identical rerun
python -m src.pipeline status    # config hash, data version, artefact checksums
pytest                           # 94 tests
```

`build` writes into `data/`, which is gitignored — market data never enters the
repository.

## The gate

`CLAUDE.md` §6 asks Phase 1 to catch one deliberately leaky feature and to reproduce
byte-identically. Both hold, and the canary requirement is deliberately stricter than
asked: one canary per leak mechanism, plus clean features that must come through
unflagged.

```
Leakage canaries (6 anchors each)
  caught   leak_centered_moving_average
  caught   leak_full_session_vwap
  caught   leak_global_zscore
  caught   leak_open_ts_filter
  caught   leak_peek_next_bar

Clean features (must not be flagged)
  passed   clean_session_vwap_so_far
  passed   clean_trailing_range
  passed   clean_trailing_return

Determinism: byte-identical
GATE: PASS
```

## Synthetic data

There is no Rithmic ingest yet. `src/ingest/synthetic.py` generates a seeded tape so the
machinery can be exercised and gated before real data exists. It has session boundaries,
a volume roll, an intraday volatility shape and jumps — and no positioning imbalance, no
catalysts, no auction structure. **Nothing produced from it is a research result.**

Three things keep that from being forgotten: `data.version` starts with `synthetic` and is
hashed into every artefact; the pipeline prints a banner on every run; and `build()` raises
`NotImplementedError` if the config claims a real snapshot.

## Layout

```
CLAUDE.md              the specification
config/                desk.yaml + the CME calendar table; hashed into every artefact
research/reports/      phase reports
research/decisions/    scope decisions and their reasoning
research/reviews/      methodology reviews
research/hypotheses/   pre-registered hypotheses (empty until Phase 4)
src/ingest/            calendar, contract roll, schemas, synthetic tape
src/bars/              time, volume, dollar bar construction
src/data/              Store and PointInTimeView — the only read path for features
src/validation/        leakage probes and the canary suite
src/experiment/        SQLite run ledger
src/reporting/         static site generator
site/                  generated HTML, committed — this is what gets served
tests/                 pytest
```

`src/events/`, `src/features/`, `src/labels/`, `src/models/` and `src/stats/` from
`CLAUDE.md` §4 do not exist yet. They arrive at their phases, not before.

## The report site

`site/` is a static rendering of the markdown in this repository, **generated locally and
committed**, so hosting is a pure file serve with no build step and no runtime.

```sh
python -m src.reporting.build_site           # regenerate site/
python -m src.reporting.build_site --check   # fail if site/ is stale
```

Regenerate and commit `site/` whenever you change `CLAUDE.md` or anything under
`research/`. `tests/test_site_up_to_date.py` fails the build if you forget. To preview:
`python -m http.server -d site`.

## Deploying to Vercel

`vercel.json` configures a static deploy: no install step, no build command, output
directory `site/`.

```sh
npx vercel        # preview deployment
npx vercel --prod # production
```

Or import the repository at vercel.com and accept the settings in `vercel.json`.

### Why the deployment is static, and stays that way

`CLAUDE.md` §4 requires the research pipeline to be local, file-based, and reproducible,
and C7 requires byte-identical output from the same config and data. Hosting a *rendering*
of committed artefacts is compatible with both: the deployment holds no data, runs no
computation, and nothing in the pipeline depends on it being up.

Moving research computation, data storage, or artefact generation onto Vercel would breach
those constraints — build environments are not reproducible, and market data does not leave
`data/`. If the desk later wants an interactive dashboard, the honest version is a local
app; the hosted surface stays read-only.

The site is served with `X-Robots-Tag: noindex` and a restrictive CSP. It is unlisted, not
private — anyone with the URL can read it, and this repository is public. Do not publish
anything here that should not be readable by whoever has the link.
