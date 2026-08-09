# Quantitative Research Desk

Measurement apparatus for a discretionary intraday futures desk. The research question,
constraints, phase gates, and statistical standards live in [`CLAUDE.md`](CLAUDE.md).

**Status: Phase 0.** No pipeline code exists. The specification requires a written review
before Phase 1 begins; that review is
[`research/reviews/2026-08-09-spec-review.md`](research/reviews/2026-08-09-spec-review.md),
and Phase 1 is blocked on the principal answering its §7.

## Layout

```
CLAUDE.md              the specification
research/reviews/      methodology reviews
research/reports/      generated research output (empty at Phase 0)
research/hypotheses/   pre-registered hypotheses (empty until Phase 4)
src/reporting/         static site generator for the committed markdown above
site/                  generated HTML, committed — this is what gets served
tests/                 pytest
```

`src/` will grow the rest of the tree in `CLAUDE.md` §4 at Phase 1. It has not yet.

## The report site

`site/` is a static rendering of the markdown artefacts in this repository. It is
**generated locally and committed**, so hosting is a pure file serve with no build step and
no runtime.

```sh
pip install -r requirements-reporting.txt
python -m src.reporting.build_site      # regenerate site/
python -m src.reporting.build_site --check   # fail if site/ is stale
pytest tests                            # includes the staleness and determinism checks
```

Regenerate and commit `site/` whenever you change `CLAUDE.md` or anything under
`research/`. `tests/test_site_up_to_date.py` fails the build if you forget.

To preview: `python -m http.server -d site`.

## Deploying to Vercel

`vercel.json` configures a static deploy: no install step, no build command, output
directory `site/`.

```sh
npx vercel        # preview deployment
npx vercel --prod # production
```

Or import the repository at vercel.com and accept the settings in `vercel.json`. Pushes to
this branch will then redeploy automatically.

### Why the deployment is static, and stays that way

`CLAUDE.md` §4 requires the research pipeline to be local, file-based, and reproducible, and
C7 requires byte-identical output from the same config and data. Hosting a *rendering* of
committed artefacts is compatible with both: the deployment holds no data, runs no
computation, and nothing in the pipeline depends on it being up.

Moving research computation, data storage, or artefact generation onto Vercel would breach
those constraints — build environments are not reproducible, and market data does not leave
`data/`, which is gitignored. If the desk later wants an interactive dashboard, the honest
version is a local app; the hosted surface stays read-only.

The site is served with `X-Robots-Tag: noindex` and a restrictive CSP. It is unlisted, not
private — anyone with the URL can read it. Do not publish anything here that should not be
readable by whoever has the link.
