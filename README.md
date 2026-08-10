# Bayline

**The cost of waiting.** An expected-cost-of-delay engine for commercial fleet maintenance.

Every telematics product on the market predicts *failures*. None of them price the
*decision*. A fleet director does not need to be told that truck BL-0147's turbo is
degrading — the dashboard has been saying that for three weeks. What they need to know, at
07:00 on a Monday, is this:

> Given six workshop bays in Barcelona, forty penalty-bearing delivery contracts and three
> hundred trucks at different stages of wear — **which vehicles do I pull off the road this
> week, and what does waiting cost me?**

That question has an exact answer. Bayline computes it.

```
€63,207   avoidable cost over the next 28 days, against the best incumbent policy
366       jobs scheduled into 392 bay-days across three depots
1,365     jobs correctly deferred, each with the euro cost of deferring it
€11,743   what a seventh bay in Barcelona would be worth over the same 28 days
```

---

## Why this is hard, and why a risk score does not answer it

A ranked list of failure probabilities is the wrong object. Three reasons, all of which
show up in the output:

**Money and probability disagree.** A €620 battery on an automotive line-side contract
carries more expected cost than a €3,850 turbo on spot freight, because the penalty
(€4,100/day) dwarfs both repair bills. A risk-ranked queue books the turbo first. It is
confidently, expensively wrong, and the comparison in the app quantifies exactly how wrong:
€63,207 over four weeks on a 300-vehicle fleet.

**Capacity is the binding constraint, not information.** There are 1,731 jobs worth doing
and 392 bay-days to do them in. Which means the interesting decision is never "is this
vehicle at risk" but "which of these two vehicles gets Thursday". Jobs occupy *consecutive*
bay-days, so pulling one cheap two-day job forward can free the slot an expensive one needs
— a sorted list cannot see that, and an exact solver can.

**Deferral is a priced option, not a failure.** Most jobs should be deferred. The product's
job is to say what each deferral costs, so that "we are short of capacity" stops being a
complaint and becomes €11,743 — a number a finance director can sign a fit-out against.

---

## What it does

```
telemetry ──► health signals ──► calibrated risk ──► expected cost ──► MILP schedule
2.3M rows      981k rows           7 models            1,731 jobs        HiGHS, ~2s
```

**Simulate** — 300 vehicles, 548 days, hourly telemetry: 2,301,600 rows (76 MB Parquet).
Seven components with Weibull wear-out hazards (shapes 2.8–5.0), each driven by a different
duty-cycle quantity — engine hours, urban hours, brake energy, clutch engagements, thermal
stress, cold starts. Sensors saturate and carry a per-vehicle bias, because real ones do.

**Health** — daily per-component signals, computed in SQL over the hourly rows. Every window
is trailing and closed at the day in question, and exposure is measured from the last
replacement rather than the vehicle's build date, so the model cannot fall back on vehicle
age — which is the mileage-schedule logic Bayline exists to beat.

**Risk** — one gradient-boosted hazard model per component, contiguous time-ordered split,
per-component calibration. **Calibration is the ship gate, not discrimination.** Three of
the seven models cannot beat their own base rate; their predictions are replaced by the base
rate, the rows are labelled `risk_source: base_rate`, and the app says so on the scorecard.

**Economics** — the all-in cost of a failure, itemised: repair, recovery tow, driver idle
hours, contract penalty, lost revenue days. Against the same work done as a booked slot. The
difference is the risk premium, and it is what waiting actually costs.

**Schedule** — an exact mixed-integer program over bay capacity, solved with HiGHS through
`scipy.optimize.milp`:

```
variables   x[j,d] ∈ {0,1}                        job j starts on day d
objective   min Σ (c[j,d] − c_never[j]) · x[j,d]
subject to  Σ_d x[j,d] ≤ 1                        each job starts at most once
            Σ_d x[j,d] = 1                        if j is safety-critical
            Σ_{j@g} Σ_{d ≤ t < d+dur} x[j,d] ≤ bays[g]     capacity, every depot, every day
```

Safety-critical work is a hard equality, not a large cost. Pricing a brake job and letting
the optimiser weigh it against revenue is how a spreadsheet ends up recommending something
indefensible.

**App** — a static single page. No framework, no CDN, no build step, no network calls: the
whole plan is embedded as column-oriented JSON and re-priced in the browser as you move the
assumption sliders.

---

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m bayline all      # ~60 seconds end to end
python -m http.server 8000 --directory app
```

Individual stages (`simulate`, `health`, `risk`, `plan`, `app`) can be rerun alone; each
reads the previous stage's Parquet. `python -m bayline status` reports what exists.

```bash
python -m pytest          # 50 tests, ~4 seconds
```

Deploys to Vercel as a static site with no build step — `vercel.json` points at `app/`.

---

## The numbers are honest, and here is how you can tell

This is a demonstration built on a **simulated fleet**, stated on every screen. No real
vehicle data is used anywhere. What the simulator cannot fake is whether the apparatus
around it is sound, and that is what the following are for.

**Three of seven models fail their gate and ship as failures.** Turbo, injectors and brakes
cannot beat their own base rate (Brier skill −0.004, −0.052, −0.002). The config sets
`irreducible_failure_share: 0.12` — a slice of failures that wear cannot predict, which real
telemetry has and a flattering simulator would omit. The honest response to a model that
adds nothing is to say so on the scorecard, not to show a per-vehicle score anyway.

**No baseline can beat the MILP.** An exact optimum cannot lose to a greedy heuristic on the
same objective under the same constraints, so if it does, something is wrong with the
comparison rather than with the heuristic. It did, once, by €24,799: the baselines were
permitted to defer safety-critical brake work that the MILP was forced to schedule. They
were buying their advantage with brake jobs. `tests/test_optimiser.py` now asserts the
invariant, and it is the single most valuable test in the suite.

**No lookahead, tested by construction.** `tests/test_health.py` rebuilds the entire health
table from telemetry *physically truncated* at day D and asserts the row for day D is
identical to the one the full-history build produced. Reading the SQL and confirming the
windows say `ROWS BETWEEN n PRECEDING AND CURRENT ROW` is not a test — it is the person who
wrote the bug checking for it. A deliberate one-day-ahead probe confirms the harness can
actually detect a leak, because a canary that never fires proves nothing.

**The browser's arithmetic is the engine's.** The what-if sliders re-price 1,731 jobs
client-side, which means the pricing logic exists twice in two languages. `tests/test_webapp.py`
executes the *shipped* `reprice` from `app.js` under Node and asserts it reproduces the
engine to the cent. It drifted twice before that test existed — once reverse-engineering a
planned cost out of a total, once using the 21-day probability where the engine used the
28-day one, which made the penalty slider move total exposure by −0.2% when the right answer
was +30.3%. Both were invisible on screen.

**Two clean runs produce byte-identical artefacts.** Every Parquet file and the generated
page itself. This was not free: the MILP's solve time was embedded in the payload, so a
rebuild changed the artefact without changing any input, and the config hash printed beside
it quietly stopped meaning what it claims. Timing belongs in the run log.

**Everything is hashed.** `config/bayline.yaml` holds every price and every physics
assumption, and its hash (`53b6bb9d…`) is stamped on the page. Six months from now, when
somebody asks why BL-0147 was pulled off a pharma run on a Tuesday, the only defensible
answer names the prices that were in force — and the hash is insensitive to key order, so
reformatting the YAML does not look like a price change.

---

## Repository

```
config/bayline.yaml        every price and physics assumption, hashed into every artefact
src/bayline/
  simulate/                fleet and hourly telemetry generation
  health/signals.py        daily component signals, trailing windows only
  risk/survival.py         hazard models, calibration, the ship gate
  economics/costs.py       what a failure costs and what waiting costs
  schedule/optimiser.py    the MILP, the baselines, the bay shadow price
  webapp/                  static single-page build
tests/                     50 tests
app/                       generated — the deployable artefact
warehouse/                 generated — Parquet, not committed, rebuilt from the seed
```

---

## What this is not

Not a telematics platform, not a CMMS, not a work-order system. It consumes a telemetry feed
and a contract table and emits a priced weekly plan; it integrates with the systems that own
those, and it does not try to replace them.

The plan is a recommendation. Bays get double-booked, parts arrive late, and drivers call in
sick — a scheduler that cannot be overridden is a scheduler that gets ignored. Every job in
the app carries its own euro figure precisely so that a workshop manager who moves one knows
what the move costs.
