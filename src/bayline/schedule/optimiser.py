"""The scheduler: an exact mixed-integer program over bay capacity.

The decision is which jobs to start on which day, subject to a fixed number of workshop
bays per depot, minimising total expected cost. That is a real optimisation problem and it
is solved here as one — a MILP handed to HiGHS through ``scipy.optimize.milp`` — rather than
as a sorted list, because a sorted list cannot see that pulling one cheap job forward frees
the bay a much more expensive job needs on Thursday.

The formulation:

    variables   x[j, d] ∈ {0, 1}   job j starts on day d
    objective   minimise  Σ_j Σ_d (c[j,d] − c_never[j]) · x[j,d]
    subject to  Σ_d x[j, d] ≤ 1                                  each job starts at most once
                Σ_d x[j, d] = 1                                   if j is safety-critical
                Σ_{j at g} Σ_{d: d ≤ t < d + dur_j} x[j,d] ≤ bays[g]    capacity, every day t

The objective is expressed as a *saving against deferral* so the constant Σ c_never drops
out; the reported total adds it back. Jobs occupy consecutive bay-days, which is why the
capacity constraint sums over the days a job started earlier is still occupying — the detail
that separates this from a knapsack and the reason a greedy heuristic leaves money behind.

Safety-critical work is a hard equality, not a large cost. Pricing a brake job and letting
the optimiser weigh it against revenue is how a spreadsheet ends up recommending something
indefensible; it is excluded from the trade instead.

Two baselines are computed with the identical objective so the headline number is a fair
comparison rather than a marketing figure: the incumbent mileage-interval policy, and a
risk-ranked queue, which is what a telematics alert list gives you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import lil_matrix

from bayline.config import Config


@dataclass
class Assignment:
    job_id: str
    start_day: int | None  # None means deferred beyond the horizon
    cost_eur: float


@dataclass
class Plan:
    assignments: list[Assignment]
    total_cost_eur: float
    scheduled: int
    deferred: int
    status: str
    bay_utilisation: list[dict] = field(default_factory=list)
    solve_seconds: float = 0.0

    def by_job(self) -> dict[str, Assignment]:
        return {a.job_id: a for a in self.assignments}


def _capacity(cfg: Config) -> dict[str, int]:
    return {depot["id"]: int(depot["bays"]) for depot in cfg.depots}


def solve(cfg: Config, jobs: list[dict], *, time_limit: float = 60.0) -> Plan:
    """Exact MILP over the planning horizon."""
    import time

    horizon = int(cfg.get("risk.planning_horizon_days"))
    bays = _capacity(cfg)
    n_jobs = len(jobs)
    if n_jobs == 0:
        return Plan([], 0.0, 0, 0, "empty")

    # Variable j*horizon + d is "job j starts on day d". A job whose duration would run past
    # the horizon cannot start that late, so those columns are fixed out rather than left to
    # the solver to discover.
    n_vars = n_jobs * horizon
    objective = np.zeros(n_vars)
    upper = np.ones(n_vars)

    for j, job in enumerate(jobs):
        duration = int(job["bay_days"])
        never = float(job["cost_if_never_eur"])
        for d in range(horizon):
            index = j * horizon + d
            if d + duration > horizon:
                upper[index] = 0.0
                continue
            objective[index] = float(job["cost_by_start_day"][d]) - never

    # Constraint block 1: each job starts at most once — exactly once if safety-critical.
    once = lil_matrix((n_jobs, n_vars))
    once_lower = np.zeros(n_jobs)
    for j, job in enumerate(jobs):
        once[j, j * horizon : (j + 1) * horizon] = 1.0
        if job["safety_critical"]:
            once_lower[j] = 1.0

    # Constraint block 2: bay capacity per depot per day, counting jobs still occupying.
    depots = sorted(bays)
    rows = len(depots) * horizon
    capacity = lil_matrix((rows, n_vars))
    capacity_upper = np.zeros(rows)
    for gi, depot in enumerate(depots):
        for t in range(horizon):
            r = gi * horizon + t
            capacity_upper[r] = bays[depot]
            for j, job in enumerate(jobs):
                if job["depot_id"] != depot:
                    continue
                duration = int(job["bay_days"])
                # Any start day d with d <= t < d + duration keeps a bay busy on day t.
                for d in range(max(0, t - duration + 1), t + 1):
                    if d + duration <= horizon:
                        capacity[r, j * horizon + d] = 1.0

    constraints = [
        LinearConstraint(once.tocsr(), lb=once_lower, ub=np.ones(n_jobs)),
        LinearConstraint(capacity.tocsr(), lb=np.zeros(rows), ub=capacity_upper),
    ]

    started = time.perf_counter()
    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(n_vars),
        bounds=Bounds(lb=np.zeros(n_vars), ub=upper),
        options={"time_limit": time_limit, "presolve": True},
    )
    elapsed = time.perf_counter() - started

    if result.x is None:
        raise RuntimeError(f"scheduler found no feasible plan: {result.message}")

    return _read_solution(cfg, jobs, result, horizon, bays, elapsed)


def _read_solution(cfg: Config, jobs, result, horizon: int, bays, elapsed: float) -> Plan:
    picked = np.asarray(result.x).reshape(len(jobs), horizon)
    assignments, total = [], 0.0
    occupancy = {depot: np.zeros(horizon) for depot in bays}

    for j, job in enumerate(jobs):
        day = int(np.argmax(picked[j])) if picked[j].max() > 0.5 else None
        if day is None:
            cost = float(job["cost_if_never_eur"])
            assignments.append(Assignment(job["job_id"], None, round(cost, 2)))
        else:
            cost = float(job["cost_by_start_day"][day])
            assignments.append(Assignment(job["job_id"], day, round(cost, 2)))
            for t in range(day, min(day + int(job["bay_days"]), horizon)):
                occupancy[job["depot_id"]][t] += 1
        total += cost

    utilisation = [
        {
            "depot_id": depot,
            "day": t,
            "bays_used": int(occupancy[depot][t]),
            "bays_total": int(bays[depot]),
            "utilisation": round(float(occupancy[depot][t]) / bays[depot], 4),
        }
        for depot in sorted(bays)
        for t in range(horizon)
    ]

    scheduled = sum(1 for a in assignments if a.start_day is not None)
    return Plan(
        assignments=assignments,
        total_cost_eur=round(total, 2),
        scheduled=scheduled,
        deferred=len(assignments) - scheduled,
        status=str(result.message).split(".")[0],
        bay_utilisation=utilisation,
        solve_seconds=round(elapsed, 3),
    )


# --------------------------------------------------------------------------- baselines


def _greedy(cfg: Config, jobs: list[dict], order: list[int], label: str) -> Plan:
    """Fill the earliest available bay-day in the given priority order.

    The shared machinery behind both baselines, scored with the *same* objective and under
    the *same* constraints as the MILP. Both matter. An earlier version let the baselines
    defer safety-critical work while the MILP was required to schedule it, and the
    baselines duly came out cheaper — they were buying their advantage with brake jobs.
    A comparison against a baseline playing by different rules is worse than no comparison,
    because it looks like evidence.
    """
    # Safety-critical work is scheduled first and unconditionally, exactly as the MILP's
    # equality constraint requires. Policy order applies only to what is left.
    order = [j for j in order if jobs[j]["safety_critical"]] + [
        j for j in order if not jobs[j]["safety_critical"]
    ]
    horizon = int(cfg.get("risk.planning_horizon_days"))
    bays = _capacity(cfg)
    occupancy = {depot: np.zeros(horizon) for depot in bays}
    assignments, total = [], 0.0
    placed: dict[str, int | None] = {}

    for j in order:
        job = jobs[j]
        duration = int(job["bay_days"])
        depot = job["depot_id"]
        chosen = None
        for d in range(horizon - duration + 1):
            window = occupancy[depot][d : d + duration]
            if float(window.max()) + 1 <= bays[depot]:
                chosen = d
                break
        if chosen is None:
            placed[job["job_id"]] = None
            total += float(job["cost_if_never_eur"])
        else:
            placed[job["job_id"]] = chosen
            occupancy[depot][chosen : chosen + duration] += 1
            total += float(job["cost_by_start_day"][chosen])

    for job in jobs:
        day = placed[job["job_id"]]
        cost = (
            float(job["cost_if_never_eur"])
            if day is None
            else float(job["cost_by_start_day"][day])
        )
        assignments.append(Assignment(job["job_id"], day, round(cost, 2)))

    scheduled = sum(1 for a in assignments if a.start_day is not None)
    return Plan(
        assignments=assignments,
        total_cost_eur=round(total, 2),
        scheduled=scheduled,
        deferred=len(assignments) - scheduled,
        status=label,
    )


def baseline_risk_ranked(cfg: Config, jobs: list[dict]) -> Plan:
    """What a telematics alert list gives you: highest failure probability first.

    The honest incumbent for anyone who has bought a predictive-maintenance product. It is
    blind to money — it will book a €620 battery ahead of a job carrying a €4,100 daily
    penalty because the battery's probability is higher.
    """
    order = sorted(range(len(jobs)), key=lambda j: -float(jobs[j]["failure_probability"]))
    return _greedy(cfg, jobs, order, "baseline: risk-ranked queue")


def baseline_mileage_interval(cfg: Config, jobs: list[dict]) -> Plan:
    """The incumbent policy: whatever has gone longest since its last service.

    No sensors, no prices. This is what most fleets actually run, and it is the number a
    buyer is comparing against whether or not they say so.
    """
    order = sorted(range(len(jobs)), key=lambda j: -int(jobs[j]["days_since_service"]))
    return _greedy(cfg, jobs, order, "baseline: mileage interval")


def bay_value(cfg: Config, jobs: list[dict], optimal: Plan, *, extra: int = 1) -> list[dict]:
    """What one more bay at each depot would be worth over the horizon.

    The shadow price, computed by re-solving with the extra capacity rather than read off
    the LP dual — the dual of a relaxation is not the value of an integer bay, and a fleet
    director asked to sign off a fit-out wants the real number.

    This is the question the whole plan begs and nobody else answers. Every telematics
    product can tell a fleet it is behind on maintenance. None of them can say what the
    seventh bay in Barcelona is worth, which is the only form in which the finance director
    can act on it.
    """
    rows = []
    for depot in cfg.depots:
        widened = cfg.with_overrides(
            **{
                "fleet.depots": [
                    {**d, "bays": d["bays"] + extra} if d["id"] == depot["id"] else dict(d)
                    for d in cfg.depots
                ]
            }
        )
        alternative = solve(widened, jobs)
        rows.append(
            {
                "depot_id": depot["id"],
                "depot_name": depot["name"],
                "bays": int(depot["bays"]),
                "extra_bays": extra,
                "value_eur": round(optimal.total_cost_eur - alternative.total_cost_eur, 2),
                "extra_jobs_scheduled": alternative.scheduled - optimal.scheduled,
            }
        )
    return sorted(rows, key=lambda r: -r["value_eur"])


def compare(cfg: Config, jobs: list[dict], optimal: Plan) -> list[dict]:
    """The optimiser against both baselines, on one objective.

    ``saving_eur`` is what the schedule is worth over four weeks. Annualising it is left to
    the reader, and the UI says so rather than multiplying by thirteen and calling it ARR.
    """
    rows = []
    for plan in (
        optimal,
        baseline_risk_ranked(cfg, jobs),
        baseline_mileage_interval(cfg, jobs),
    ):
        rows.append(
            {
                "policy": "bayline: cost-optimal MILP"
                if plan is optimal
                else plan.status,
                "total_expected_cost_eur": plan.total_cost_eur,
                "scheduled": plan.scheduled,
                "deferred": plan.deferred,
                "saving_vs_this_eur": round(plan.total_cost_eur - optimal.total_cost_eur, 2),
            }
        )
    return rows
