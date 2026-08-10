/* Bayline — client.
 *
 * Vanilla ES modules, no dependencies. The plan arrives embedded in the page, so the whole
 * application is one request and works offline in a depot with bad signal.
 *
 * The what-if controls are not a mock. They re-run the same cost arithmetic the scheduler
 * used, on the same itemised breakdown, in the browser: change the penalty assumption and
 * every expected saving is recomputed from its parts. What they cannot do is re-solve the
 * MILP — that needs the engine — so the app is explicit that it is re-pricing the existing
 * schedule rather than re-planning, and says how much the ranking moved.
 */

const raw = JSON.parse(document.getElementById("plan-data").textContent);
const COL = Object.fromEntries(raw.columns.map((name, i) => [name, i]));
const META = raw.meta;

/** One job, as an object view over its row array. Cheap, and keeps call sites readable. */
const jobs = raw.jobs.map((row) => {
  const job = {};
  raw.columns.forEach((name, i) => (job[name] = row[i]));
  return job;
});

const state = {
  depot: "",
  component: "",
  contract: "",
  search: "",
  scheduledOnly: false,
  sort: { key: "expected_saving_eur", dir: -1 },
  selected: null,
  whatif: { penalty: 1, revenue: 1, recovery: 1, risk: 1 },
};

// ---------------------------------------------------------------- formatting

const eur = (v) =>
  "€" + Math.round(v).toLocaleString("en-GB", { maximumFractionDigits: 0 });
const eur2 = (v) => "€" + v.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (v, digits = 1) => (v * 100).toFixed(digits) + "%";

const startDate = new Date(META.start_date + "T00:00:00Z");
function dayLabel(day) {
  const d = new Date(startDate.getTime() + day * 86400000);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
}

// ---------------------------------------------------------------- what-if pricing

/**
 * Re-price one job under the current assumptions.
 *
 * The premium is the difference between an unplanned failure and the same work booked. Only
 * three of its five parts move with the sliders — the repair bill and the driver's depot
 * hour do not depend on the customer's contract terms — so the recomputation touches
 * exactly those and leaves the rest alone.
 */
function reprice(job) {
  const w = state.whatif;
  const penalty = job.penalty_eur * w.penalty;
  const revenueLoss = job.revenue_loss_eur * w.revenue;
  const recovery = job.recovery_eur * w.recovery;
  const unplanned =
    job.repair_eur + recovery + job.driver_eur + penalty + revenueLoss;
  // The planned branch loses revenue too, for the bay days it occupies, and that scales
  // with the same revenue assumption. Its parts arrive itemised from the engine rather than
  // being reverse-engineered from a total, so at 100% this reproduces the engine's own
  // number exactly — which `tests/test_webapp.py` checks.
  const planned =
    job.planned_repair_eur + job.planned_driver_eur + job.planned_revenue_eur * w.revenue;
  const premium = unplanned - planned;
  // Money uses the planning-horizon probability; the label-horizon one is for display.
  const probability = Math.min(job.horizon_probability * w.risk, 0.999);
  return {
    unplanned,
    planned,
    premium,
    probability,
    shown_probability: Math.min(job.failure_probability * w.risk, 0.999),
    saving: probability * premium,
  };
}

function priced() {
  return jobs.map((job) => ({ ...job, ...reprice(job) }));
}

// ---------------------------------------------------------------- filtering

function visible(rows) {
  const needle = state.search.trim().toLowerCase();
  return rows.filter((j) => {
    if (state.depot && j.depot_id !== state.depot) return false;
    if (state.component && j.component_id !== state.component) return false;
    if (state.contract && j.contract_id !== state.contract) return false;
    if (state.scheduledOnly && j.start_day === null) return false;
    if (needle) {
      const hay = (j.vehicle_id + " " + j.registration).toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
}

function sorted(rows) {
  const { key, dir } = state.sort;
  return [...rows].sort((a, b) => {
    let x = a[key], y = b[key];
    if (key === "start_day") {
      // Deferred jobs sort last whichever way the column is pointing: "no date" is not a
      // date, and burying scheduled work under 1,300 blanks helps nobody.
      x = x === null ? Infinity : x;
      y = y === null ? Infinity : y;
      return (x - y) * (dir === -1 ? -1 : 1);
    }
    if (typeof x === "string") return x.localeCompare(y) * dir;
    return (x - y) * dir;
  });
}

// ---------------------------------------------------------------- table

const HEADERS = [
  { key: "start_day", label: "When", num: false },
  { key: "vehicle_id", label: "Vehicle", num: false },
  { key: "component_name", label: "Component", num: false, optional: true },
  { key: "depot_id", label: "Depot", num: false, optional: true },
  { key: "probability", label: "P(fail)", num: true },
  { key: "premium", label: "If it fails", num: true, optional: true },
  { key: "saving", label: "Saved by acting", num: true },
];

function renderHead() {
  const tr = document.getElementById("plan-head");
  tr.innerHTML = "";
  for (const h of HEADERS) {
    const th = document.createElement("th");
    th.textContent = h.label;
    th.className = [h.num ? "num" : "", "sortable", h.optional ? "col-optional" : ""]
      .filter(Boolean)
      .join(" ");
    th.tabIndex = 0;
    th.setAttribute("role", "columnheader");
    if (state.sort.key === h.key) {
      th.setAttribute("aria-sort", state.sort.dir === -1 ? "descending" : "ascending");
    }
    const activate = () => {
      state.sort =
        state.sort.key === h.key
          ? { key: h.key, dir: -state.sort.dir }
          : { key: h.key, dir: h.num ? -1 : 1 };
      render();
    };
    th.addEventListener("click", activate);
    th.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
    });
    tr.appendChild(th);
  }
}

function whenCell(job) {
  if (job.start_day === null) {
    return `<span class="pill pill-defer">defer</span>`;
  }
  const cls = job.start_day <= 2 ? "pill-now" : job.start_day <= 9 ? "pill-soon" : "pill-later";
  return `<span class="pill ${cls}">${dayLabel(job.start_day)}</span>`;
}

function renderTable(rows) {
  const body = document.getElementById("plan-body");
  const empty = document.getElementById("plan-empty");
  body.innerHTML = "";
  empty.hidden = rows.length > 0;

  // Only the first 400 rows are put in the DOM. Beyond that the browser spends longer on
  // layout than anyone spends reading, and the filters are the way to reach the rest.
  const shown = rows.slice(0, 400);
  const fragment = document.createDocumentFragment();

  for (const job of shown) {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    if (state.selected === job.job_id) tr.setAttribute("aria-selected", "true");
    const safety = job.safety_critical
      ? ` <span class="pill pill-safety" title="Cannot be deferred at any price">safety</span>`
      : "";
    tr.innerHTML = `
      <td>${whenCell(job)}${safety}</td>
      <td>${job.vehicle_id}<br><span class="reg">${job.registration}</span></td>
      <td class="col-optional">${job.component_name}</td>
      <td class="col-optional">${job.depot_id.replace("DEP-", "")}</td>
      <td class="num">${pct(job.shown_probability)}</td>
      <td class="num col-optional">${eur(job.premium)}</td>
      <td class="num"><strong>${eur(job.saving)}</strong></td>`;
    const open = () => openDrawer(job);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); open(); }
    });
    fragment.appendChild(tr);
  }
  body.appendChild(fragment);

  document.getElementById("plan-count").textContent =
    `${rows.length.toLocaleString("en-GB")} jobs` +
    (rows.length > shown.length ? ` · showing the top ${shown.length}` : "");
}

// ---------------------------------------------------------------- headline

function renderHeadline(rows) {
  // The engine's own figure, unscaled, whenever the assumptions are untouched. Scaling it
  // by a filtered subtotal would make the headline move when someone picks a depot from a
  // dropdown, which is not what "avoidable cost" means.
  const saving = raw.comparison[1] ? raw.comparison[1].saving_vs_this_eur : 0;
  const w = state.whatif;
  const moved = w.penalty !== 1 || w.revenue !== 1 || w.recovery !== 1 || w.risk !== 1;
  const scale = moved
    ? totalSaving(jobs.map((j) => ({ ...j, ...reprice(j) }))) / Math.max(baseTotalSaving(), 1e-9)
    : 1;

  document.getElementById("headline").textContent =
    `${eur(saving * scale)} of avoidable cost over the next ${META.horizon_days} days` +
    (moved ? " · re-priced" : "");
  document.getElementById("headline-sub").textContent =
    `Against a risk-ranked alert list, on ${META.vehicles} vehicles across ` +
    `${Object.keys(META.depots).length} depots. The plan below is what produces it: ` +
    `${raw.totals.scheduled} jobs booked into the bays you have, ${raw.totals.deferred} ` +
    `priced and deliberately left running.`;
}

function totalSaving(rows) {
  return rows.reduce((sum, j) => sum + j.saving, 0);
}
let cachedBase = null;
function baseTotalSaving() {
  if (cachedBase === null) {
    cachedBase = jobs.reduce((sum, j) => sum + j.expected_saving_eur, 0);
  }
  return cachedBase;
}

function renderKpis(rows) {
  const scheduled = rows.filter((j) => j.start_day !== null);
  const atRisk = rows.filter((j) => j.shown_probability >= 0.15).length;
  const modelled = raw.scorecards.filter((s) => s.passes_gate).length;
  const bays = raw.bay_utilisation;
  const used = bays.reduce((s, b) => s + b.bays_used, 0);
  const capacity = bays.reduce((s, b) => s + b.bays_total, 0);

  const tiles = [
    {
      label: "Exposure removed",
      value: eur(totalSaving(scheduled)),
      note: `expected failure cost taken off the road by the ${scheduled.length} booked jobs`,
      cls: "kpi-hero",
    },
    {
      label: "Exposure carried",
      value: eur(totalSaving(rows) - totalSaving(scheduled)),
      note: "priced and knowingly left running — the bays are full",
      cls: "kpi-risk",
    },
    {
      label: "Bay utilisation",
      value: pct(used / capacity, 0),
      note: `${used} of ${capacity} bay-days booked over ${META.horizon_days} days`,
    },
    {
      label: "Vehicles above 15%",
      value: String(atRisk),
      note: "component-failure probability inside the next " + META.label_horizon_days + " days",
    },
    {
      label: "Components modelled",
      value: `${modelled} of ${raw.scorecards.length}`,
      note: "the rest fall back to their base rate and are labelled, not hidden",
    },
  ];

  document.getElementById("kpis").innerHTML = tiles
    .map(
      (t) => `<div class="kpi ${t.cls || ""}">
        <p class="kpi-label">${t.label}</p>
        <p class="kpi-value tnum">${t.value}</p>
        <p class="kpi-note">${t.note}</p></div>`
    )
    .join("");
}

// ---------------------------------------------------------------- charts

function svg(width, height, body, title) {
  return `<svg class="viz" viewBox="0 0 ${width} ${height}" role="img" aria-label="${title}">${body}</svg>`;
}

function renderHeatmap() {
  const depots = [...new Set(raw.bay_utilisation.map((b) => b.depot_id))].sort();
  const days = META.horizon_days;
  const cell = 17, gap = 2, left = 52, top = 22;
  const width = left + days * (cell + gap);
  const height = top + depots.length * (cell + gap) + 30;

  let body = "";
  depots.forEach((depot, r) => {
    body += `<text class="viz-label" x="${left - 8}" y="${top + r * (cell + gap) + 12}" text-anchor="end">${depot.replace("DEP-", "")}</text>`;
    for (let t = 0; t < days; t++) {
      const entry = raw.bay_utilisation.find((b) => b.depot_id === depot && b.day === t);
      const u = entry ? entry.utilisation : 0;
      // One hue, light to dark. Utilisation is a magnitude, not an identity.
      const alpha = u === 0 ? 0.06 : 0.18 + 0.82 * Math.min(u, 1);
      body += `<g class="viz-hit"><title>${depot} · ${dayLabel(t)} · ${entry ? entry.bays_used : 0} of ${entry ? entry.bays_total : 0} bays</title>
        <rect class="viz-cell" x="${left + t * (cell + gap)}" y="${top + r * (cell + gap)}"
          width="${cell}" height="${cell}" rx="3"
          fill="var(--series)" fill-opacity="${alpha.toFixed(3)}"/></g>`;
    }
  });
  for (let t = 0; t < days; t += 7) {
    body += `<text class="viz-tick" x="${left + t * (cell + gap)}" y="${top - 8}">${dayLabel(t).split(" ").slice(1).join(" ")}</text>`;
  }

  const values = (raw.bay_value || [])
    .map(
      (v) => `<tr><td>${v.depot_name}</td>
        <td class="num">${v.bays} → ${v.bays + v.extra_bays}</td>
        <td class="num"><strong>${eur(v.value_eur)}</strong></td>
        <td class="num">${v.extra_jobs_scheduled >= 0 ? "+" : ""}${v.extra_jobs_scheduled}</td></tr>`
    )
    .join("");

  document.getElementById("heatmap").innerHTML =
    svg(width, height, body, "Bay utilisation by depot and day") +
    `<p class="caption">Darker is fuller. Every row saturates, which is itself the finding:
     ${raw.totals.candidates.toLocaleString("en-GB")} jobs are worth doing and there are only
     ${raw.bay_utilisation.reduce((s, b) => s + b.bays_total, 0)} bay-days to do them in. The
     jobs marked <em>defer</em> are not low risk — they are the ones that lost the bay.</p>
     <h3 style="font-size:0.8125rem;margin:1.1rem 0 0.3rem">What one more bay is worth</h3>
     <div class="scroll-x"><table class="ledger"><thead><tr>
       <th style="text-align:left">Depot</th><th class="num">Bays</th>
       <th class="num">Value over ${META.horizon_days}d</th><th class="num">Extra jobs</th>
     </tr></thead><tbody>${values}</tbody></table></div>
     <p class="caption">Computed by re-solving the whole schedule with the extra bay, not
     read off a linear relaxation — the dual of a relaxed problem is not the value of a real
     bay, and this number is meant to survive a finance meeting.</p>`;
}

function renderComparison() {
  const rows = raw.comparison;
  const max = Math.max(...rows.map((r) => r.total_expected_cost_eur));
  const w = 520, rowH = 34, left = 8;
  const height = rows.length * rowH + 34;
  let body = "";
  rows.forEach((r, i) => {
    const y = i * rowH + 8;
    const barW = (r.total_expected_cost_eur / max) * (w - left - 130);
    const best = i === 0;
    body += `<g class="viz-hit"><title>${r.policy}: ${eur(r.total_expected_cost_eur)} expected cost, ${r.scheduled} scheduled</title>
      <rect class="${best ? "viz-mark" : "viz-mark-muted"}" x="${left}" y="${y}" width="${Math.max(barW, 2)}" height="16" rx="4"/></g>
      <text class="viz-value" x="${left + barW + 8}" y="${y + 13}">${eur(r.total_expected_cost_eur)}</text>
      <text class="viz-label" x="${left}" y="${y + 30}">${r.policy}${best ? "" : ` · ${eur(r.saving_vs_this_eur)} worse`}</text>`;
  });
  document.getElementById("comparison").innerHTML =
    svg(w, height + 12, body, "Total expected cost by scheduling policy") +
    `<p class="caption">Same objective, same bays, same safety rule for all three — a
     baseline allowed to defer brake work would win on price and be unbuildable. The
     risk-ranked queue is what a predictive-maintenance alert list gives you; it is blind to
     what a failure costs, so it books cheap components ahead of expensive contracts.</p>`;
}

function renderScorecards() {
  const rows = raw.scorecards;
  if (!rows.length) {
    document.getElementById("scorecards").innerHTML = `<p class="caption">No scorecards in this build.</p>`;
    return;
  }
  const head = `<tr><th>Component</th><th class="num">Base rate</th><th class="num">AUC</th>
    <th class="num">Calibration error</th><th class="num">Brier skill</th>
    <th>Calibrator</th><th>Used for</th></tr>`;
  const body = rows
    .map((s) => {
      const pass = s.passes_gate;
      return `<tr>
        <td>${META.components[s.component_id] || s.component_id}</td>
        <td class="num">${pct(s.base_rate, 2)}</td>
        <td class="num">${s.auc.toFixed(3)}</td>
        <td class="num">${pct(s.calibration_error, 1)}</td>
        <td class="num">${s.brier_skill >= 0 ? "+" : ""}${s.brier_skill.toFixed(3)}</td>
        <td>${s.calibrator} <span class="reg">(${s.calibration_positives} pos.)</span></td>
        <td>${pass
          ? `<span class="pill pill-later" style="color:var(--good)">model</span>`
          : `<span class="pill pill-later">base rate</span>`}</td>
      </tr>`;
    })
    .join("");
  document.getElementById("scorecards").innerHTML =
    `<div class="table-wrap" style="max-height:none"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>
     <p class="caption"><strong>Calibration is the gate, not accuracy.</strong> Every euro on
     this page is a probability multiplied by a cost, so a model that says 40% when the truth
     is 8% does not produce a slightly wrong plan — it produces a confidently wrong budget.
     Three components cannot beat their own base rate at this horizon, so they use it and the
     table says so. Publishing a per-vehicle score for all seven regardless is the reason
     nobody trusts these products.</p>`;
}

// ---------------------------------------------------------------- drawer

function waitCurve(job) {
  const days = META.horizon_days;
  const w = 430, h = 150, left = 46, right = 12, top = 10, bottom = 26;
  const hazard = 1 - Math.pow(1 - Math.min(job.shown_probability, 0.999), 1 / META.label_horizon_days);
  const points = [];
  for (let d = 0; d <= days; d++) {
    const p = 1 - Math.pow(1 - hazard, d);
    points.push([d, p * job.premium]);
  }
  const maxCost = points[points.length - 1][1] || 1;
  const x = (d) => left + (d / days) * (w - left - right);
  const y = (v) => h - bottom - (v / maxCost) * (h - top - bottom);

  let body = "";
  for (const frac of [0, 0.5, 1]) {
    const yy = y(maxCost * frac);
    body += `<line class="viz-grid" x1="${left}" y1="${yy}" x2="${w - right}" y2="${yy}"/>
      <text class="viz-tick" x="${left - 6}" y="${yy + 3.5}" text-anchor="end">${eur(maxCost * frac)}</text>`;
  }
  body += `<path class="viz-line" d="${points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}"/>`;
  if (job.start_day !== null) {
    const cx = x(job.start_day);
    body += `<line class="viz-rule" x1="${cx}" y1="${top}" x2="${cx}" y2="${h - bottom}"/>
      <circle class="viz-dot" cx="${cx}" cy="${y(points[job.start_day][1])}" r="4.5"/>
      <text class="viz-label" x="${Math.min(cx + 6, w - 120)}" y="${top + 10}">booked ${dayLabel(job.start_day)}</text>`;
  }
  body += `<line class="viz-axis" x1="${left}" y1="${h - bottom}" x2="${w - right}" y2="${h - bottom}"/>
    <text class="viz-tick" x="${left}" y="${h - 8}">today</text>
    <text class="viz-tick" x="${w - right}" y="${h - 8}" text-anchor="end">+${days}d</text>`;
  return svg(w, h, body, "Expected cost of waiting");
}

function openDrawer(job) {
  state.selected = job.job_id;
  const drawer = document.getElementById("drawer");
  document.getElementById("drawer-title").textContent =
    `${job.vehicle_id} · ${job.component_name}`;
  document.getElementById("drawer-sub").textContent =
    `${job.registration} · ${META.depots[job.depot_id]} · ${META.contracts[job.contract_id]}`;

  const recommendation =
    job.start_day === null
      ? `<strong>Leave it running.</strong> Every bay in ${META.depots[job.depot_id]} is
         committed to work with a higher expected cost. Carrying this one costs
         ${eur(job.saving)} in expected failure cost over ${META.horizon_days} days — that is
         the price of the capacity you do not have, not an oversight.`
      : `<strong>Book it for ${dayLabel(job.start_day)}.</strong> Waiting the full
         ${META.horizon_days} days instead carries ${eur(job.saving)} of expected failure
         cost; acting on the booked day carries
         ${eur(job.saving * (job.start_day / META.horizon_days))}. The difference is what the
         slot is worth.`;

  const source =
    job.risk_source === "model"
      ? `a calibrated hazard model`
      : `this component's fleet base rate — its model does not beat it, so it is not used`;

  const ledger = [
    ["Repair", job.repair_eur],
    ["Recovery and tow", job.recovery_eur * state.whatif.recovery],
    ["Driver hours", job.driver_eur],
    ["Contract penalty", job.penalty_eur * state.whatif.penalty],
    ["Lost revenue", job.revenue_loss_eur * state.whatif.revenue],
  ];

  document.getElementById("drawer-body").innerHTML = `
    <p class="recommend">${recommendation}</p>

    <p class="caption" style="margin-top:0">
      Failure probability inside ${META.label_horizon_days} days is
      <strong>${pct(job.shown_probability)}</strong>, from ${source};
      over the ${META.horizon_days}-day planning window, ${pct(job.probability)}.
      ${job.days_since_service} days since this component was last replaced.
      ${job.safety_critical ? " This is safety-critical work and is never traded against cost." : ""}
    </p>

    <h3 style="font-size:0.8125rem;margin:1.2rem 0 0.2rem">Expected cost of waiting</h3>
    ${waitCurve(job)}
    <p class="caption">Cumulative expected cost of leaving it in service, day by day. The
    curve is the failure probability spread over the horizon at a constant hazard.</p>

    <table class="ledger">
      <caption>If it fails on the road</caption>
      <tbody>
        ${ledger.map(([k, v]) => `<tr><td>${k}</td><td>${eur2(v)}</td></tr>`).join("")}
        <tr class="total"><td>Unplanned total</td><td>${eur2(job.unplanned)}</td></tr>
      </tbody>
    </table>

    <table class="ledger">
      <caption>If it is booked into a bay</caption>
      <tbody>
        <tr><td>Planned repair</td><td>${eur2(job.planned_repair_eur)}</td></tr>
        <tr><td>Driver hours</td><td>${eur2(job.planned_driver_eur)}</td></tr>
        <tr><td>Lost revenue, ${job.bay_days} bay-day${job.bay_days > 1 ? "s" : ""}</td><td>${eur2(job.planned_revenue_eur * state.whatif.revenue)}</td></tr>
        <tr class="total"><td>Planned total</td><td>${eur2(job.planned)}</td></tr>
      </tbody>
    </table>

    <table class="ledger">
      <caption>The decision</caption>
      <tbody>
        <tr><td>Cost of being wrong</td><td>${eur2(job.premium)}</td></tr>
        <tr><td>Probability over ${META.horizon_days} days</td><td>${pct(job.probability)}</td></tr>
        <tr class="total"><td>Expected cost of waiting</td><td>${eur2(job.saving)}</td></tr>
      </tbody>
    </table>

    <p class="caption">${job.immobilising
      ? "An immobilising failure: the vehicle stops where it is and needs recovery."
      : "A derate rather than a strand — the vehicle limps to a depot, so no recovery cost."}
      Bay time ${job.bay_days} day${job.bay_days > 1 ? "s" : ""}.</p>`;

  drawer.dataset.open = "true";
  drawer.setAttribute("aria-hidden", "false");
  document.getElementById("scrim").hidden = false;
  document.getElementById("drawer-close").focus();
  render();
}

function closeDrawer() {
  const drawer = document.getElementById("drawer");
  drawer.dataset.open = "false";
  drawer.setAttribute("aria-hidden", "true");
  document.getElementById("scrim").hidden = true;
  state.selected = null;
  render();
}

// ---------------------------------------------------------------- what-if

function renderWhatIf(rows) {
  const w = state.whatif;
  const changed =
    w.penalty !== 1 || w.revenue !== 1 || w.recovery !== 1 || w.risk !== 1;
  const now = totalSaving(rows);
  const base = baseTotalSaving();
  const readout = document.getElementById("whatif-readout");

  if (!changed) {
    readout.innerHTML = `At the configured prices the fleet is carrying
      <strong>${eur(base)}</strong> of expected failure cost across all
      ${jobs.length.toLocaleString("en-GB")} priced jobs.`;
    return;
  }
  const delta = now - base;
  const reordered = rankChurn(rows);
  readout.innerHTML = `Carried exposure moves to <strong>${eur(now)}</strong>
    (${delta >= 0 ? "+" : ""}${eur(delta)}, ${((delta / base) * 100).toFixed(1)}%).
    <strong>${reordered}</strong> of the top 50 jobs change places, which is the number that
    matters: if the ranking barely moves, the plan is robust to this assumption being wrong.
    Re-pricing only — the bay schedule itself is fixed until the engine re-solves.`;
}

/** How many of the top 50 jobs the re-pricing actually reorders. */
function rankChurn(rows) {
  const byBase = [...rows].sort((a, b) => b.expected_saving_eur - a.expected_saving_eur).slice(0, 50);
  const byNow = [...rows].sort((a, b) => b.saving - a.saving).slice(0, 50);
  const baseIds = new Set(byBase.map((j) => j.job_id));
  return byNow.filter((j, i) => !baseIds.has(j.job_id) || byBase[i].job_id !== j.job_id).length;
}

// ---------------------------------------------------------------- wiring

function render() {
  const rows = sorted(visible(priced()));
  renderHead();
  renderTable(rows);
  renderHeadline(rows);
  renderKpis(rows);
  renderWhatIf(rows);
}

function fillSelect(id, entries) {
  const select = document.getElementById(id);
  for (const [value, label] of Object.entries(entries)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }
}

function init() {
  fillSelect("f-depot", META.depots);
  fillSelect("f-component", META.components);
  fillSelect("f-contract", META.contracts);

  const bind = (id, key) =>
    document.getElementById(id).addEventListener("input", (e) => {
      state[key] = e.target.value;
      render();
    });
  bind("f-depot", "depot");
  bind("f-component", "component");
  bind("f-contract", "contract");
  bind("f-search", "search");

  document.getElementById("only-scheduled").addEventListener("click", (e) => {
    state.scheduledOnly = !state.scheduledOnly;
    e.currentTarget.setAttribute("aria-pressed", String(state.scheduledOnly));
    render();
  });

  const sliders = [
    ["s-penalty", "o-penalty", "penalty"],
    ["s-revenue", "o-revenue", "revenue"],
    ["s-recovery", "o-recovery", "recovery"],
    ["s-risk", "o-risk", "risk"],
  ];
  for (const [input, output, key] of sliders) {
    document.getElementById(input).addEventListener("input", (e) => {
      state.whatif[key] = Number(e.target.value) / 100;
      document.getElementById(output).textContent = e.target.value + "%";
      render();
    });
  }
  document.getElementById("reset-whatif").addEventListener("click", () => {
    for (const [input, output, key] of sliders) {
      const el = document.getElementById(input);
      el.value = 100;
      document.getElementById(output).textContent = "100%";
      state.whatif[key] = 1;
    }
    render();
  });

  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.getElementById("drawer").dataset.open === "true") {
      closeDrawer();
    }
  });

  document.getElementById("theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = getComputedStyle(root).colorScheme.includes("dark");
    root.dataset.theme = dark ? "light" : "dark";
  });

  renderHeatmap();
  renderComparison();
  renderScorecards();
  render();
}

init();
