"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const fmt1 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });
const money = new Intl.NumberFormat("en-NG", { style: "currency", currency: "NGN", maximumFractionDigits: 0 });
const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`);
const signedPct = (v) => (v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const STATUS = { forecast: "Forecast", insufficient_history: "Insufficient data", no_recent_sales: "No recent sales" };

const state = { datasets: [], dataset: null, run: null, horizon: 30, stage: "test", selectedKey: null, poll: null };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const body = await res.json(); detail = body.detail || detail; } catch (_) { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? null : res.json();
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { el.hidden = true; }, 3500);
}

/* ---------------- datasets ---------------- */
async function loadDatasets(selectId) {
  state.datasets = await api("/api/datasets");
  const sel = $("#datasetSelect");
  sel.innerHTML = state.datasets.length
    ? state.datasets.map((d) => `<option value="${d.id}">${d.is_synthetic ? "[Synthetic] " : ""}${esc(d.name)} · ${new Date(d.created_at).toLocaleDateString()}</option>`).join("")
    : `<option value="">No datasets yet</option>`;
  const wanted = selectId || location.hash.slice(1) || state.datasets[0]?.id;
  if (wanted && state.datasets.some((d) => d.id === wanted)) {
    sel.value = wanted;
    await selectDataset(wanted);
  } else {
    showEmpty();
  }
}

function showEmpty() {
  $("#emptyState").hidden = false;
  $("#datasetView").hidden = true;
  $("#synthBanner").hidden = true;
}

async function selectDataset(id) {
  stopPolling();
  state.dataset = await api(`/api/datasets/${id}`);
  state.run = null;
  state.selectedKey = null;
  history.replaceState(null, "", `#${id}`);
  $("#emptyState").hidden = true;
  $("#datasetView").hidden = false;
  $("#synthBanner").hidden = !state.dataset.is_synthetic;
  renderDataset();
  resetResults();
  const runs = await api(`/api/runs?dataset_id=${id}`);
  const latest = runs.find((r) => r.status === "completed") || runs.find((r) => ["queued", "running"].includes(r.status)) || runs[0];
  if (!latest) return;
  if (latest.status === "completed") await showRun(latest.id);
  else if (["queued", "running"].includes(latest.status)) pollRun(latest.id);
  else showRunError(latest.error);
}

function renderDataset() {
  const d = state.dataset;
  const r = d.report;
  $("#dsTitle").textContent = d.name;
  $("#dsMeta").textContent = `${r.date_range.start} to ${r.date_range.end} · ${fmt.format(r.date_range.days)} days · ${fmt.format(r.rows_kept)} of ${fmt.format(r.rows_in)} rows kept`;
  $("#kpiProducts").textContent = fmt.format(r.products);
  $("#kpiProductsSub").textContent = "in the cleaned data";
  $("#kpiUnits").textContent = fmt.format(r.total_units);
  $("#kpiUnitsSub").textContent = `${r.date_range.start} to ${r.date_range.end}`;
  $("#dqSub").textContent = d.options.absent_days_are_zero
    ? "Absent days count as zero sales (you confirmed the file lists every sale)."
    : "Absent days are treated as unknown, not zero.";
  const issues = r.issues.filter((i) => i.code !== "columns" || i.level !== "info");
  $("#issues").innerHTML = issues.length
    ? issues.map((i) => `<li class="${i.level}">${esc(i.message)}${i.count !== undefined ? ` <b>(${fmt.format(i.count)})</b>` : ""}${i.examples ? `<small>Examples: ${i.examples.map(esc).join(", ")}</small>` : ""}</li>`).join("")
    : `<li class="info">No problems found.</li>`;
  const merges = r.name_merges || [];
  $("#mergesBox").hidden = !merges.length;
  $("#mergesSummary").textContent = `${merges.length} medicines had more than one spelling and were grouped`;
  $("#merges").innerHTML = `<table class="table"><thead><tr><th>Kept as</th><th>Also written as</th></tr></thead><tbody>${merges.map((m) => `<tr><td>${esc(m.kept_as)}</td><td>${m.variants.slice(1).map((v) => `${esc(v.name)} <span class="muted">(${fmt.format(v.rows)})</span>`).join("<br>")}</td></tr>`).join("")}</tbody></table>`;
  const dupes = r.possible_duplicates || [];
  $("#dupesBox").hidden = !dupes.length;
  $("#dupes").innerHTML = `<p class="muted">Kept as separate medicines. If two are really the same product, upload again with a name alias.</p><table class="table"><tbody>${dupes.map((p) => `<tr><td>${esc(p.a)}</td><td>${esc(p.b)}</td></tr>`).join("")}</tbody></table>`;
}

function resetResults() {
  $("#runError").hidden = true;
  $("#progress").hidden = true;
  $("#runBtn").disabled = false;
  $("#runBtn").textContent = "Train and forecast";
  $("#kpiForecast").textContent = "–";
  $("#kpiForecastSub").textContent = "Train a model to see forecasts";
  $("#kpiModel").textContent = "–";
  $("#kpiModelSub").textContent = "";
  $("#medTable tbody").innerHTML = `<tr><td colspan="7" class="muted pad">Train a model to see forecasts per medicine.</td></tr>`;
  $("#modelTable tbody").innerHTML = `<tr><td colspan="6" class="muted pad">No results yet.</td></tr>`;
  $("#selectionReason").textContent = "";
  $("#warnings").innerHTML = "";
  $("#splitInfo").textContent = "";
  $("#detail").hidden = true;
  $$("[data-export]").forEach((a) => { a.classList.add("is-disabled"); a.removeAttribute("href"); });
}

/* ---------------- runs ---------------- */
async function startRun() {
  if (!state.dataset) return;
  $("#runError").hidden = true;
  $("#runBtn").disabled = true;
  try {
    const run = await api(`/api/datasets/${state.dataset.id}/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    pollRun(run.id);
  } catch (err) {
    showRunError(err.message);
  }
}

function stopPolling() { clearTimeout(state.poll); state.poll = null; }

function pollRun(id) {
  stopPolling();
  $("#progress").hidden = false;
  $("#runBtn").disabled = true;
  $("#runBtn").textContent = "Training…";
  const tick = async () => {
    try {
      const run = await api(`/api/runs/${id}`);
      const p = Math.round((run.progress || 0) * 100);
      $("#progressMsg").textContent = run.message || run.status;
      $("#progressPct").textContent = `${p}%`;
      $("#progressBar").style.width = `${p}%`;
      if (run.status === "completed") { $("#progress").hidden = true; await showRun(id); toast("Forecast ready"); return; }
      if (run.status === "failed") { $("#progress").hidden = true; showRunError(run.error); return; }
      state.poll = setTimeout(tick, 900);
    } catch (err) {
      state.poll = setTimeout(tick, 2500);
    }
  };
  tick();
}

function showRunError(message) {
  $("#runBtn").disabled = false;
  $("#runBtn").textContent = "Try again";
  const el = $("#runError");
  el.textContent = message || "The run failed.";
  el.hidden = false;
}

async function showRun(id) {
  state.run = await api(`/api/runs/${id}`);
  $("#runBtn").disabled = false;
  $("#runBtn").textContent = "Retrain";
  renderResults();
}

function renderResults() {
  const res = state.run.result;
  if (!res) return;
  const products = res.products;
  renderForecastKpi();
  const chosenVal = res.model_metrics.find((m) => m.selected && m.stage === "validation");
  const chosenTest = res.model_metrics.find((m) => m.selected && m.stage === "test");
  $("#kpiModel").textContent = res.selected_model;
  $("#kpiModelSub").textContent = chosenTest ? `Test error on totals ${pct(chosenTest.horizon_wape)} · validation ${pct(chosenVal?.horizon_wape)}` : "";
  renderModels();
  renderMedicines();
  $("#selectionReason").textContent = res.selection_reason;
  $("#warnings").innerHTML = (res.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
  const s = res.split;
  $("#splitInfo").textContent = `Trained up to ${s.train_end}. Validation ${s.val_start} to ${s.val_end}; test ${s.test_start} to ${s.test_end}. ${s.evaluated_products} of ${s.total_products} medicines evaluated. Forecast starts ${s.forecast_start}.`;
  $$("[data-export]").forEach((a) => {
    a.classList.remove("is-disabled");
    a.removeAttribute("aria-disabled");
    a.href = `/api/runs/${state.run.id}/export/${a.dataset.export}${a.dataset.export.endsWith(".html") ? "" : ""}`;
    if (a.dataset.export.endsWith(".html")) a.target = "_blank";
  });
  const first = products.find((p) => p.status === "forecast") || products[0];
  if (first) selectMedicine(state.selectedKey || first.product_key);
}

function renderForecastKpi() {
  const res = state.run?.result;
  if (!res) return;
  const h = state.horizon;
  const total = res.products.reduce((s, p) => s + (p[`forecast_${h}d`] || 0), 0);
  const revenue = res.products.reduce((s, p) => s + (p[`revenue_${h}d`] || 0), 0);
  $("#kpiForecast").textContent = `${fmt.format(total)} units`;
  $("#kpiForecastSub").textContent = revenue ? `About ${money.format(revenue)} at current prices, next ${h} days` : `Next ${h} days`;
}

function renderModels() {
  const rows = state.run.result.model_metrics.filter((m) => m.stage === state.stage).sort((a, b) => (a.horizon_wape ?? 9) - (b.horizon_wape ?? 9));
  $("#modelSub").textContent = state.stage === "test"
    ? "Measured on the final weeks, which no model saw while being chosen."
    : "Measured on the weeks used to choose the model.";
  $("#modelTable tbody").innerHTML = rows.map((m) => `<tr class="${m.selected ? "is-chosen" : ""}">
      <td>${esc(m.model)}${m.selected ? '<span class="tag">Chosen</span>' : ""}<small>${esc(m.family)}</small></td>
      <td class="num">${m.mae?.toFixed(2) ?? "–"}</td><td class="num">${m.rmse?.toFixed(2) ?? "–"}</td>
      <td class="num">${pct(m.wape)}</td><td class="num"><b>${pct(m.horizon_wape)}</b></td><td class="num">${signedPct(m.bias)}</td></tr>`).join("");
}

function renderMedicines() {
  const res = state.run?.result;
  if (!res) return;
  const q = $("#medSearch").value.trim().toLowerCase();
  const f = $("#medFilter").value;
  const list = res.products.filter((p) => (!q || p.product_name.toLowerCase().includes(q)) && (!f || p.status === f));
  $("#medTable tbody").innerHTML = list.length ? list.map((p) => `<tr data-key="${esc(p.product_key)}" tabindex="0" class="${p.product_key === state.selectedKey ? "is-selected" : ""}">
      <td>${esc(p.product_name)}</td>
      <td class="num">${fmt1.format(p.forecast_7d)}</td><td class="num">${fmt1.format(p.forecast_14d)}</td><td class="num"><b>${fmt1.format(p.forecast_30d)}</b></td>
      <td class="num">${p.revenue_30d ? money.format(p.revenue_30d) : "–"}</td>
      <td class="num">${pct(p.test_total_error, 0)}</td>
      <td><span class="chip chip--${p.status}">${STATUS[p.status] || p.status}</span></td></tr>`).join("")
    : `<tr><td colspan="7" class="muted pad">No medicines match.</td></tr>`;
}

async function selectMedicine(key) {
  state.selectedKey = key;
  $$("#medTable tbody tr").forEach((tr) => tr.classList.toggle("is-selected", tr.dataset.key === key));
  const data = await api(`/api/runs/${state.run.id}/products/${encodeURIComponent(key)}`);
  const p = data.product;
  $("#detail").hidden = false;
  $("#detailName").textContent = p.product_name;
  $("#detailStatus").textContent = `${STATUS[p.status] || p.status}${p.status === "forecast" ? ` · ${data.selected_model}` : ""}${p.intermittent ? " · sells irregularly" : ""}`;
  $("#detailFacts").innerHTML = [
    ["Next 7 days", fmt1.format(p.forecast_7d)],
    ["Next 14 days", fmt1.format(p.forecast_14d)],
    ["Next 30 days", fmt1.format(p.forecast_30d)],
    ["Average per day", p.mean_daily === null ? "–" : fmt1.format(p.mean_daily)],
    ["Test total: actual vs forecast", p.test_actual_total === null ? "–" : `${fmt.format(p.test_actual_total)} vs ${fmt.format(p.test_predicted_total)}`],
    ["Last price", p.last_price ? money.format(p.last_price) : "–"],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
  const note = $("#detailNote");
  note.hidden = !p.note;
  note.textContent = p.note;
  drawChart($("#detailChart"), data.series);
}

/* ---------------- chart ---------------- */
function drawChart(el, series) {
  const W = el.clientWidth || 800;
  const H = el.clientHeight || 300;
  const m = { t: 18, r: 12, b: 26, l: 40 };
  const hist = series.history.map((d) => ({ x: new Date(d.date), y: d.units }));
  const back = series.backtest.map((d) => ({ x: new Date(d.date), y: d.units }));
  const fc = series.forecast.map((d) => ({ x: new Date(d.date), y: d.units }));
  const all = [...hist, ...back, ...fc];
  if (!all.length) { el.innerHTML = `<p class="muted pad">No data to chart.</p>`; return; }
  const x0 = Math.min(...all.map((d) => d.x)), x1 = Math.max(...all.map((d) => d.x));
  const yMax = Math.max(1, ...all.map((d) => d.y ?? 0)) * 1.1;
  const sx = (x) => m.l + ((x - x0) / Math.max(1, x1 - x0)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - (y / yMax) * (H - m.t - m.b);
  const path = (pts) => {
    let d = "", pen = false;
    for (const p of pts) {
      if (p.y === null || p.y === undefined) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  const ticks = 4;
  let grid = "";
  for (let i = 0; i <= ticks; i++) {
    const v = (yMax / ticks) * i;
    grid += `<line class="grid-line" x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/><text class="axis" x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmt.format(v)}</text>`;
  }
  let months = "";
  const d = new Date(x0); d.setDate(1); d.setMonth(d.getMonth() + 1);
  while (d <= x1) {
    months += `<text class="axis" x="${sx(d)}" y="${H - 6}" text-anchor="middle">${d.toLocaleDateString("en-GB", { month: "short" })}</text>`;
    d.setMonth(d.getMonth() + 1);
  }
  const fStart = fc[0]?.x;
  const area = fc.length ? `${path(fc)}L${sx(fc[fc.length - 1].x)},${sy(0)}L${sx(fc[0].x)},${sy(0)}Z` : "";
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily sales history, test forecast and future forecast">
    <defs><linearGradient id="fcGrad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#2457d6" stop-opacity="0.22"/><stop offset="1" stop-color="#2457d6" stop-opacity="0"/></linearGradient></defs>
    ${grid}${months}
    ${fStart ? `<line class="divider" x1="${sx(fStart)}" x2="${sx(fStart)}" y1="${m.t - 6}" y2="${H - m.b}"/><text class="label" x="${sx(fStart) + 6}" y="${m.t}">Forecast</text>` : ""}
    ${back.length ? `<text class="label" x="${sx(back[0].x) + 6}" y="${m.t}">Test</text><line class="divider" x1="${sx(back[0].x)}" x2="${sx(back[0].x)}" y1="${m.t - 6}" y2="${H - m.b}"/>` : ""}
    <path class="hist" d="${path(hist)}"/>
    <path class="back" d="${path(back)}"/>
    ${area ? `<path class="fc-area" d="${area}"/>` : ""}
    <path class="fc" d="${path(fc)}"/>
    <line class="hover-line" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/>
  </svg><div class="tip"></div>`;
  const svg = $("svg", el), tip = $(".tip", el), hl = $(".hover-line", el);
  const byDay = new Map();
  const add = (arr, k) => arr.forEach((p) => { const key = p.x.toISOString().slice(0, 10); byDay.set(key, { ...(byDay.get(key) || { x: p.x }), [k]: p.y }); });
  add(hist, "a"); add(back, "b"); add(fc, "f");
  const days = [...byDay.values()].sort((a, b) => a.x - b.x);
  svg.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const t = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
    let best = days[0];
    for (const dd of days) if (Math.abs(dd.x - t) < Math.abs(best.x - t)) best = dd;
    const x = sx(best.x);
    hl.setAttribute("x1", x); hl.setAttribute("x2", x); hl.setAttribute("visibility", "visible");
    const parts = [best.x.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" })];
    if (best.a !== undefined) parts.push(`actual ${best.a === null ? "unknown" : fmt1.format(best.a)}`);
    if (best.b !== undefined) parts.push(`test forecast ${fmt1.format(best.b)}`);
    if (best.f !== undefined) parts.push(`forecast ${fmt1.format(best.f)}`);
    tip.textContent = parts.join(" · ");
    tip.style.left = `${(x / W) * rect.width}px`;
    tip.style.top = `${e.clientY - rect.top}px`;
    tip.style.opacity = 1;
  });
  svg.addEventListener("mouseleave", () => { tip.style.opacity = 0; hl.setAttribute("visibility", "hidden"); });
}

/* ---------------- upload and demo ---------------- */
async function runDemo() {
  $$("#demoBtn, [data-demo]").forEach((b) => { b.disabled = true; });
  toast("Generating synthetic demo data…");
  try {
    const ds = await api("/api/datasets/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    await loadDatasets(ds.id);
    await startRun();
  } catch (err) {
    toast(err.message);
  } finally {
    $$("#demoBtn, [data-demo]").forEach((b) => { b.disabled = false; });
  }
}

function setupUpload() {
  const dlg = $("#uploadDialog");
  const open = () => { $("#uploadError").hidden = true; dlg.showModal(); };
  $("#uploadBtn").addEventListener("click", open);
  $$("[data-open-upload]").forEach((b) => b.addEventListener("click", open));
  $$("[data-close]", dlg).forEach((b) => b.addEventListener("click", () => dlg.close()));
  const drop = $("#drop"), input = $("#fileInput");
  input.addEventListener("change", () => { $("#dropText").innerHTML = input.files[0] ? `<b>${esc(input.files[0].name)}</b>` : "<b>Choose a CSV or Excel file</b> or drag it here"; });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("is-over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event("change")); } });

  $("#uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    if (!input.files.length) { $("#uploadError").textContent = "Choose a file first."; $("#uploadError").hidden = false; return; }
    const fd = new FormData();
    fd.append("file", input.files[0]);
    fd.append("absent_days_are_zero", form.absent_days_are_zero.checked);
    fd.append("drop_exact_duplicates", form.drop_exact_duplicates.checked);
    fd.append("date_format", form.date_format.value);
    if (form.column_mapping.value.trim()) fd.append("column_mapping", form.column_mapping.value.trim());
    if (form.name_aliases.value.trim()) fd.append("name_aliases", form.name_aliases.value.trim());
    const btn = $("#uploadSubmit");
    btn.disabled = true; btn.textContent = "Checking…";
    try {
      const ds = await api("/api/datasets", { method: "POST", body: fd });
      dlg.close();
      form.reset(); $("#dropText").innerHTML = "<b>Choose a CSV or Excel file</b> or drag it here";
      await loadDatasets(ds.id);
      toast("Uploaded. Review the data quality notes, then train.");
    } catch (err) {
      $("#uploadError").textContent = err.message;
      $("#uploadError").hidden = false;
    } finally {
      btn.disabled = false; btn.textContent = "Upload and check";
    }
  });
}

/* ---------------- wiring ---------------- */
function init() {
  setupUpload();
  $("#demoBtn").addEventListener("click", runDemo);
  $$("[data-demo]").forEach((b) => b.addEventListener("click", runDemo));
  $("#runBtn").addEventListener("click", startRun);
  $("#datasetSelect").addEventListener("change", (e) => e.target.value && selectDataset(e.target.value));
  $$(".kpi .seg button").forEach((b) => b.addEventListener("click", () => {
    $$(".kpi .seg button").forEach((x) => x.classList.toggle("is-on", x === b));
    state.horizon = Number(b.dataset.h);
    renderForecastKpi();
  }));
  $$("[data-stage]").forEach((b) => b.addEventListener("click", () => {
    $$("[data-stage]").forEach((x) => x.classList.toggle("is-on", x === b));
    state.stage = b.dataset.stage;
    if (state.run) renderModels();
  }));
  $("#medSearch").addEventListener("input", renderMedicines);
  $("#medFilter").addEventListener("change", renderMedicines);
  $("#medTable").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-key]"); if (tr) selectMedicine(tr.dataset.key); });
  $("#medTable").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-key]"); if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectMedicine(tr.dataset.key); } });
  let resize;
  window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => state.selectedKey && state.run && selectMedicine(state.selectedKey), 200); });
  loadDatasets().catch((err) => toast(err.message));
}

document.addEventListener("DOMContentLoaded", init);
