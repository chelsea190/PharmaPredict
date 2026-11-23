"""Exports: forecast CSVs and a self-contained HTML report with Matplotlib charts."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, BLUE, TEAL, AMBER, LINE = "#14213d", "#5b6b80", "#2457d6", "#0f9d8a", "#e08a00", "#dfe5ee"


def daily_forecast_csv(forecasts: pd.DataFrame, is_synthetic: bool) -> str:
    out = forecasts.rename(columns={"yhat": "forecast_units"}).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["forecast_units"] = out["forecast_units"].round(2)
    out.insert(0, "data_source", "SYNTHETIC DEMO DATA" if is_synthetic else "uploaded data")
    return out[["data_source", "product_name", "product_key", "date", "day", "forecast_units", "method"]].to_csv(index=False)


def summary_csv(products: list[dict], horizons: tuple[int, ...], is_synthetic: bool) -> str:
    frame = pd.DataFrame(products)
    cols = ["product_name", "status"] + [f"forecast_{h}d" for h in horizons] + [f"revenue_{h}d" for h in horizons] + [
        "last_price", "test_wape", "test_total_error", "test_bias", "observed_days", "total_units", "note"]
    frame = frame[[c for c in cols if c in frame.columns]]
    frame = frame.round({c: 2 for c in frame.columns if c.startswith(("forecast_", "revenue_", "last_price"))})
    frame = frame.round({c: 4 for c in ("test_wape", "test_total_error", "test_bias") if c in frame.columns})
    frame.insert(0, "data_source", "SYNTHETIC DEMO DATA" if is_synthetic else "uploaded data")
    return frame.to_csv(index=False)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)


def model_chart(metrics: list[dict]) -> str:
    test = [m for m in metrics if m["stage"] == "test" and m.get("horizon_wape") is not None]
    test.sort(key=lambda m: m["horizon_wape"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(test) + 0.8))
    colors = [BLUE if m["selected"] else "#b8c4d6" for m in test]
    ax.barh([m["model"] for m in test], [m["horizon_wape"] * 100 for m in test], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test error on 7/14/30-day totals (horizon WAPE, %). Lower is better.", color=MUTED, fontsize=8)
    for i, m in enumerate(test):
        ax.text(m["horizon_wape"] * 100 + 0.3, i, f"{m['horizon_wape']:.1%}", va="center", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="x", color=LINE, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def product_chart(name: str, entry: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 2.4))
    hist = pd.DataFrame(entry["history"])
    if len(hist):
        ax.plot(pd.to_datetime(hist["date"]), hist["units"], color="#9aa8bc", linewidth=1, label="Actual sales")
    if entry.get("backtest"):
        bt = pd.DataFrame(entry["backtest"])
        ax.plot(pd.to_datetime(bt["date"]), bt["units"], color=AMBER, linewidth=1.6, label="Test forecast (backtest)")
    fc = pd.DataFrame(entry["forecast"])
    if len(fc):
        ax.plot(pd.to_datetime(fc["date"]), fc["units"], color=BLUE, linewidth=2, label="Forecast")
    ax.set_title(name, loc="left", fontsize=10, color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    _style(ax)
    return _png(fig)


def html_report(dataset: dict, run: dict) -> str:
    result = run["result"]
    is_synth = dataset["is_synthetic"]
    esc = html.escape
    horizons = tuple(run["config"].get("horizons", (7, 14, 30)))
    products = result["products"]
    forecastable = [p for p in products if p["status"] == "forecast"]
    top = forecastable[:6]
    split = result["split"]
    rep = dataset["report"]

    rows = "".join(
        f"<tr class='{'sel' if m['selected'] else ''}'><td>{esc(m['model'])}{' (chosen)' if m['selected'] else ''}</td>"
        f"<td>{esc(m['family'])}</td><td>{m['mae']:.2f}</td><td>{m['rmse']:.2f}</td><td>{m['wape']:.1%}</td>"
        f"<td>{(m.get('horizon_wape') or 0):.1%}</td><td>{(m.get('bias') or 0):+.1%}</td></tr>"
        for m in result["model_metrics"] if m["stage"] == "test"
    )
    def pct(value):
        return "" if value is None else f"{value:.0%}"

    prod_rows = "".join(
        "<tr><td>" + esc(p["product_name"]) + "</td>"
        + "".join(f"<td>{p['forecast_' + str(h) + 'd']:,.0f}</td>" for h in horizons)
        + "<td>" + pct(p["test_total_error"]) + "</td><td>" + esc(p["status"].replace("_", " ")) + "</td></tr>"
        for p in products
    )
    issues = "".join(
        "<li><b>" + esc(i["level"]) + "</b>: " + esc(i["message"])
        + (f" ({i['count']:,})" if "count" in i else "") + "</li>"
        for i in rep.get("issues", [])
    )
    charts = "".join(
        "<img src='" + product_chart(p["product_name"], result["series"][p["product_key"]]) + "' alt='Sales and forecast for "
        + esc(p["product_name"]) + "'>"
        for p in top
    )
    banner = ("<div class='synth'>SYNTHETIC DEMO DATA. These numbers come from generated data, not a real pharmacy. "
              "They show how the system works and say nothing about real-world accuracy.</div>") if is_synth else ""
    warns = "".join(f"<li>{esc(w)}</li>" for w in result.get("warnings", []))
    warn_block = f"<h2>Warnings</h2><ul>{warns}</ul>" if warns else ""
    head_cells = "".join(f"<th>Next {h} days</th>" for h in horizons)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PharmaPredict report: {esc(dataset['name'])}</title>
<style>
body{{font-family:Inter,Segoe UI,system-ui,sans-serif;color:{INK};max-width:980px;margin:32px auto;padding:0 20px;line-height:1.5}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}} .muted{{color:{MUTED}}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid {LINE}}}
th{{color:{MUTED};font-weight:600}} tr.sel td{{background:#eef3ff;font-weight:600}} img{{max-width:100%;margin:6px 0}}
.synth{{background:#fff4e0;border:1px solid #f2c36b;color:#7a4b00;padding:12px 14px;border-radius:10px;font-weight:600;margin:16px 0}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}} .kpi{{border:1px solid {LINE};border-radius:10px;padding:12px}}
.kpi b{{display:block;font-size:22px}}
</style></head><body>
{banner}
<h1>PharmaPredict demand forecast</h1>
<p class="muted">Dataset: {esc(dataset['name'])} · Generated {datetime.now().strftime('%d %b %Y %H:%M')} · Run {esc(run['id'])}</p>
<div class="kpis">
<div class="kpi"><span class="muted">Medicines</span><b>{len(products)}</b></div>
<div class="kpi"><span class="muted">Units in history</span><b>{rep.get('total_units', 0):,.0f}</b></div>
<div class="kpi"><span class="muted">Forecast, next 30 days</span><b>{sum(p.get('forecast_30d', 0) for p in products):,.0f}</b></div>
<div class="kpi"><span class="muted">Chosen model</span><b style="font-size:16px">{esc(result['selected_model'])}</b></div>
</div>
<p>{esc(result['selection_reason'])}</p>
<h2>How the models were tested</h2>
<p>Data from {esc(rep['date_range']['start'])} to {esc(rep['date_range']['end'])}. Training up to {esc(split['train_end'])},
validation {esc(split['val_start'])} to {esc(split['val_end'])}, test {esc(split['test_start'])} to {esc(split['test_end'])}.
The model was chosen on validation only. The table shows the test window, which no model saw while being chosen.
{split['evaluated_products']} of {split['total_products']} medicines had enough history to be evaluated.</p>
<img src="{model_chart(result['model_metrics'])}" alt="Model comparison">
<table><tr><th>Model</th><th>Type</th><th>MAE</th><th>RMSE</th><th>WAPE (daily)</th><th>WAPE (7/14/30-day totals)</th><th>Bias</th></tr>{rows}</table>
<p class="muted">MAE and RMSE are in units per medicine per day. Bias above zero means over-forecasting (overstock risk); below zero means under-forecasting (stock-out risk).</p>
<h2>Forecasts by medicine</h2>
<table><tr><th>Medicine</th>{head_cells}<th>Test error (total)</th><th>Status</th></tr>{prod_rows}</table>
<h2>Top medicines: history, backtest and forecast</h2>{charts}
<h2>Data quality</h2><ul>{issues}</ul>
{warn_block}
<h2>Limitations</h2>
<ul><li>Forecasts assume the future looks like the past. Promotions, new competitors, stock-outs and outbreaks are not known to the model.</li>
<li>Accuracy is measured on one recent test window. It is an estimate, not a guarantee.</li>
<li>Commercial-grade accuracy has not been shown. Validate on several months of your own real data before relying on these numbers for ordering.</li></ul>
</body></html>"""
