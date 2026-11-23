"""End to end: clean sales -> daily series -> evaluate models -> pick one -> forecast.

Stages
------
1. Validation. Fit every model on the training window and forecast the
   validation window from its first day. Pick the model with the lowest
   validation "horizon WAPE": the WAPE of each medicine's total demand over the
   first 7, 14 and 30 days, averaged. Those totals are exactly what the app
   promises, and they are what a pharmacy orders against. Daily MAE, RMSE and
   WAPE are reported too. Training accuracy is never used for the choice.
2. Test. Refit every model on training + validation and forecast the test
   window. This is the honest estimate of accuracy, reported for all models.
   The choice made in stage 1 is not changed by test results.
3. Production. Refit the chosen model on all data and forecast the next 30
   days. 7 and 14 day figures are the first 7 and 14 days of that forecast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from app.config import settings
from app.evaluation.metrics import all_metrics
from app.evaluation.splits import Split, make_split
from app.features.series import build_daily_panel, series_profile
from app.models.base import Forecaster
from app.models.baselines import MovingAverage, SeasonalNaive
from app.models.ml import linear_regression, random_forest, xgboost_model
from app.models.statistical import CrostonSBA, HoltWinters

Progress = Callable[[str, float], None]
FAMILY_ORDER = {"baseline": 0, "statistical": 1, "machine learning": 2}
HORIZON_BUCKETS = ((1, 7), (8, 14), (15, 30))
NO_RECENT_SALES_DAYS = 60


@dataclass
class RunConfig:
    absent_days_are_zero: bool = False
    models: list[str] | None = None          # names to include; None = all
    eval_days: int = settings.eval_days
    min_eval_days: int = settings.min_eval_days
    min_train_days: int = settings.min_train_days
    horizons: tuple[int, ...] = settings.horizons
    random_state: int = settings.random_state


@dataclass
class RunResult:
    split: dict
    selected_model: str
    selection_reason: str
    model_metrics: list[dict]
    products: list[dict]
    series: dict[str, dict]
    forecasts: pd.DataFrame
    feature_importance: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def default_models(random_state: int = 42) -> list[Forecaster]:
    return [
        SeasonalNaive(),
        MovingAverage(28),
        HoltWinters(),
        CrostonSBA(),
        linear_regression(),
        random_forest(random_state),
        xgboost_model(random_state),
    ]


def _total_wape(merged: pd.DataFrame, days: int) -> float:
    """Error on each medicine's TOTAL demand over the first ``days`` days.

    This is what matters for ordering stock: being off by 3 units on Monday and
    -3 on Tuesday cancels out in a weekly order. Only days with a known actual
    count, for both the actual and the forecast.
    """
    part = merged[(merged["day"] <= days) & merged["quantity"].notna()]
    if part.empty:
        return float("nan")
    sums = part.groupby("product_key")[["quantity", "yhat"]].sum()
    denom = sums["quantity"].abs().sum()
    return float((sums["yhat"] - sums["quantity"]).abs().sum() / denom) if denom > 0 else float("nan")


def _score(pred: pd.DataFrame, actual: pd.DataFrame, start: pd.Timestamp,
           horizons: tuple[int, ...] = (7, 14, 30)) -> tuple[dict, list[dict], pd.DataFrame]:
    merged = pred.merge(actual[["product_key", "date", "quantity"]], on=["product_key", "date"], how="left")
    merged["day"] = (merged["date"] - start).dt.days + 1
    overall = all_metrics(merged["quantity"], merged["yhat"])
    window = int(merged["day"].max()) if len(merged) else 0
    totals = []
    for h in horizons:
        if h <= window:
            overall[f"total_wape_{h}d"] = _total_wape(merged, h)
            totals.append(overall[f"total_wape_{h}d"])
    if window and window not in horizons:
        overall[f"total_wape_{window}d"] = _total_wape(merged, window)
        totals.append(overall[f"total_wape_{window}d"])
    finite = [t for t in totals if np.isfinite(t)]
    overall["horizon_wape"] = float(np.mean(finite)) if finite else float("nan")
    buckets = []
    for lo, hi in HORIZON_BUCKETS:
        part = merged[(merged["day"] >= lo) & (merged["day"] <= hi)]
        if len(part):
            buckets.append({"days": f"{lo}-{hi}", **all_metrics(part["quantity"], part["yhat"])})
    return overall, buckets, merged


def closed_weekdays(panel: pd.DataFrame, weeks: int = 12) -> list[int]:
    """Weekdays the shop is normally shut (open on fewer than 10% of the last ``weeks`` weeks).

    Forecasts for those weekdays are set to zero, so weekly totals are not
    inflated by sales on days the pharmacy does not trade.
    """
    days = panel.drop_duplicates("date")[["date", "open"]]
    days = days[days["date"] > days["date"].max() - pd.Timedelta(weeks=weeks)]
    share_open = days.groupby(days["date"].dt.dayofweek)["open"].mean()
    return [int(d) for d, share in share_open.items() if share < 0.1]


WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _clean_float(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), 4)


def run_forecasting(clean: pd.DataFrame, config: RunConfig | None = None, progress: Progress | None = None) -> RunResult:
    config = config or RunConfig()
    say = progress or (lambda message, fraction: None)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    say("Building daily series", 0.02)
    panel = build_daily_panel(clean, config.absent_days_are_zero)
    profile = series_profile(panel)
    panel["first_date"] = panel["product_key"].map(profile["first_date"])
    first, last = panel["date"].min(), panel["date"].max()
    split: Split = make_split(first, last, config.eval_days, config.min_eval_days, config.min_train_days)
    if split.window < config.eval_days:
        warnings.append(f"The data is short, so validation and test windows were cut to {split.window} days each.")

    names = clean.drop_duplicates("product_key").set_index("product_key")["product_name"]
    train_obs = panel[panel["date"] <= split.train_end].groupby("product_key")["quantity"].count()
    profile["train_observed_days"] = train_obs.reindex(profile.index, fill_value=0)
    eligible = profile.index[profile["train_observed_days"] >= config.min_train_days].tolist()
    if not eligible:
        raise ValueError(
            f"No medicine has {config.min_train_days} days of known sales before the validation window. "
            "Upload a longer history, or confirm that your file lists every sale so absent days count as zero."
        )
    if len(eligible) < len(profile):
        warnings.append(
            f"{len(profile) - len(eligible)} of {len(profile)} medicines have less than {config.min_train_days} days of "
            "known history. They are not evaluated and get a rough estimate only."
        )

    models = default_models(config.random_state)
    if config.models:
        models = [m for m in models if m.name in config.models] or models
    model_metrics: list[dict] = []
    val_scores: dict[str, float] = {}

    def history_until(end: pd.Timestamp, keys: list[str]) -> pd.DataFrame:
        return panel[(panel["date"] <= end) & panel["product_key"].isin(keys)]

    def prepare(model: Forecaster, end: pd.Timestamp) -> list[int]:
        """Tell the model which weekdays are shut, using only data up to ``end``."""
        shut_days = closed_weekdays(panel[panel["date"] <= end])
        if hasattr(model, "closed_days"):
            model.closed_days = shut_days
        return shut_days

    def zero_closed(pred: pd.DataFrame, shut_days: list[int]) -> pd.DataFrame:
        if shut_days:
            pred.loc[pred["date"].dt.dayofweek.isin(shut_days), "yhat"] = 0.0
        return pred

    # ---- stage 1: validation ------------------------------------------
    steps = len(models) * 2 + 1
    done = 0
    train_all = panel[panel["date"] <= split.train_end]
    val_actual = panel[(panel["date"] >= split.val_start) & (panel["date"] <= split.val_end)]
    for model in models:
        say(f"Validation: {model.name}", 0.05 + 0.85 * done / steps)
        t0 = time.time()
        try:
            shut_days = prepare(model, split.train_end)
            model.fit(train_all)
            pred = zero_closed(model.forecast(history_until(split.train_end, eligible), split.val_start, split.window), shut_days)
            overall, buckets, _ = _score(pred, val_actual, split.val_start, config.horizons)
            val_scores[model.name] = overall["horizon_wape"]
            model_metrics.append({"model": model.name, "family": model.family, "stage": "validation",
                                  **overall, "by_horizon": buckets})
        except Exception as exc:  # noqa: BLE001 - one failing model must not stop the run
            warnings.append(f"{model.name} failed during validation and was skipped: {exc}")
        timings[f"val:{model.name}"] = round(time.time() - t0, 2)
        done += 1

    ranked = [m for m in models if m.name in val_scores and np.isfinite(val_scores[m.name])]
    if not ranked:
        raise ValueError("Every model failed during validation. See the warnings.")
    ranked.sort(key=lambda m: (round(val_scores[m.name], 4), FAMILY_ORDER.get(m.family, 9)))
    chosen = ranked[0]
    naive_wape = val_scores.get(SeasonalNaive.name)
    reason = f"Lowest validation error on 7, 14 and 30 day totals (horizon WAPE {val_scores[chosen.name]:.1%})"
    if naive_wape and np.isfinite(naive_wape) and chosen.name != SeasonalNaive.name:
        reason += f", against {naive_wape:.1%} for the seasonal naive baseline"
    reason += ". Chosen before looking at the test window."

    # ---- stage 2: test ------------------------------------------------
    trainval = panel[panel["date"] <= split.val_end]
    test_actual = panel[(panel["date"] >= split.test_start) & (panel["date"] <= split.test_end)]
    backtest = None
    for model in models:
        if model.name not in val_scores:
            continue
        say(f"Test: {model.name}", 0.05 + 0.85 * done / steps)
        t0 = time.time()
        try:
            shut_days = prepare(model, split.val_end)
            model.fit(trainval)
            pred = zero_closed(model.forecast(history_until(split.val_end, eligible), split.test_start, split.window), shut_days)
            overall, buckets, merged = _score(pred, test_actual, split.test_start, config.horizons)
            model_metrics.append({"model": model.name, "family": model.family, "stage": "test",
                                  **overall, "by_horizon": buckets})
            if model.name == chosen.name:
                backtest = merged
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{model.name} failed during testing: {exc}")
        timings[f"test:{model.name}"] = round(time.time() - t0, 2)
        done += 1

    # ---- stage 3: production forecast ---------------------------------
    say(f"Forecasting the next {max(config.horizons)} days with {chosen.name}", 0.05 + 0.85 * done / steps)
    horizon = max(config.horizons)
    start = last + pd.Timedelta(days=1)
    t0 = time.time()
    shut = prepare(chosen, last)
    chosen.fit(panel)
    future = chosen.forecast(history_until(last, eligible), start, horizon)
    future["method"] = chosen.name
    rough_keys = [k for k in profile.index if k not in eligible]
    if rough_keys:
        rough = MovingAverage(28).forecast(history_until(last, rough_keys), start, horizon)
        rough["method"] = "Rough estimate (28-day average, not validated)"
        future = pd.concat([future, rough], ignore_index=True)
    if shut:
        future = zero_closed(future, shut)
        warnings.append("The pharmacy does not normally trade on " + ", ".join(WEEKDAY_NAMES[d] for d in shut)
                        + ", so forecasts for those days are zero.")
    timings["production"] = round(time.time() - t0, 2)
    importance = getattr(chosen, "feature_importance", {}) or {}

    # ---- per-product summary ------------------------------------------
    say("Summarising results", 0.95)
    last_price = clean.dropna(subset=["price"]).sort_values("date").groupby("product_key")["price"].last()
    future["day"] = (future["date"] - start).dt.days + 1
    totals = {h: future[future["day"] <= h].groupby("product_key")["yhat"].sum() for h in config.horizons}

    per_product_test: dict[str, dict] = {}
    if backtest is not None:
        for key, grp in backtest.groupby("product_key"):
            m = all_metrics(grp["quantity"], grp["yhat"])
            known = grp[grp["quantity"].notna()]
            actual_total = known["quantity"].sum()
            m["total_error"] = (abs(known["yhat"].sum() - actual_total) / actual_total) if actual_total > 0 else float("nan")
            m["actual_total"] = float(actual_total)
            m["predicted_total"] = float(known["yhat"].sum())
            per_product_test[key] = m

    products = []
    for key, row in profile.iterrows():
        if key in eligible:
            no_recent = pd.isna(row["last_sale"]) or (last - row["last_sale"]).days > NO_RECENT_SALES_DAYS
            status = "no_recent_sales" if no_recent else "forecast"
            note = (f"No sales in the last {NO_RECENT_SALES_DAYS} days. Possibly discontinued or out of stock."
                    if no_recent else "")
        else:
            status = "insufficient_history"
            note = (f"Only {int(row['train_observed_days'])} known days before the evaluation window "
                    f"(needs {config.min_train_days}). Rough estimate, not validated.")
        test_m = per_product_test.get(key, {})
        price = last_price.get(key, np.nan)
        item = {
            "product_key": key,
            "product_name": names.get(key, key),
            "status": status,
            "note": note,
            "first_date": row["first_date"].date().isoformat(),
            "observed_days": int(row["observed_days"]),
            "selling_days": int(row["selling_days"]),
            "total_units": float(row["total_units"]),
            "mean_daily": _clean_float(row["mean_daily"]),
            "intermittent": bool(row["adi"] > 1.32) if pd.notna(row["adi"]) else None,
            "last_price": _clean_float(price),
            "test_wape": _clean_float(test_m.get("wape")),
            "test_mae": _clean_float(test_m.get("mae")),
            "test_bias": _clean_float(test_m.get("bias")),
            "test_total_error": _clean_float(test_m.get("total_error")),
            "test_actual_total": _clean_float(test_m.get("actual_total")),
            "test_predicted_total": _clean_float(test_m.get("predicted_total")),
        }
        for h in config.horizons:
            units = float(totals[h].get(key, 0.0))
            item[f"forecast_{h}d"] = round(units, 2)
            item[f"revenue_{h}d"] = _clean_float(units * price) if pd.notna(price) else None
        products.append(item)
    products.sort(key=lambda p: -p[f"forecast_{max(config.horizons)}d"])

    # ---- chart series -------------------------------------------------
    series: dict[str, dict] = {}
    hist_from = last - pd.Timedelta(days=179)
    recent = panel[panel["date"] >= hist_from]
    for key in profile.index:
        h = recent[recent["product_key"] == key]
        f = future[future["product_key"] == key]
        entry = {
            "history": [{"date": d.date().isoformat(), "units": _clean_float(q)} for d, q in zip(h["date"], h["quantity"])],
            "forecast": [{"date": d.date().isoformat(), "units": _clean_float(q)} for d, q in zip(f["date"], f["yhat"])],
            "backtest": [],
        }
        if backtest is not None:
            b = backtest[backtest["product_key"] == key]
            entry["backtest"] = [{"date": d.date().isoformat(), "units": _clean_float(q), "actual": _clean_float(a)}
                                 for d, q, a in zip(b["date"], b["yhat"], b["quantity"])]
        series[key] = entry

    for metric in model_metrics:
        for k in [k for k in metric if k in ("mae", "rmse", "wape", "bias", "horizon_wape") or k.startswith("total_wape")]:
            metric[k] = _clean_float(metric[k])
        for bucket in metric["by_horizon"]:
            for k in ("mae", "rmse", "wape", "bias"):
                bucket[k] = _clean_float(bucket[k])
        metric["selected"] = metric["model"] == chosen.name

    say("Done", 1.0)
    return RunResult(
        split={**split.as_dict(), "evaluated_products": len(eligible), "total_products": int(len(profile)),
               "closed_weekdays": [WEEKDAY_NAMES[d] for d in shut], "forecast_start": start.date().isoformat()},
        selected_model=chosen.name,
        selection_reason=reason,
        model_metrics=model_metrics,
        products=products,
        series=series,
        forecasts=future.assign(product_name=future["product_key"].map(names)),
        feature_importance=importance,
        warnings=warnings,
        timings=timings,
    )
