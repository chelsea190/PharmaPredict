"""Classic time-series models, fitted separately for each medicine."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import Forecaster, future_frame


def _filled_series(history: pd.DataFrame, key: str) -> pd.Series:
    """Daily series for one product with unknown days interpolated.

    Holt-Winters cannot skip gaps, so short gaps are interpolated for fitting
    only. They are never counted as real observations in evaluation.
    """
    s = history.loc[history["product_key"] == key].set_index("date")["quantity"].asfreq("D")
    return s.interpolate(limit_direction="both").fillna(0.0)


class HoltWinters(Forecaster):
    """Exponential smoothing with a damped trend and weekly seasonality."""

    name = "Holt-Winters (weekly)"
    family = "statistical"
    description = "Exponential smoothing with weekly seasonality, fitted per medicine."
    min_points = 56  # 8 weeks

    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        out = future_frame(list(history["product_key"].unique()), start, horizon)
        preds: dict[str, np.ndarray] = {}
        for key in out["product_key"].unique():
            s = _filled_series(history, key).tail(365)
            fallback = np.full(horizon, s.tail(28).mean() if len(s) else 0.0)
            if len(s) < self.min_points or s.sum() == 0:
                preds[key] = fallback
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        s.to_numpy(dtype=float), trend="add", damped_trend=True,
                        seasonal="add", seasonal_periods=7, initialization_method="estimated",
                    ).fit(optimized=True)
                    values = np.asarray(model.forecast(horizon), dtype=float)
                if not np.all(np.isfinite(values)):
                    values = fallback
            except Exception:  # noqa: BLE001 - statsmodels can fail on odd series
                values = fallback
            preds[key] = values
        out["yhat"] = np.concatenate([preds[k] for k in out["product_key"].unique()])
        out["yhat"] = out["yhat"].clip(lower=0)
        return out


class CrostonSBA(Forecaster):
    """Croston's method with the Syntetos-Boylan correction.

    Built for intermittent demand (medicines that sell on some days and not
    others). It forecasts an average daily rate, so the forecast is flat.
    """

    name = "Croston SBA (intermittent)"
    family = "statistical"
    description = "For medicines that sell on some days and not others. Forecasts an average daily rate."

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def _rate(self, values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) == 0:
            return 0.0
        size = values[nonzero[0]]
        interval = float(nonzero[0] + 1)
        last = nonzero[0]
        for idx in nonzero[1:]:
            size += self.alpha * (values[idx] - size)
            interval += self.alpha * ((idx - last) - interval)
            last = idx
        return float((1 - self.alpha / 2) * size / max(interval, 1.0))

    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        out = future_frame(list(history["product_key"].unique()), start, horizon)
        rates = {
            key: self._rate(group["quantity"].to_numpy(dtype=float)[-365:])
            for key, group in history.sort_values("date").groupby("product_key")
        }
        out["yhat"] = out["product_key"].map(rates).fillna(0.0)
        return out
