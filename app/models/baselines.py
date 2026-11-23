"""Simple baselines. A real model has to beat these to be worth using."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, future_frame


class SeasonalNaive(Forecaster):
    """Next Monday = the most recent known Monday. Falls back to the 28-day average."""

    name = "Seasonal naive (last week)"
    family = "baseline"
    description = "Repeats the most recent known value for the same weekday."

    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        recent = history[history["date"] >= start - pd.Timedelta(days=56)].dropna(subset=["quantity"])
        last_by_dow = (
            recent.assign(dow=recent["date"].dt.dayofweek)
            .sort_values("date")
            .groupby(["product_key", "dow"])["quantity"].last()
        )
        fallback = recent.groupby("product_key")["quantity"].mean()
        out = future_frame(list(history["product_key"].unique()), start, horizon)
        out["dow"] = out["date"].dt.dayofweek
        out["yhat"] = [last_by_dow.get((k, d), np.nan) for k, d in zip(out["product_key"], out["dow"])]
        out["yhat"] = out["yhat"].fillna(out["product_key"].map(fallback)).fillna(0.0)
        return out.drop(columns="dow")


class MovingAverage(Forecaster):
    """Flat forecast at the average of the last 28 known days."""

    name = "Moving average (28 days)"
    family = "baseline"
    description = "Average of the last 28 known days, repeated forward."

    def __init__(self, window: int = 28):
        self.window = window

    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        recent = history[history["date"] >= start - pd.Timedelta(days=self.window)]
        level = recent.groupby("product_key")["quantity"].mean()
        older = history.groupby("product_key")["quantity"].apply(lambda s: s.dropna().tail(self.window).mean())
        out = future_frame(list(history["product_key"].unique()), start, horizon)
        out["yhat"] = out["product_key"].map(level).fillna(out["product_key"].map(older)).fillna(0.0)
        return out
