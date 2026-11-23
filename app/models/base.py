"""Common interface for every forecasting model.

A model is fitted on history that ends before ``start`` and then asked for a
forecast of ``horizon`` days starting at ``start``. It never sees actual values
from the forecast window. Multi-day forecasts are produced recursively by the
ML models (each predicted day feeds the next day's lag features), so the
evaluation matches how the model is used in real life.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    name: str = "model"
    family: str = "baseline"   # baseline | statistical | machine learning
    description: str = ""

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Learn from a daily panel (product_key, date, quantity, price, first_date)."""
        return self

    @abstractmethod
    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Return columns product_key, date, yhat for every product in ``history``."""


def future_frame(products: list[str], start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=horizon, freq="D")
    return pd.MultiIndex.from_product([products, dates], names=["product_key", "date"]).to_frame(index=False)
