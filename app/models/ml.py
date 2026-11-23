"""Global machine learning models: one model learns from every medicine at once.

Training one model across all medicines lets slow sellers borrow patterns
(weekday effects, seasonality, price reactions) from busy ones. Forecasts
are made one day at a time: each predicted day becomes the lag for the next.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features.build import FEATURES, TAIL_DAYS, make_features

from .base import Forecaster, future_frame

MIN_HISTORY_FOR_ROW = 7  # a training row needs a week of past data


class GlobalMLForecaster(Forecaster):
    family = "machine learning"

    def __init__(self, name: str, factory: Callable[[], object], description: str = ""):
        self.name = name
        self.factory = factory
        self.description = description
        self.model = None
        self.feature_importance: dict[str, float] = {}
        # Weekdays the shop is shut. Those days are unknown (NaN) in training,
        # so during recursion they must be NaN too, not a predicted value.
        self.closed_days: list[int] = []

    def training_rows(self, history: pd.DataFrame) -> pd.DataFrame:
        feats = make_features(history)
        mask = feats["quantity"].notna() & (feats["days_since_first"] >= MIN_HISTORY_FOR_ROW)
        return feats.loc[mask]

    def fit(self, history: pd.DataFrame) -> "GlobalMLForecaster":
        rows = self.training_rows(history)
        if len(rows) < 50:
            raise ValueError(f"{self.name}: not enough training rows ({len(rows)}).")
        self.model = self.factory()
        self.model.fit(rows[FEATURES], rows["quantity"].to_numpy(dtype=float))
        self.feature_importance = self._importance()
        return self

    def _importance(self) -> dict[str, float]:
        est = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        if hasattr(est, "feature_importances_"):
            values = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            values = np.abs(np.asarray(est.coef_, dtype=float))
        else:
            return {}
        total = values.sum() or 1.0
        return dict(sorted(zip(FEATURES, (values / total).round(4).tolist()), key=lambda kv: -kv[1]))

    def forecast(self, history: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before forecasting.")
        tail = history[history["date"] >= start - pd.Timedelta(days=TAIL_DAYS)].copy()
        tail = tail[["product_key", "date", "quantity", "price", "first_date"]]
        last_price = history.dropna(subset=["price"]).sort_values("date").groupby("product_key")["price"].last()
        first_date = history.groupby("product_key")["first_date"].first()
        products = list(history["product_key"].unique())
        results = []
        for day in pd.date_range(start, periods=horizon, freq="D"):
            step = pd.DataFrame({"product_key": products, "date": day, "quantity": np.nan})
            step["price"] = step["product_key"].map(last_price)
            step["first_date"] = step["product_key"].map(first_date)
            tail = pd.concat([tail, step], ignore_index=True)
            feats = make_features(tail)
            today = feats[feats["date"] == day]
            yhat = np.clip(self.model.predict(today[FEATURES]), 0, None)
            pred = pd.Series(yhat, index=today["product_key"].to_numpy())
            mask = tail["date"] == day
            if day.dayofweek in self.closed_days:
                yhat = np.zeros_like(yhat)
                tail.loc[mask, "quantity"] = np.nan
            else:
                tail.loc[mask, "quantity"] = tail.loc[mask, "product_key"].map(pred).to_numpy()
            results.append(pd.DataFrame({"product_key": today["product_key"].to_numpy(), "date": day, "yhat": yhat}))
            # Keep the window short so each step stays fast.
            tail = tail[tail["date"] > day - pd.Timedelta(days=TAIL_DAYS)]
        out = pd.concat(results, ignore_index=True)
        frame = future_frame(products, start, horizon).merge(out, on=["product_key", "date"], how="left")
        return frame


def linear_regression() -> GlobalMLForecaster:
    return GlobalMLForecaster(
        "Linear Regression",
        lambda: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LinearRegression()),
        "Straight-line model on recent sales, weekday and season. The machine learning baseline.",
    )


def random_forest(random_state: int = 42) -> GlobalMLForecaster:
    return GlobalMLForecaster(
        "Random Forest",
        lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=random_state
        ),
        "Many decision trees averaged. Handles non-linear patterns and missing values.",
    )


def xgboost_model(random_state: int = 42) -> GlobalMLForecaster:
    from xgboost import XGBRegressor

    return GlobalMLForecaster(
        "XGBoost",
        lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0, objective="reg:squarederror",
            n_jobs=2, random_state=random_state, verbosity=0,
        ),
        "Gradient-boosted trees. Usually the strongest model when there is enough history.",
    )
