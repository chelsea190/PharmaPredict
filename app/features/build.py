"""Features for the machine learning models.

Leakage rule: every feature for day t is computed from data up to day t-1.
All rolling windows run on the series shifted by one day. Calendar features
(weekday, month) are known in advance and are safe. Price uses the last price
seen before day t, because tomorrow's price is not known when forecasting.

The longest look-back is 90 days, so a 100-day tail of history is enough to
rebuild the exact same features during recursive forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 21, 28)
WINDOWS = (7, 14, 28)
LEVEL_WINDOW = 90
TAIL_DAYS = 100

FEATURES: list[str] = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + ["roll_std_28", "roll_max_28", "selling_share_28", "same_weekday_mean", "level_90",
       "lag_1_vs_level", "roll_7_vs_level",
       "dow", "dow_sin", "dow_cos", "month", "doy_sin", "doy_cos", "day_of_month",
       "days_since_first", "price_last", "price_vs_median"]
)


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to a panel sorted by product_key, date.

    Expects columns product_key, date, quantity (NaN allowed), price (NaN allowed),
    first_date.
    """
    df = panel.sort_values(["product_key", "date"]).copy()
    g = df.groupby("product_key", sort=False)
    past = g["quantity"].shift(1)
    df["_past"] = past
    gp = df.groupby("product_key", sort=False)["_past"]

    for k in LAGS:
        df[f"lag_{k}"] = g["quantity"].shift(k)
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = gp.transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 3)).mean())
    df["roll_std_28"] = gp.transform(lambda s: s.rolling(28, min_periods=7).std())
    df["roll_max_28"] = gp.transform(lambda s: s.rolling(28, min_periods=7).max())
    df["selling_share_28"] = gp.transform(lambda s: (s > 0).where(s.notna()).rolling(28, min_periods=7).mean())
    df["same_weekday_mean"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1, skipna=True)
    df["level_90"] = gp.transform(lambda s: s.rolling(LEVEL_WINDOW, min_periods=7).mean())
    level = df["level_90"].where(df["level_90"] > 0)
    df["lag_1_vs_level"] = df["lag_1"] / level
    df["roll_7_vs_level"] = df["roll_mean_7"] / level

    dow = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["dow"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["day_of_month"] = df["date"].dt.day
    df["days_since_first"] = (df["date"] - df["first_date"]).dt.days

    past_price = g["price"].shift(1)
    df["price_last"] = past_price.groupby(df["product_key"], sort=False).ffill()
    median_price = past_price.groupby(df["product_key"], sort=False).transform(
        lambda s: s.rolling(LEVEL_WINDOW, min_periods=1).median()
    )
    median_price = median_price.groupby(df["product_key"], sort=False).ffill()
    df["price_vs_median"] = df["price_last"] / median_price.where(median_price > 0)

    df = df.drop(columns=["_past"])
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df
