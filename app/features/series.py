"""Build one daily demand series per medicine.

The key decision here is what an absent day means.

* A day on which **no medicine at all** has a record is treated as unknown
  (shop closed, public holiday, or data missing). It is never zero demand.
* A day on which other medicines sold but this one has no record is zero
  **only** if the user confirms the file lists every sale
  (``absent_days_are_zero``). Otherwise it is unknown (NaN).
* Before a medicine's first record it did not exist in the data, so those
  days are left out entirely.

Unknown days are excluded from training targets and from error metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_panel(clean: pd.DataFrame, absent_days_are_zero: bool) -> pd.DataFrame:
    """Return columns: product_key, date, quantity (NaN = unknown), price, open."""
    daily = clean.groupby(["product_key", "date"], as_index=False).agg(quantity=("quantity", "sum"))
    # Unit price for the day: revenue / units over priced lines (falls back to the
    # plain mean when every priced line had zero units).
    priced = clean.dropna(subset=["price"])
    if len(priced):
        day = (
            priced.assign(revenue=priced["quantity"] * priced["price"])
            .groupby(["product_key", "date"])
            .agg(revenue=("revenue", "sum"), units=("quantity", "sum"), mean_price=("price", "mean"))
        )
        day["price"] = (day["revenue"] / day["units"].where(day["units"] > 0)).fillna(day["mean_price"])
        daily = daily.merge(day[["price"]].reset_index(), on=["product_key", "date"], how="left")
    else:
        daily["price"] = np.nan

    start, end = clean["date"].min(), clean["date"].max()
    all_days = pd.date_range(start, end, freq="D")
    open_days = set(clean["date"].unique())
    first_seen = clean.groupby("product_key")["date"].min()

    frames = []
    for key, first in first_seen.items():
        days = all_days[all_days >= first]
        frame = pd.DataFrame({"product_key": key, "date": days})
        frames.append(frame)
    grid = pd.concat(frames, ignore_index=True)
    panel = grid.merge(daily, on=["product_key", "date"], how="left")
    panel["open"] = panel["date"].isin(open_days)

    missing = panel["quantity"].isna()
    if absent_days_are_zero:
        panel.loc[missing & panel["open"], "quantity"] = 0.0
    # Closed / no-data days stay NaN in both modes.
    panel.loc[~panel["open"], "quantity"] = np.nan
    return panel.sort_values(["product_key", "date"]).reset_index(drop=True)


def series_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-medicine facts used for eligibility and for choosing sensible models."""
    g = panel.groupby("product_key")
    observed = g["quantity"].count()
    positive = panel[panel["quantity"] > 0].groupby("product_key")["quantity"].count().reindex(observed.index, fill_value=0)
    profile = pd.DataFrame({
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "days": g["date"].count(),
        "observed_days": observed,
        "selling_days": positive,
        "total_units": g["quantity"].sum(),
        "mean_daily": g["quantity"].mean(),
    })
    profile["zero_share"] = 1 - profile["selling_days"] / profile["observed_days"].where(profile["observed_days"] > 0)
    # Average gap between selling days; > 1.32 is the usual threshold for "intermittent" demand.
    profile["adi"] = profile["observed_days"] / profile["selling_days"].where(profile["selling_days"] > 0)
    last_sale = panel[panel["quantity"] > 0].groupby("product_key")["date"].max().reindex(observed.index)
    profile["last_sale"] = last_sale
    return profile
