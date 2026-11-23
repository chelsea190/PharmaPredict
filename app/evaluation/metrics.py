"""Error metrics. Days with unknown actuals (NaN) are ignored, never counted as zero.

MAE   average size of the daily error, in units.
RMSE  like MAE but punishes big misses more.
WAPE  total absolute error / total actual units. Easy to read as a percentage
      and safe for medicines with many zero days (unlike MAPE, which divides by
      zero). Lower is better; 0.25 means the forecast is off by 25% of volume.
"""

from __future__ import annotations

import numpy as np


def _clean(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p)
    return a[mask], p[mask]


def mae(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(np.abs(a - p))) if len(a) else float("nan")


def rmse(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else float("nan")


def wape(actual, predicted) -> float:
    a, p = _clean(actual, predicted)
    denom = np.abs(a).sum()
    return float(np.abs(a - p).sum() / denom) if denom > 0 else float("nan")


def bias(actual, predicted) -> float:
    """Positive = over-forecasting (risk of overstock); negative = under (risk of stock-outs)."""
    a, p = _clean(actual, predicted)
    denom = np.abs(a).sum()
    return float((p - a).sum() / denom) if denom > 0 else float("nan")


def all_metrics(actual, predicted) -> dict[str, float]:
    a, _ = _clean(actual, predicted)
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "bias": bias(actual, predicted),
        "n": int(len(a)),
    }
