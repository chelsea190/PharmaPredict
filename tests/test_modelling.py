import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import all_metrics, mae, rmse, wape
from app.evaluation.splits import InsufficientDataError, make_split
from app.features.build import make_features
from app.features.series import build_daily_panel


def test_metrics():
    a, p = [10, 0, 5, np.nan], [8, 1, 5, 3]
    assert mae(a, p) == pytest.approx(1.0)
    assert rmse(a, p) == pytest.approx(np.sqrt(5 / 3))
    assert wape(a, p) == pytest.approx(3 / 15)
    assert set(all_metrics(a, p)) >= {"mae", "rmse", "wape", "bias"}


def test_split_is_chronological_and_non_overlapping():
    s = make_split(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), 30, 14, 60)
    assert s.train_end < s.val_start <= s.val_end < s.test_start <= s.test_end
    assert s.test_end == pd.Timestamp("2024-12-31")
    assert (s.val_end - s.val_start).days + 1 == s.window == 30


def test_short_data_shrinks_window_then_refuses():
    s = make_split(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-05"), 30, 14, 60)
    assert 14 <= s.window < 30
    with pytest.raises(InsufficientDataError):
        make_split(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-15"), 30, 14, 60)


def _panel(days=120):
    dates = pd.date_range("2024-01-01", periods=days)
    clean = pd.DataFrame({"date": dates, "product_key": "a", "product_name": "A", "product_id": None,
                          "quantity": np.arange(days, dtype=float) + 1, "price": 10.0, "row": range(days)})
    panel = build_daily_panel(clean, absent_days_are_zero=True)
    panel["first_date"] = panel["date"].min()
    return panel


def test_features_use_only_the_past():
    feats = make_features(_panel()).set_index("date")
    day = pd.Timestamp("2024-03-01")
    # quantity on day t is t+1; lag_1 must equal the previous day's quantity
    assert feats.loc[day, "lag_1"] == feats.loc[day - pd.Timedelta(days=1), "quantity"]
    assert feats.loc[day, "roll_mean_7"] == pytest.approx(
        feats.loc[day - pd.Timedelta(days=7): day - pd.Timedelta(days=1), "quantity"].mean())


def test_changing_the_future_does_not_change_past_features():
    p1 = _panel()
    p2 = p1.copy()
    p2.loc[p2["date"] >= "2024-03-15", "quantity"] = 9999
    cols = [c for c in make_features(p1).columns if c not in ("quantity", "date")]
    f1 = make_features(p1).set_index("date").loc[:"2024-03-15", cols]
    f2 = make_features(p2).set_index("date").loc[:"2024-03-15", cols]
    pd.testing.assert_frame_equal(f1, f2)
