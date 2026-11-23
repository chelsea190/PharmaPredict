# Modelling decisions

This note explains each choice and why it was made. It is written so a reviewer can disagree with a specific rule rather than the whole system.

## 1. Data rules

| Situation | Rule | Why |
|---|---|---|
| No medicine sold on a day | Day is **unknown** (NaN) | Usually the shop was closed or data is missing. Calling it zero would teach the model false low demand. |
| Others sold, this medicine did not | Zero **only** if the user ticks "my file lists every sale"; otherwise unknown | Some exports leave out zero lines, some leave out whole products. We do not guess. |
| Before a medicine's first sale | Left out | It was not stocked yet. |
| Negative quantity | Treated as a return and left out of demand | A return is not negative demand. Counted in the report. |
| Missing or unreadable quantity | Row removed, not zero | Unknown is not zero. |
| Same invoice line twice | Removed when an invoice column exists | Double exports are common. |
| Same row twice, no invoice column | Kept by default, optional removal | Two identical sales on one day are possible. |
| Name variants | Grouped if they match after cleaning (case, spaces, units, `tabs` to `tablets`, common short forms) or are very close (92% similar) **and** have the same strength numbers | Strength must never be merged: 500mg and 1000mg are different products. |
| Brand vs generic | Never merged automatically. Shown as "look alike" for the user to alias | A wrong merge is worse than a missed one. |
| Shop shut on a weekday | If the shop was open on under 10% of that weekday in the last 12 weeks, forecasts for that weekday are zero | Avoids forecasting sales for Sundays when the shop is closed. |

## 2. Features (machine learning models)

Every feature uses only days **before** the day being predicted (all shifted by one day). A unit test changes future values and checks past features do not move.

* Lags: 1, 2, 3, 7, 14, 21, 28 days
* Rolling mean over 7, 14, 28 days; rolling std and max over 28 days
* Share of the last 28 days with a sale (spots slow sellers)
* Mean of the same weekday over the last 4 weeks
* 90-day level and ratios of recent demand to it (lets one global model serve fast and slow medicines)
* Weekday and day of year as sine/cosine, days since first sale, latest price

One model is trained across all medicines ("global"). This works better than one model each when most medicines have only a few hundred days.

Multi-day forecasts are made **recursively**: predict tomorrow, feed it back as a lag, predict the next day. This is the same way the model will be used for real, so validation reflects reality.

## 3. Models

| Model | Role |
|---|---|
| Seasonal naive (same weekday last week) | Floor. Anything worse is not worth using. |
| Moving average (28 days) | Simple, stable baseline. |
| Holt-Winters (damped trend, weekly season) | Classic per-medicine time series model. |
| Croston SBA | Built for medicines that sell irregularly. |
| Linear Regression | Simplest learned model. |
| Random Forest | Handles non-linear patterns, robust. |
| XGBoost (squared error) | Strongest general tabular model. |

XGBoost first used a Tweedie loss. It under-forecast by 8% on validation and 19% on test, so it was changed to squared error.

## 4. Splitting

```
|------------- train (rest) -------------|-- validation (30d) --|-- test (30d) --|
```

* Always by time, never random. Random splits leak future information.
* Windows shrink to 14 days for short files. At least 60 training days are needed or the run stops with a clear message.
* Models are fitted on train, scored on validation, and the best one is chosen.
* All models are then refitted on train + validation and scored once on test. This is the number to quote.
* The chosen model is refitted on all data for the real forecast.

## 5. How the best model is chosen

Daily WAPE (sum of absolute errors / sum of actual sales) was tried first. Daily errors in pharmacy data are dominated by noise, so every model scored about 57% and the metric rewarded forecasts that were too low.

What a pharmacist actually orders against is a **total** for the next week or month. So the selection score is **horizon WAPE**: the mean of the total-demand WAPE over the next 7, 14 and 30 days, per medicine, weighted by sales. Ties go to the simpler model.

MAE, RMSE, daily WAPE and bias are all still reported.

## 6. Result on synthetic data (24 medicines, 2 years)

| Model | Test daily WAPE | Test horizon WAPE | Bias |
|---|---|---|---|
| Random Forest | 56.6% | 22.7% | +3.2% |
| Linear Regression | 56.1% | 23.8% | -4.8% |
| XGBoost | 58.3% | 24.6% | +6.6% |
| **Holt-Winters (chosen on validation)** | 57.0% | 25.1% | +0.9% |
| Moving average | 56.9% | 26.0% | -1.7% |
| Croston SBA | 58.9% | 26.2% | -3.7% |
| Seasonal naive | 75.5% | 32.9% | -6.2% |

Notes:

* Holt-Winters was best on validation (15.5%) and is fourth on test. Random Forest would have been better on test, but picking it now would be choosing on the test set. Reporting this honestly is the point of a held-out test.
* All learned models beat the seasonal naive floor by a clear margin. The gap between the top five is small, which is normal for noisy daily retail data.
* These numbers come from made-up data and are not a claim about real-world accuracy.

## 7. Next steps if this goes to real use

1. Run on 6 to 12 months of real sales from one pharmacy and compare against what the pharmacist would have ordered.
2. Add stock on hand, so stock-out days are marked unknown instead of low demand.
3. Add prediction ranges (quantile models) so reorder points can include safety stock.
4. Retrain weekly and track error over time.
