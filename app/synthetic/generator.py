"""SYNTHETIC demo data. Not real pharmacy sales.

Generates a till-style transaction export for an imaginary Lagos community
pharmacy, so the whole workflow can be shown without real patient or business
data. Every dataset made here is stored with is_synthetic = True and shown with
a "Synthetic demo data" banner. Forecast accuracy on this data says nothing
about accuracy on real pharmacy data.

Patterns built in (so the models have something real to find):
* weekly rhythm (busier Monday and Saturday, closed on Sundays and public holidays)
* rainy-season malaria and ORS peaks (June to October)
* harmattan cough, allergy and asthma peaks (December to February)
* month-start bump for chronic medicines (salary week)
* a price rise part-way through with a small drop in demand
* stock-outs, a newly launched product and a discontinued product
* intermittent sellers (insulin, injections)

Data quality problems are added on purpose so the cleaning step can be seen
working: name variants, day-first text dates, currency symbols, a few missing
prices, unreadable quantities, returns, and double-exported invoice lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SYNTHETIC_FILENAME = "SYNTHETIC_demo_pharmacy_sales.csv"


@dataclass(frozen=True)
class Medicine:
    name: str
    variants: tuple[str, ...]
    daily_mean: float
    price: float
    season: str = "flat"         # flat | rainy | harmattan
    chronic: bool = False        # month-start bump
    trend: float = 0.0           # growth per year (0.1 = +10%)
    dispersion: float = 4.0      # lower = more irregular
    launch_day: int = 0          # days after start
    stop_day: int | None = None  # days after start


CATALOGUE = (
    Medicine("Paracetamol 500mg Tablets", ("paracetamol 500 mg tabs", "PARACETAMOL 500MG TAB"), 26, 400),
    Medicine("Artemether/Lumefantrine 20/120mg Tablets", ("artemether lumefantrine 20/120mg tab", "ARTEMETHER/LUMEFANTRINE 20/120 MG TABS"), 9, 2500, "rainy"),
    Medicine("Amoxicillin 500mg Capsules", ("amoxicillin 500mg caps", "AMOXICILLIN 500 MG CAP"), 7, 1800, "harmattan", dispersion=3),
    Medicine("Oral Rehydration Salts Sachet", ("ORS sachet",), 6, 300, "rainy"),
    Medicine("Benylin Cough Syrup 100ml", ("benylin cough syr 100ml",), 4, 2200, "harmattan"),
    Medicine("Loratadine 10mg Tablets", ("loratadine 10 mg tabs",), 3, 900, "harmattan", dispersion=2.5),
    Medicine("Vitamin C 1000mg Tablets", ("vit c 1000mg tabs", "Vitamin C 1000 mg Tablets"), 8, 1500, trend=0.25),
    Medicine("Metformin 500mg Tablets", ("metformin 500 mg tablets",), 6, 1200, chronic=True),
    Medicine("Amlodipine 5mg Tablets", ("amlodipine 5 mg tabs",), 5, 1000, chronic=True),
    Medicine("Ibuprofen 400mg Tablets", ("ibuprofen 400 mg tab",), 9, 600),
    Medicine("Omeprazole 20mg Capsules", ("omeprazole 20 mg caps",), 5, 1300),
    Medicine("Ciprofloxacin 500mg Tablets", ("cipro 500mg tabs",), 3, 1600, dispersion=2.5),
    Medicine("Folic Acid 5mg Tablets", (), 4, 350, chronic=True),
    Medicine("Ferrous Sulphate 200mg Tablets", (), 3, 450, dispersion=3),
    Medicine("Salbutamol 100mcg Inhaler", ("salbutamol 100 mcg inhaler",), 0.9, 3500, "harmattan", dispersion=1.2),
    Medicine("Mixtard 30/70 Insulin 10ml", (), 0.4, 9000, chronic=True, dispersion=0.8),
    Medicine("Ceftriaxone 1g Injection", ("ceftriaxone 1 g inj",), 0.7, 1800, dispersion=0.9),
    Medicine("Fluconazole 150mg Capsules", (), 1.2, 700, dispersion=1.5),
    Medicine("Diclofenac Gel 50g", (), 1.5, 1400, dispersion=2),
    Medicine("Zinc Sulphate 20mg Tablets", (), 2.5, 500, "rainy", dispersion=2.5),
    Medicine("Multivitamin Syrup 200ml", (), 2.0, 1900, trend=0.1, dispersion=2),
    Medicine("Hydrochlorothiazide 25mg Tablets", (), 2.2, 600, chronic=True),
    Medicine("Vitamin D3 1000IU Capsules", (), 3.5, 2800, launch_day=-50),   # launched 50 days before the end
    Medicine("Chloroquine Phosphate 250mg Tablets", (), 2.0, 500, "rainy", stop_day=-90),  # discontinued
)

NG_HOLIDAYS = ("01-01", "05-01", "06-12", "10-01", "12-25", "12-26")


def _season_factor(kind: str, day_of_year: np.ndarray) -> np.ndarray:
    if kind == "rainy":     # peak mid-August
        return 1 + 0.7 * np.exp(-0.5 * ((day_of_year - 225) / 40) ** 2)
    if kind == "harmattan":  # peak early January (wraps around the year end)
        dist = np.minimum(np.abs(day_of_year - 5), 365 - np.abs(day_of_year - 5))
        return 1 + 0.8 * np.exp(-0.5 * (dist / 30) ** 2)
    return np.ones_like(day_of_year, dtype=float)


def generate(days: int = 730, end: str | pd.Timestamp | None = None, seed: int = 7,
             products: int | None = None, inject_issues: bool = True) -> pd.DataFrame:
    """Return a SYNTHETIC transaction table with deliberately messy columns."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    is_holiday = dates.strftime("%m-%d").isin(NG_HOLIDAYS)
    open_day = (dates.dayofweek != 6) & ~is_holiday
    doy = dates.dayofyear.to_numpy()
    weekday = np.array([1.15, 1.0, 0.95, 0.95, 1.05, 1.2, 0.0])[dates.dayofweek]
    price_rise_day = int(days * 0.55)

    rows = []
    invoice = 100000
    catalogue = CATALOGUE[: products or len(CATALOGUE)]
    for med in catalogue:
        launch = med.launch_day if med.launch_day >= 0 else days + med.launch_day
        stop = days if med.stop_day is None else days + med.stop_day
        t = np.arange(days) / 365
        mean = (
            med.daily_mean * weekday * _season_factor(med.season, doy) * (1 + med.trend * t)
            * np.where(med.chronic & (dates.day <= 5), 1.35, 1.0)
            * np.where(np.arange(days) >= price_rise_day, 0.9, 1.0)
        )
        # Negative binomial (gamma-Poisson) gives realistic over-dispersed counts.
        lam = rng.gamma(med.dispersion, mean / med.dispersion)
        units = rng.poisson(lam)
        # Stock-outs: a few short runs of zero sales.
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, days - 7)
            units[s: s + rng.integers(3, 8)] = 0
        active = (np.arange(days) >= launch) & (np.arange(days) < stop) & open_day
        price = np.where(np.arange(days) >= price_rise_day, med.price * 1.25, med.price)
        for i in np.flatnonzero(active & (units > 0)):
            remaining = int(units[i])
            while remaining > 0:
                q = int(min(remaining, rng.choice([1, 1, 1, 2, 2, 3, 4])))
                remaining -= q
                invoice += 1
                rows.append((f"INV-{invoice}", dates[i], med.name, med.variants, q, price[i]))

    frame = pd.DataFrame(rows, columns=["invoice", "date", "name", "variants", "qty", "price"])
    out = pd.DataFrame({
        "Invoice No": frame["invoice"],
        "Date": frame["date"].dt.strftime("%d/%m/%Y"),
        "Medicine Name": frame["name"],
        "Qty": frame["qty"].astype(str),
        "Unit Price (NGN)": frame["price"].map(lambda p: f"₦{p:,.0f}"),
    })
    if not inject_issues:
        return out.sample(frac=1, random_state=seed).sort_values("Date", kind="stable").reset_index(drop=True)

    n = len(out)
    has_variant = frame["variants"].map(len) > 0
    swap = has_variant & (rng.random(n) < 0.15)
    out.loc[swap, "Medicine Name"] = [rng.choice(v) for v in frame.loc[swap, "variants"]]
    out.loc[rng.random(n) < 0.004, "Unit Price (NGN)"] = ""
    out.loc[rng.random(n) < 0.0005, "Qty"] = "N/A"
    returns = out.sample(frac=0.002, random_state=seed + 1).assign(Qty="-1")
    doubles = out.sample(frac=0.003, random_state=seed + 2)
    out = pd.concat([out, returns, doubles], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
