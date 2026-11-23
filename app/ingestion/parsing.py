"""Parse dates and numbers the way real pharmacy exports write them."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_SLASHED = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
_CURRENCY = re.compile(r"(ngn|naira|usd|gbp|eur|₦|\$|£|€|n(?=\s*\d))", re.IGNORECASE)


@dataclass
class DateParseResult:
    values: pd.Series          # datetime64 normalised to day, NaT where unparseable
    dayfirst: bool | None      # None when not relevant (ISO, Excel serials, real datetimes)
    ambiguous: bool            # True when the order could not be proven from the data
    conflicting: bool          # True when some rows prove day-first and others month-first


def detect_dayfirst(raw: pd.Series) -> tuple[bool | None, bool, bool]:
    """Decide day/month order from values like 03/04/2024.

    A first part above 12 proves day-first; a second part above 12 proves
    month-first. Returns (dayfirst, ambiguous, conflicting).
    """
    first_gt_12 = second_gt_12 = False
    seen = False
    for value in raw.dropna().astype(str).head(20000):
        match = _SLASHED.match(value)
        if not match:
            continue
        seen = True
        a, b = int(match.group(1)), int(match.group(2))
        first_gt_12 |= a > 12
        second_gt_12 |= b > 12
    if not seen:
        return None, False, False
    if first_gt_12 and second_gt_12:
        return True, False, True
    if first_gt_12:
        return True, False, False
    if second_gt_12:
        return False, False, False
    return True, True, False  # cannot tell: assume day-first and warn


def parse_dates(raw: pd.Series, hint: str = "auto") -> DateParseResult:
    if pd.api.types.is_datetime64_any_dtype(raw):
        return DateParseResult(pd.to_datetime(raw).dt.normalize(), None, False, False)

    as_text = raw.map(lambda v: None if pd.isna(v) else str(v).strip()).astype(object)
    numeric = pd.to_numeric(as_text, errors="coerce").astype(float)
    is_serial = numeric.between(20000, 80000).fillna(False).astype(bool)  # Excel serials (1954 to 2119)

    if hint == "dayfirst":
        dayfirst, ambiguous, conflicting = True, False, False
    elif hint == "monthfirst":
        dayfirst, ambiguous, conflicting = False, False, False
    else:
        dayfirst, ambiguous, conflicting = detect_dayfirst(as_text[~is_serial])

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    text_mask = ~is_serial & as_text.notna()
    if text_mask.any():
        parsed[text_mask] = pd.to_datetime(as_text[text_mask], errors="coerce", dayfirst=bool(dayfirst), format="mixed").astype("datetime64[ns]")
    if is_serial.any():
        parsed[is_serial] = (pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric[is_serial], unit="D")).astype("datetime64[ns]")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return DateParseResult(parsed.dt.normalize(), dayfirst, ambiguous, conflicting)


def parse_numbers(raw: pd.Series) -> pd.Series:
    """Turn '₦1,250.00', 'N 300', '(12)', ' 4 ' into floats. Unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(raw):
        return raw.astype(float)
    text = raw.astype("string").str.strip()
    negative = text.str.match(r"^\(.*\)$", na=False)
    text = text.str.replace(_CURRENCY, "", regex=True)
    text = text.str.replace(r"[,\s()]", "", regex=True)
    values = pd.to_numeric(text, errors="coerce").astype(float)
    values[negative.fillna(False).to_numpy()] *= -1
    return values.replace([np.inf, -np.inf], np.nan)
