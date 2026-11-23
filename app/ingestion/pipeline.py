"""Turn a raw upload into clean sales lines plus a validation report.

Clean output columns
--------------------
date          datetime64, day precision
product_key   stable key used everywhere else
product_name  most common original spelling, for display
product_id    original id if the file had one, else None
quantity      units sold (float, >= 0). Returns are excluded and reported.
price         price per unit (float) or NaN when unknown
row           original row number (1-based, header excluded) for traceability
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .columns import ColumnMapping, map_columns
from .names import display_names, group_names, normalise_name, possible_duplicates
from .parsing import parse_dates, parse_numbers
from .reader import DatasetError


@dataclass
class IngestOptions:
    date_format: str = "auto"              # auto | dayfirst | monthfirst
    drop_exact_duplicates: bool = False    # only used when there is no transaction id
    absent_days_are_zero: bool = False     # the user confirms the file lists every sale
    column_mapping: dict[str, str] | None = None
    name_aliases: dict[str, str] | None = None  # name as written -> name it should count as


@dataclass
class ValidationReport:
    rows_in: int = 0
    rows_kept: int = 0
    columns: dict = field(default_factory=dict)
    date_range: dict = field(default_factory=dict)
    date_order: str | None = None
    products: int = 0
    total_units: float = 0.0
    issues: list[dict] = field(default_factory=list)   # {level, code, message, count?, examples?}
    name_merges: list[dict] = field(default_factory=list)
    possible_duplicates: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, count: int | None = None, examples: list | None = None) -> None:
        item = {"level": level, "code": code, "message": message}
        if count is not None:
            item["count"] = int(count)
        if examples:
            item["examples"] = [str(e) for e in examples[:5]]
        self.issues.append(item)

    @property
    def has_errors(self) -> bool:
        return any(i["level"] == "error" for i in self.issues)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestResult:
    clean: pd.DataFrame
    report: ValidationReport
    mapping: ColumnMapping


def _examples(raw: pd.DataFrame, mask: pd.Series, column: str | None) -> list:
    if column is None or not mask.any():
        return []
    values = raw.loc[mask, column].head(5).tolist()
    return ["(blank)" if pd.isna(v) or str(v).strip() == "" else str(v) for v in values]


def ingest(raw: pd.DataFrame, options: IngestOptions | None = None, today: datetime | None = None) -> IngestResult:
    options = options or IngestOptions()
    report = ValidationReport(rows_in=len(raw), options=asdict(options))
    mapping = map_columns(list(raw.columns), options.column_mapping)
    report.columns = mapping.as_dict()
    for message in mapping.errors:
        report.add("error", "columns", message)
    for message in mapping.warnings:
        report.add("warning", "columns", message)
    if mapping.errors:
        raise DatasetError("; ".join(mapping.errors))

    col = mapping.column_for
    out = pd.DataFrame(index=raw.index)
    out["row"] = raw.index + 1

    # ---- dates ---------------------------------------------------------
    dates = parse_dates(raw[col("date")], options.date_format)
    out["date"] = dates.values
    if dates.dayfirst is not None:
        report.date_order = "day/month/year" if dates.dayfirst else "month/day/year"
    if dates.conflicting:
        report.add("warning", "date_order_conflict",
                   "Some dates are day-first and others month-first. Day-first was used. Check the source system.")
    elif dates.ambiguous:
        report.add("warning", "date_order_assumed",
                   "Dates like 03/04/2024 could be 3 April or 4 March. Assumed day-first (dd/mm/yyyy). "
                   "Re-upload with date format set to month-first if that is wrong.")
    bad_date = out["date"].isna()
    if bad_date.any():
        report.add("warning", "bad_date", "Rows with a date that could not be read were removed.",
                   bad_date.sum(), _examples(raw, bad_date, col("date")))
    today = pd.Timestamp(today or datetime.now()).normalize()
    future = out["date"] > today
    if future.any():
        report.add("warning", "future_date", "Rows dated in the future were removed.", future.sum(),
                   _examples(raw, future, col("date")))

    # ---- quantity ------------------------------------------------------
    out["quantity"] = parse_numbers(raw[col("quantity")])
    bad_qty = out["quantity"].isna()
    if bad_qty.any():
        report.add("warning", "bad_quantity", "Rows with a missing or unreadable quantity were removed "
                   "(not treated as zero).", bad_qty.sum(), _examples(raw, bad_qty, col("quantity")))
    returns = out["quantity"] < 0
    if returns.any():
        report.add("info", "returns",
                   f"{int(returns.sum())} rows have negative quantities (returns, {abs(out.loc[returns, 'quantity'].sum()):,.0f} units). "
                   "They are left out of demand, because a return is not negative demand.", returns.sum())
    fractional = (out["quantity"] % 1 != 0) & out["quantity"].notna()
    if fractional.any():
        report.add("info", "fractional_quantity", "Some quantities are not whole numbers (for example part packs). Kept as they are.",
                   fractional.sum(), _examples(raw, fractional, col("quantity")))

    # ---- price ---------------------------------------------------------
    price = parse_numbers(raw[col("price")]) if col("price") else pd.Series(np.nan, index=raw.index)
    if col("total_amount"):
        total = parse_numbers(raw[col("total_amount")])
        derive = price.isna() & total.notna() & (out["quantity"] > 0)
        price = price.where(~derive, total / out["quantity"])
        if derive.any() and col("price"):
            report.add("info", "price_from_total", "Missing unit prices were worked out from the line total.", derive.sum())
    bad_price = price.notna() & (price <= 0)
    if bad_price.any():
        report.add("warning", "bad_price", "Zero or negative prices were treated as unknown.", bad_price.sum(),
                   _examples(raw, bad_price, col("price")))
    price = price.where(~bad_price)
    if not col("price") and not col("total_amount"):
        report.add("info", "no_price", "The file has no price column. Revenue figures will not be available.")
    elif price.isna().mean() > 0.2:
        report.add("warning", "missing_price", f"{price.isna().mean():.0%} of rows have no usable price.")
    out["price"] = price

    # ---- product -------------------------------------------------------
    name_col, id_col = col("product_name"), col("product_id")
    names = raw[name_col].astype("string").str.strip() if name_col else None
    ids = raw[id_col].astype("string").str.strip() if id_col else None
    missing_product = pd.Series(False, index=raw.index)
    if ids is not None:
        missing_product |= ids.isna() | (ids == "")
    if names is not None and ids is None:
        missing_product |= names.isna() | (names == "")
    if missing_product.any():
        report.add("warning", "missing_product", "Rows with no product were removed.", missing_product.sum())

    keep = ~(bad_date | future | bad_qty | returns | missing_product)
    out, raw_kept = out[keep].copy(), raw[keep]
    names = names[keep] if names is not None else None
    ids = ids[keep] if ids is not None else None

    if ids is not None:
        out["product_key"] = "id:" + ids.map(lambda v: normalise_name(v))
        out["product_id"] = ids
        if names is not None:
            label_map = display_names(names.fillna(ids), out["product_key"])
            per_id = pd.DataFrame({"k": out["product_key"], "n": names.map(normalise_name, na_action="ignore")}).dropna()
            multi = per_id.groupby("k")["n"].nunique()
            multi = multi[multi > 1]
            if len(multi):
                report.add("warning", "id_many_names", "Some product IDs appear with more than one name. The ID was trusted.",
                           len(multi), [k.removeprefix("id:") for k in multi.index[:5]])
            per_name = per_id.groupby("n")["k"].nunique()
            per_name = per_name[per_name > 1]
            if len(per_name):
                report.add("warning", "name_many_ids", "Some medicine names appear under more than one product ID. They were kept apart.",
                           len(per_name), list(per_name.index[:5]))
        else:
            label_map = {k: k.removeprefix("id:") for k in out["product_key"].unique()}
    else:
        raw_to_key, merges = group_names(names, options.name_aliases)
        out["product_key"] = names.map(raw_to_key)
        out["product_id"] = None
        label_map = display_names(names, out["product_key"])
        report.name_merges = merges
        report.possible_duplicates = possible_duplicates(list(set(raw_to_key.values())))
        if report.possible_duplicates:
            report.add("warning", "possible_duplicates",
                       f"{len(report.possible_duplicates)} pairs of medicines look alike but were kept apart (for example a brand "
                       "and a generic). Review them and add aliases if they are really the same product.",
                       len(report.possible_duplicates))
        if merges:
            report.add("info", "name_merges",
                       f"{len(merges)} medicines were written in more than one way and have been grouped. Review the list.",
                       len(merges))
    out["product_name"] = out["product_key"].map(label_map)

    # ---- duplicates ----------------------------------------------------
    tx_col = col("transaction_id")
    if tx_col:
        tx = raw_kept[tx_col].astype("string").str.strip()
        dup = tx.notna() & pd.DataFrame({"tx": tx, "k": out["product_key"], "q": out["quantity"], "d": out["date"]}).duplicated()
        if dup.any():
            report.add("warning", "duplicate_transactions",
                       "Repeated lines (same invoice, product, quantity and date) were removed.", dup.sum(), tx[dup].head(5).tolist())
            out = out[~dup]
    else:
        exact = raw_kept.duplicated(keep="first")
        if exact.any():
            if options.drop_exact_duplicates:
                report.add("warning", "exact_duplicates_removed", "Exact duplicate rows were removed, as requested.", exact.sum())
                out = out[~exact.to_numpy()]
            else:
                report.add("warning", "exact_duplicates_kept",
                           "Some rows are exact copies of another row. They were kept, because two identical sales on one day "
                           "can be real. Turn on 'remove exact duplicates' if your system double-exports lines.", exact.sum())

    out = out.sort_values(["date", "product_key"]).reset_index(drop=True)
    out = out[["date", "product_key", "product_name", "product_id", "quantity", "price", "row"]]

    if out.empty:
        report.add("error", "no_rows", "No usable rows were left after cleaning.")
        raise DatasetError("No usable rows were left after cleaning. See the validation report.")

    # ---- summary and coverage -----------------------------------------
    report.rows_kept = len(out)
    report.products = int(out["product_key"].nunique())
    report.total_units = float(out["quantity"].sum())
    first, last = out["date"].min(), out["date"].max()
    span = (last - first).days + 1
    report.date_range = {"start": first.date().isoformat(), "end": last.date().isoformat(), "days": span}
    days_with_sales = out["date"].nunique()
    explicit_zero_rows = int((out["quantity"] == 0).sum())
    per_product = out.groupby("product_key")["date"].agg(["min", "nunique"])
    active_days = (last - per_product["min"]).dt.days + 1
    report.coverage = {
        "days_in_range": span,
        "days_with_any_record": int(days_with_sales),
        "days_with_no_records": int(span - days_with_sales),
        "explicit_zero_rows": explicit_zero_rows,
        "median_product_coverage": float(np.median(per_product["nunique"] / active_days)),
        "looks_like_transaction_log": bool(explicit_zero_rows == 0 and out.duplicated(["date", "product_key"]).any()),
    }
    if span < 90:
        report.add("warning", "short_history",
                   f"Only {span} days of history. Forecasts need about 4 months or more to be evaluated properly.")
    if span - days_with_sales > 0:
        report.add("info", "closed_days",
                   f"{span - days_with_sales} days in the range have no sales at all for any medicine. They are treated as "
                   "'shop closed or data missing', not as zero demand.", span - days_with_sales)
    if not options.absent_days_are_zero:
        report.add("info", "absent_days_unknown",
                   "Days when a medicine has no record are treated as unknown, not as zero sales. If your file lists every "
                   "sale (for example a till export), turn on 'my file lists every sale' for better forecasts.")
    return IngestResult(out, report, mapping)
