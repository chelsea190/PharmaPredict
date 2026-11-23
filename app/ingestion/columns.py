"""Map the columns of an uploaded file to the roles the system needs.

Roles
-----
date            required  when the sale happened
quantity        required  units sold on that line
product_name    one of product_name / product_id is required
product_id      optional  a stable code (SKU, drug id)
price           optional  selling price per unit
total_amount    optional  line total; used to derive price when price is missing
transaction_id  optional  invoice / receipt number, used for duplicate detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

ROLES: dict[str, list[str]] = {
    "product_name": [
        "product_name", "medicine_name", "medicine", "drug_name", "drug", "item_name", "item",
        "product", "product_description", "item_description", "description", "brand_name", "medication",
    ],
    "product_id": [
        "product_id", "drug_id", "medicine_id", "item_id", "sku", "product_code", "item_code",
        "drug_code", "barcode", "upc",
    ],
    "date": [
        "date", "transaction_date", "sale_date", "sales_date", "invoice_date", "order_date",
        "txn_date", "datetime", "timestamp", "date_sold", "sold_on", "dispense_date",
    ],
    "quantity": [
        "quantity", "qty", "units_sold", "units", "quantity_sold", "qty_sold", "sold_qty",
        "unit_sold", "number_sold", "volume", "packs_sold", "quantity_dispensed",
    ],
    "price": [
        "price", "selling_price", "unit_price", "sale_price", "price_per_unit", "retail_price",
        "rate", "unit_cost_to_customer",
    ],
    "total_amount": [
        "total_amount", "amount", "line_total", "total", "sales_amount", "revenue", "total_price",
        "net_amount", "value",
    ],
    "transaction_id": [
        "transaction_id", "invoice_id", "invoice_no", "invoice_number", "receipt_no", "receipt_number",
        "order_id", "txn_id", "sale_id", "bill_no", "transaction_no",
    ],
}

REQUIRED = ("date", "quantity")
FUZZY_THRESHOLD = 0.86


_CURRENCY_WORDS = r"\b(ngn|naira|usd|gbp|eur|n|in)\b"


def normalise_header(name: object) -> str:
    """'Unit Price (NGN)' -> 'unit_price', 'Qty. Sold' -> 'qty_sold'."""
    text = str(name).strip().lower().replace("₦", " ")
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(_CURRENCY_WORDS, " ", text)
    return re.sub(r"\s+", "_", text.strip())


@dataclass
class ColumnMatch:
    role: str
    column: str
    method: str  # "exact", "fuzzy" or "user"
    score: float


@dataclass
class ColumnMapping:
    matches: dict[str, ColumnMatch] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def column_for(self, role: str) -> str | None:
        match = self.matches.get(role)
        return match.column if match else None

    def as_dict(self) -> dict:
        return {
            "matches": {r: {"column": m.column, "method": m.method, "score": round(m.score, 2)} for r, m in self.matches.items()},
            "unmapped_columns": self.unmapped,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _best_role(header: str) -> tuple[str | None, str, float]:
    for role, names in ROLES.items():
        if header in names:
            return role, "exact", 1.0
    best: tuple[str | None, float] = (None, 0.0)
    for role, names in ROLES.items():
        for name in names:
            score = SequenceMatcher(None, header, name).ratio()
            if score > best[1]:
                best = (role, score)
    if best[1] >= FUZZY_THRESHOLD:
        return best[0], "fuzzy", best[1]
    return None, "none", best[1]


def map_columns(columns: list[object], user_mapping: dict[str, str] | None = None) -> ColumnMapping:
    """Work out which uploaded column plays which role.

    ``user_mapping`` maps role -> column name and always wins. Automatic matches
    prefer exact synonym hits over fuzzy ones, and never assign two columns to
    the same role silently.
    """
    mapping = ColumnMapping()
    originals = [str(c) for c in columns]
    taken: set[str] = set()

    for role, column in (user_mapping or {}).items():
        if role not in ROLES:
            mapping.errors.append(f"Unknown role '{role}' in the column mapping.")
            continue
        if column not in originals:
            mapping.errors.append(f"Column '{column}' (given for {role}) is not in the file.")
            continue
        mapping.matches[role] = ColumnMatch(role, column, "user", 1.0)
        taken.add(column)

    candidates: dict[str, list[ColumnMatch]] = {}
    for column in originals:
        if column in taken:
            continue
        role, method, score = _best_role(normalise_header(column))
        if role and role not in mapping.matches:
            candidates.setdefault(role, []).append(ColumnMatch(role, column, method, score))
        else:
            mapping.unmapped.append(column)

    for role, options in candidates.items():
        options.sort(key=lambda m: (m.method != "exact", -m.score))
        chosen = options[0]
        mapping.matches[role] = chosen
        for other in options[1:]:
            mapping.unmapped.append(other.column)
            mapping.warnings.append(
                f"Both '{chosen.column}' and '{other.column}' look like {role}. Using '{chosen.column}'. "
                "Set the mapping by hand if that is wrong."
            )
        if chosen.method == "fuzzy":
            mapping.warnings.append(f"Guessed that '{chosen.column}' is {role} (similarity {chosen.score:.2f}). Please confirm.")

    for role in REQUIRED:
        if role not in mapping.matches:
            mapping.errors.append(f"No column found for {role}. Expected something like: {', '.join(ROLES[role][:4])}.")
    if "product_name" not in mapping.matches and "product_id" not in mapping.matches:
        mapping.errors.append("No product column found. Expected a medicine name or a product ID column.")
    return mapping
