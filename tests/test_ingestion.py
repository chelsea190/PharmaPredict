import numpy as np
import pandas as pd

from app.ingestion.columns import map_columns, normalise_header
from app.ingestion.names import group_names, normalise_name
from app.ingestion.parsing import parse_dates, parse_numbers
from app.ingestion.pipeline import IngestOptions, ingest


def test_column_variants_are_mapped():
    m = map_columns(["Invoice No", "Txn Date", "Medicine Name", "Qty Sold", "Unit Price (NGN)"])
    roles = {r: x.column for r, x in m.matches.items()}
    assert roles["date"] == "Txn Date"
    assert roles["product_name"] == "Medicine Name"
    assert roles["quantity"] == "Qty Sold"
    assert roles["price"] == "Unit Price (NGN)"
    assert not m.errors


def test_missing_required_column_is_an_error():
    m = map_columns(["Medicine", "Qty"])
    assert m.errors


def test_header_normalisation():
    assert normalise_header("  Unit Price (₦) ") == normalise_header("unit_price")


def test_day_first_dates_detected():
    r = parse_dates(pd.Series(["03/02/2025", "25/02/2025", "01/03/2025"]))
    assert r.dayfirst is True
    assert r.values.iloc[0] == pd.Timestamp("2025-02-03")


def test_numbers_with_currency_and_commas():
    out = parse_numbers(pd.Series(["₦1,200.50", "N/A", "(3)", "7"]))
    assert out.iloc[0] == 1200.5
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == -3
    assert out.iloc[3] == 7


def test_name_variants_group_but_strengths_stay_apart():
    names = pd.Series(["Paracetamol 500mg Tabs", "PARACETAMOL 500 MG TABLETS", "Paracetamol 1000mg Tablets"])
    raw_to_key, _ = group_names(names)
    assert raw_to_key["Paracetamol 500mg Tabs"] == raw_to_key["PARACETAMOL 500 MG TABLETS"]
    assert raw_to_key["Paracetamol 500mg Tabs"] != raw_to_key["Paracetamol 1000mg Tablets"]
    assert normalise_name("Vit C 100mg") == normalise_name("Vitamin C 100 mg")


def _raw():
    return pd.DataFrame({
        "Invoice No": ["A1", "A1", "A2", "A3", "A4", "A5"],
        "Date": ["01/03/2025", "01/03/2025", "02/03/2025", "02/03/2025", "xx", "03/03/2025"],
        "Medicine Name": ["Amoxil 500mg", "Amoxil 500mg", "Amoxil 500mg", "ORS sachet", "ORS sachet", "ORS sachet"],
        "Qty": ["2", "2", "-1", "4", "1", "N/A"],
        "Unit Price": ["100", "100", "100", "50", "50", "50"],
    })


def test_ingest_cleans_and_reports_issues():
    res = ingest(_raw(), today=pd.Timestamp("2025-04-01"))
    codes = {i["code"] for i in res.report.issues}
    assert {"duplicate_transactions", "returns", "bad_date", "bad_quantity"} <= codes
    assert len(res.clean) == 2               # one Amoxil row kept, one ORS row kept
    assert (res.clean["quantity"] > 0).all()


def test_missing_days_are_unknown_unless_confirmed():
    from app.features.series import build_daily_panel
    clean = ingest(_raw(), today=pd.Timestamp("2025-04-01")).clean
    unknown = build_daily_panel(clean, absent_days_are_zero=False)
    zero = build_daily_panel(clean, absent_days_are_zero=True)
    amox = lambda p: p[p["product_key"] == clean["product_key"].iloc[0]].set_index("date")["quantity"]
    # ORS sold on 2 March, so the shop was open; Amoxil had no sale that day.
    assert amox(zero).loc["2025-03-02"] == 0
    assert np.isnan(amox(unknown).loc["2025-03-02"])  # not confirmed -> unknown, never zero
    assert IngestOptions().absent_days_are_zero is False
