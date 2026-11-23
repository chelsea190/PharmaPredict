import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.storage.repository import SQLiteRepository
from app.synthetic.generator import generate


def _client():
    return TestClient(create_app(SQLiteRepository(":memory:"), run_inline=True))


def test_upload_train_forecast_export():
    c = _client()
    csv = generate(days=240, products=6, seed=3).to_csv(index=False).encode()
    ds = c.post("/api/datasets", files={"file": ("SYNTHETIC_test.csv", csv, "text/csv")},
                data={"absent_days_are_zero": "true"}).json()
    assert ds["is_synthetic"] is True
    run = c.post(f"/api/datasets/{ds['id']}/runs", json={"models": ["seasonal_naive", "moving_average", "linear_regression"]})
    assert run.status_code == 202
    r = c.get(f"/api/runs/{run.json()['id']}").json()
    assert r["status"] == "completed", r.get("error")
    res = r["result"]
    test = [m for m in res["model_metrics"] if m["stage"] == "test"]
    assert any(m["selected"] for m in test)
    for p in res["products"]:
        if p["status"] == "forecast":
            assert p["forecast_7d"] <= p["forecast_14d"] <= p["forecast_30d"]
    # forecasts start the day after the data ends
    assert pd.Timestamp(res["split"]["forecast_start"]) == pd.Timestamp(res["split"]["test_end"]) + pd.Timedelta(days=1)
    csv_out = c.get(f"/api/runs/{r['id']}/export/forecasts.csv").text
    assert "SYNTHETIC" in csv_out.splitlines()[0].upper() or "synthetic" in csv_out.lower()


def test_bad_upload_gives_clear_error():
    c = _client()
    r = c.post("/api/datasets", files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 422
    assert "column" in r.text.lower()


def test_too_short_dataset_fails_with_message():
    c = _client()
    csv = generate(days=50, products=3, seed=1).to_csv(index=False).encode()
    ds = c.post("/api/datasets", files={"file": ("short.csv", csv, "text/csv")}).json()
    run = c.post(f"/api/datasets/{ds['id']}/runs", json={}).json()
    r = c.get(f"/api/runs/{run['id']}").json()
    assert r["status"] == "failed"
    assert "days" in r["error"]
