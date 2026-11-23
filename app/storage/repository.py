"""Storage layer.

The rest of the app only talks to the ``Repository`` interface below. The
SQLite version is for the local prototype. A Supabase (Postgres) version can
implement the same methods later without touching the API or the models.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    return json.dumps(obj, default=default)


class NotFound(KeyError):
    pass


class Repository(ABC):
    @abstractmethod
    def create_dataset(self, name: str, clean: pd.DataFrame, report: dict, options: dict, is_synthetic: bool) -> str: ...
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> dict: ...
    @abstractmethod
    def list_datasets(self) -> list[dict]: ...
    @abstractmethod
    def load_clean(self, dataset_id: str) -> pd.DataFrame: ...
    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> None: ...
    @abstractmethod
    def create_run(self, dataset_id: str, config: dict) -> str: ...
    @abstractmethod
    def update_run(self, run_id: str, **fields) -> None: ...
    @abstractmethod
    def save_run_result(self, run_id: str, result: dict, forecasts: pd.DataFrame) -> None: ...
    @abstractmethod
    def get_run(self, run_id: str) -> dict: ...
    @abstractmethod
    def list_runs(self, dataset_id: str | None = None) -> list[dict]: ...
    @abstractmethod
    def load_forecasts(self, run_id: str) -> pd.DataFrame: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    rows INTEGER NOT NULL,
    products INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    options_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sales (
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    product_key TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_id TEXT,
    quantity REAL NOT NULL,
    price REAL,
    source_row INTEGER
);
CREATE INDEX IF NOT EXISTS ix_sales_dataset ON sales(dataset_id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    product_key TEXT NOT NULL,
    product_name TEXT,
    date TEXT NOT NULL,
    day INTEGER NOT NULL,
    yhat REAL NOT NULL,
    method TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_forecasts_run ON forecasts(run_id);
"""


class SQLiteRepository(Repository):
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL") if self.path != ":memory:" else None
        self._conn.executescript(SCHEMA)
        # A run that was running when the server stopped will never finish.
        self._conn.execute("UPDATE runs SET status='failed', error='The server restarted during this run.' "
                           "WHERE status IN ('queued', 'running')")
        self._conn.commit()

    # -- datasets -------------------------------------------------------
    def create_dataset(self, name, clean, report, options, is_synthetic):
        dataset_id = uuid.uuid4().hex[:12]
        rows = clean.assign(
            dataset_id=dataset_id,
            date=clean["date"].dt.strftime("%Y-%m-%d"),
            product_id=clean["product_id"].astype(object).where(clean["product_id"].notna(), None),
            price=clean["price"].astype(object).where(clean["price"].notna(), None),
        )[["dataset_id", "date", "product_key", "product_name", "product_id", "quantity", "price", "row"]]
        with self._lock:
            self._conn.execute(
                "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?)",
                (dataset_id, name, _now(), int(is_synthetic), len(clean), int(clean["product_key"].nunique()),
                 _json(report), _json(options)),
            )
            self._conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?,?,?)", rows.itertuples(index=False, name=None))
            self._conn.commit()
        return dataset_id

    def _dataset_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "name": row["name"], "created_at": row["created_at"],
            "is_synthetic": bool(row["is_synthetic"]), "rows": row["rows"], "products": row["products"],
            "report": json.loads(row["report_json"]), "options": json.loads(row["options_json"]),
        }

    def get_dataset(self, dataset_id):
        row = self._conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            raise NotFound(dataset_id)
        return self._dataset_row(row)

    def list_datasets(self):
        rows = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        out = []
        for row in rows:
            item = self._dataset_row(row)
            item["report"] = {k: item["report"].get(k) for k in ("rows_in", "rows_kept", "date_range", "products")}
            out.append(item)
        return out

    def load_clean(self, dataset_id):
        self.get_dataset(dataset_id)
        frame = pd.read_sql_query(
            "SELECT date, product_key, product_name, product_id, quantity, price, source_row AS row "
            "FROM sales WHERE dataset_id=?", self._conn, params=(dataset_id,)
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        return frame

    def delete_dataset(self, dataset_id):
        with self._lock:
            self._conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
            self._conn.commit()

    # -- runs -----------------------------------------------------------
    def create_run(self, dataset_id, config):
        self.get_dataset(dataset_id)
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, dataset_id, status, progress, message, created_at, config_json) VALUES (?,?,?,?,?,?,?)",
                (run_id, dataset_id, "queued", 0.0, "Waiting to start", _now(), _json(config)),
            )
            self._conn.commit()
        return run_id

    def update_run(self, run_id, **fields):
        allowed = {"status", "progress", "message", "error", "finished_at"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), run_id)
            )
            self._conn.commit()

    def save_run_result(self, run_id, result, forecasts):
        rows = forecasts.assign(run_id=run_id, date=forecasts["date"].dt.strftime("%Y-%m-%d"))[
            ["run_id", "product_key", "product_name", "date", "day", "yhat", "method"]
        ]
        with self._lock:
            self._conn.execute("DELETE FROM forecasts WHERE run_id=?", (run_id,))
            self._conn.executemany("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)", rows.itertuples(index=False, name=None))
            self._conn.execute(
                "UPDATE runs SET result_json=?, status='completed', progress=1, message='Done', finished_at=? WHERE id=?",
                (_json(result), _now(), run_id),
            )
            self._conn.commit()

    def _run_row(self, row: sqlite3.Row, with_result: bool = True) -> dict:
        item = {
            "id": row["id"], "dataset_id": row["dataset_id"], "status": row["status"], "progress": row["progress"],
            "message": row["message"], "error": row["error"], "created_at": row["created_at"],
            "finished_at": row["finished_at"], "config": json.loads(row["config_json"]),
        }
        if with_result:
            item["result"] = json.loads(row["result_json"]) if row["result_json"] else None
        return item

    def get_run(self, run_id):
        row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise NotFound(run_id)
        return self._run_row(row)

    def list_runs(self, dataset_id=None):
        sql = "SELECT * FROM runs" + (" WHERE dataset_id=?" if dataset_id else "") + " ORDER BY created_at DESC"
        rows = self._conn.execute(sql, (dataset_id,) if dataset_id else ()).fetchall()
        return [self._run_row(r, with_result=False) for r in rows]

    def load_forecasts(self, run_id):
        self.get_run(run_id)
        frame = pd.read_sql_query("SELECT product_key, product_name, date, day, yhat, method FROM forecasts WHERE run_id=? "
                                  "ORDER BY product_name, date", self._conn, params=(run_id,))
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
