"""FastAPI application: upload, validate, train, forecast, export, and the dashboard."""

from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import BASE_DIR, settings
from app.evaluation.splits import InsufficientDataError
from app.forecasting.pipeline import RunConfig, default_models, run_forecasting
from app.ingestion.pipeline import IngestOptions, ingest
from app.ingestion.reader import DatasetError, read_table
from app.reports.export import daily_forecast_csv, html_report, summary_csv
from app.storage.repository import NotFound, Repository, SQLiteRepository
from app.synthetic.generator import SYNTHETIC_FILENAME, generate

log = logging.getLogger("pharmapredict")


class RunRequest(BaseModel):
    models: list[str] | None = Field(default=None, description="Model names to compare. Leave empty for all.")


class DemoRequest(BaseModel):
    days: int = Field(default=730, ge=150, le=1500)
    products: int | None = Field(default=None, ge=3, le=24)
    seed: int = 7


def create_app(repo: Repository | None = None, run_inline: bool = False) -> FastAPI:
    """``run_inline=True`` runs training in the request (used by tests)."""
    repo = repo or SQLiteRepository(settings.db_path)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trainer")
    app = FastAPI(title="PharmaPredict AI", version="0.1.0",
                  description="Pharmacy demand forecasting. Upload sales history, compare models, get 7/14/30-day forecasts.")
    app.state.repo = repo

    def dataset_or_404(dataset_id: str) -> dict:
        try:
            return repo.get_dataset(dataset_id)
        except NotFound:
            raise HTTPException(404, "Dataset not found.") from None

    def run_or_404(run_id: str) -> dict:
        try:
            return repo.get_run(run_id)
        except NotFound:
            raise HTTPException(404, "Run not found.") from None

    def completed_run(run_id: str) -> dict:
        run = run_or_404(run_id)
        if run["status"] != "completed":
            raise HTTPException(409, f"This run is {run['status']}. Results are not ready.")
        return run

    def execute(run_id: str, dataset_id: str, config: RunConfig) -> None:
        def progress(message: str, fraction: float) -> None:
            repo.update_run(run_id, status="running", message=message, progress=round(fraction, 3))
        try:
            clean = repo.load_clean(dataset_id)
            result = run_forecasting(clean, config, progress)
            payload = {k: v for k, v in asdict(result).items() if k != "forecasts"}
            repo.save_run_result(run_id, payload, result.forecasts)
        except (InsufficientDataError, ValueError) as exc:
            repo.update_run(run_id, status="failed", error=str(exc), message="Stopped")
        except Exception as exc:  # noqa: BLE001
            log.error("Run %s failed\n%s", run_id, traceback.format_exc())
            repo.update_run(run_id, status="failed", error=f"Unexpected error: {exc}", message="Stopped")

    # ---- health and metadata ------------------------------------------
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/models")
    def models():
        return [{"name": m.name, "family": m.family, "description": m.description} for m in default_models()]

    # ---- datasets -----------------------------------------------------
    @app.post("/api/datasets", status_code=201)
    async def upload(
        file: UploadFile = File(...),
        absent_days_are_zero: bool = Form(False),
        drop_exact_duplicates: bool = Form(False),
        date_format: str = Form("auto"),
        column_mapping: str | None = Form(None, description='JSON, e.g. {"quantity": "Qty Sold"}'),
        name_aliases: str | None = Form(None, description='JSON, e.g. {"Panadol 500mg": "Paracetamol 500mg"}'),
    ):
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"The file is larger than {settings.max_upload_mb} MB.")
        if date_format not in {"auto", "dayfirst", "monthfirst"}:
            raise HTTPException(422, "date_format must be auto, dayfirst or monthfirst.")
        try:
            mapping = json.loads(column_mapping) if column_mapping else None
            aliases = json.loads(name_aliases) if name_aliases else None
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"Could not read the JSON settings: {exc}") from None
        options = IngestOptions(date_format=date_format, drop_exact_duplicates=drop_exact_duplicates,
                                absent_days_are_zero=absent_days_are_zero, column_mapping=mapping, name_aliases=aliases)
        is_synthetic = (file.filename or "").upper().startswith("SYNTHETIC")
        return _ingest_and_store(content, file.filename or "upload.csv", options, is_synthetic)

    def _ingest_and_store(content: bytes, filename: str, options: IngestOptions, is_synthetic: bool) -> dict:
        try:
            raw = read_table(content, filename)
            result = ingest(raw, options)
        except DatasetError as exc:
            raise HTTPException(422, str(exc)) from None
        dataset_id = repo.create_dataset(filename, result.clean, result.report.as_dict(), asdict(options), is_synthetic)
        return repo.get_dataset(dataset_id)

    @app.post("/api/datasets/demo", status_code=201)
    def demo(request: DemoRequest | None = None):
        request = request or DemoRequest()
        raw = generate(days=request.days, seed=request.seed, products=request.products)
        content = raw.to_csv(index=False).encode()
        # The generator writes every sale, so absent days really are zero sales.
        options = IngestOptions(absent_days_are_zero=True)
        return _ingest_and_store(content, SYNTHETIC_FILENAME, options, is_synthetic=True)

    @app.get("/api/datasets/demo/download")
    def demo_download(days: int = Query(730, ge=150, le=1500), seed: int = 7):
        csv = generate(days=days, seed=seed).to_csv(index=False)
        return PlainTextResponse(csv, media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{SYNTHETIC_FILENAME}"'})

    @app.get("/api/datasets")
    def list_datasets():
        return repo.list_datasets()

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return dataset_or_404(dataset_id)

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str):
        dataset_or_404(dataset_id)
        repo.delete_dataset(dataset_id)

    # ---- runs ---------------------------------------------------------
    @app.post("/api/datasets/{dataset_id}/runs", status_code=202)
    def start_run(dataset_id: str, request: RunRequest | None = None):
        dataset = dataset_or_404(dataset_id)
        config = RunConfig(absent_days_are_zero=bool(dataset["options"].get("absent_days_are_zero")),
                           models=(request.models if request else None))
        run_id = repo.create_run(dataset_id, asdict(config))
        if run_inline:
            execute(run_id, dataset_id, config)
        else:
            executor.submit(execute, run_id, dataset_id, config)
        return run_or_404(run_id) | {"result": None}

    @app.get("/api/runs")
    def list_runs(dataset_id: str | None = None):
        return repo.list_runs(dataset_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, include_series: bool = False):
        run = run_or_404(run_id)
        if run.get("result") and not include_series:
            run["result"] = {k: v for k, v in run["result"].items() if k != "series"}
        return run

    @app.get("/api/runs/{run_id}/products/{product_key:path}")
    def product_detail(run_id: str, product_key: str):
        run = completed_run(run_id)
        result = run["result"]
        product = next((p for p in result["products"] if p["product_key"] == product_key), None)
        if product is None:
            raise HTTPException(404, "Medicine not found in this run.")
        return {"product": product, "series": result["series"].get(product_key), "selected_model": result["selected_model"]}

    # ---- exports ------------------------------------------------------
    @app.get("/api/runs/{run_id}/export/forecasts.csv")
    def export_daily(run_id: str):
        run = completed_run(run_id)
        dataset = dataset_or_404(run["dataset_id"])
        csv = daily_forecast_csv(repo.load_forecasts(run_id), dataset["is_synthetic"])
        return PlainTextResponse(csv, media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="forecast_daily_{run_id}.csv"'})

    @app.get("/api/runs/{run_id}/export/summary.csv")
    def export_summary(run_id: str):
        run = completed_run(run_id)
        dataset = dataset_or_404(run["dataset_id"])
        horizons = tuple(run["config"].get("horizons", settings.horizons))
        csv = summary_csv(run["result"]["products"], horizons, dataset["is_synthetic"])
        return PlainTextResponse(csv, media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="forecast_summary_{run_id}.csv"'})

    @app.get("/api/runs/{run_id}/export/report.html", response_class=HTMLResponse)
    def export_report(run_id: str, download: bool = False):
        run = completed_run(run_id)
        dataset = dataset_or_404(run["dataset_id"])
        headers = {"Content-Disposition": f'attachment; filename="pharmapredict_report_{run_id}.html"'} if download else {}
        return HTMLResponse(html_report(dataset, run), headers=headers)

    # ---- dashboard ----------------------------------------------------
    web_dir = Path(BASE_DIR) / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
