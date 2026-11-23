# PharmaPredict AI

Daily demand forecasting for a pharmacy. Upload a sales export (CSV or Excel), check the data quality notes, then train and compare models and get forecasts for the next 7, 14 and 30 days for every medicine.

This is a working prototype. It has only been tested on **synthetic** data. Do not use its numbers to place real orders until it has been checked on your own sales history (see "Limits" below).

## What it does

1. **Reads messy files.** Finds the date, medicine, quantity, price and invoice columns even when they are named differently (`Qty Sold`, `Unit Price (NGN)`, `Txn Date`...). Handles day-first dates, Excel dates, currency symbols and commas.
2. **Cleans and reports.** Groups spellings of the same medicine (`Paracetamol 500mg Tabs` and `PARACETAMOL 500 MG TABLETS`), but never merges different strengths or brands on its own. Removes double-exported invoice lines, leaves out returns, and lists every problem it found.
3. **Treats missing days honestly.** A day with no sales at all is "unknown" (closed, holiday or missing data), never zero. A day where other medicines sold but this one did not counts as zero only if you tick "my file lists every sale".
4. **Trains and compares 7 models:** seasonal naive, 28-day moving average, Holt-Winters, Croston (for slow sellers), Linear Regression, Random Forest and XGBoost.
5. **Scores them fairly.** Data is split by time: train, then validation, then test. The model is chosen on validation only. The test score is reported as the honest estimate.
6. **Forecasts** 7, 14 and 30 days ahead, with revenue at the latest price.
7. **Dashboard and exports:** per-medicine charts, model comparison, data quality, CSV exports and a full HTML report.

## Setup

Needs Python 3.10 or newer.

```bash
cd pharmapredict
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.api.main:app --reload
```

Open http://127.0.0.1:8000 and click **Try synthetic demo**. Training on 24 medicines over two years takes about 30 seconds.

Optional settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `PHARMAPREDICT_DB_PATH` | `data/pharmapredict.db` | Where the SQLite database lives |

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

16 tests cover column mapping, date and number parsing, name grouping, the missing-day rule, leak-free features, the time-based split, metrics and the full upload to export flow through the API.

## Your file

At minimum: a **date**, a **medicine name** and a **quantity** column. Price and invoice number are optional but help (revenue, and removal of double exports). One row per sale line is ideal. Daily totals per medicine also work.

If a column is not found, the upload screen accepts a mapping such as `{"date": "Day of Sale"}`. If two names are the same product, add an alias such as `{"Emzor Paracetamol": "Paracetamol 500mg Tablets"}`.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/models` | List of models |
| POST | `/api/datasets` | Upload a file (multipart) |
| POST | `/api/datasets/demo` | Create synthetic demo data |
| GET | `/api/datasets/demo/download` | Download the synthetic CSV |
| GET/DELETE | `/api/datasets/{id}` | Dataset with its quality report |
| POST | `/api/datasets/{id}/runs` | Start training (runs in the background) |
| GET | `/api/runs/{id}` | Status, progress and results |
| GET | `/api/runs/{id}/products/{key}` | One medicine's history, test forecast and forecast |
| GET | `/api/runs/{id}/export/forecasts.csv` | Daily forecasts |
| GET | `/api/runs/{id}/export/summary.csv` | 7/14/30-day totals per medicine |
| GET | `/api/runs/{id}/export/report.html` | Full report |

Interactive docs at `/docs`.

## Project layout

```
app/
  ingestion/    read file, map columns, parse, clean names, validation report
  features/     daily series per medicine, past-only features
  models/       baselines, statistical and machine learning forecasters
  evaluation/   metrics and time-based split
  forecasting/  the training, selection and forecasting pipeline
  synthetic/    demo data generator (clearly labelled)
  storage/      Repository interface + SQLite version
  reports/      CSV and HTML exports
  api/          FastAPI app
web/            dashboard (plain HTML, CSS and JS)
tests/
docs/MODELLING.md   why the models and rules are what they are
```

## Moving to Supabase later

All storage goes through the `Repository` class in `app/storage/repository.py`. To use Supabase (Postgres), write a `SupabaseRepository` with the same methods and pass it to `create_app(repo=...)`. Nothing else needs to change. Keep the service role key on the server only.

## Limits

* Tested only on synthetic data. The synthetic data was written to look like a Nigerian pharmacy (rainy season malaria, harmattan coughs, Sunday closing, holidays, stock-outs), but it is still made up. Good scores on it say nothing about real accuracy.
* Stock-outs look like low demand. If a medicine was out of stock, the model learns that fewer were "wanted". Adding a stock column would fix this.
* No promotions, prices of competitors or supplier delays are used.
* Medicines with under 60 days of history get a rough average, not a model forecast.
* Forecasts are single numbers, not ranges.
