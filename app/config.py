"""Central settings. Every value can be overridden with an environment variable
prefixed PHARMAPREDICT_, for example PHARMAPREDICT_DB_PATH=/tmp/pp.db."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PHARMAPREDICT_{name}", default)


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(BASE_DIR / "data" / "pharmapredict.db"))))
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "50"))
    # Forecast horizons the product promises (days).
    horizons: tuple[int, ...] = (7, 14, 30)
    # Length of the validation and test windows (days). Shrinks automatically
    # for short datasets, down to min_eval_days.
    eval_days: int = int(_env("EVAL_DAYS", "30"))
    min_eval_days: int = int(_env("MIN_EVAL_DAYS", "14"))
    # A medicine needs at least this many observed days before the validation
    # window to be modelled. Below that it is labelled "insufficient data".
    min_train_days: int = int(_env("MIN_TRAIN_DAYS", "60"))
    random_state: int = 42


settings = Settings()
