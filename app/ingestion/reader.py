"""Read an uploaded CSV or Excel file into a DataFrame of raw text values."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


class DatasetError(ValueError):
    """A problem with the uploaded file that the user needs to fix."""


EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
CSV_EXT = {".csv", ".txt", ".tsv"}


def read_table(content: bytes, filename: str, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if not content:
        raise DatasetError("The file is empty.")
    if suffix in EXCEL_EXT:
        try:
            book = pd.ExcelFile(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001 - openpyxl raises many types
            raise DatasetError(f"Could not open the Excel file: {exc}") from exc
        if sheet is None:
            # Pick the sheet with the most rows; pharmacy exports often add notes sheets.
            sizes = {name: len(book.parse(name, nrows=None, dtype=object)) for name in book.sheet_names}
            sheet = max(sizes, key=sizes.get)
        frame = book.parse(sheet, dtype=object)
    elif suffix in CSV_EXT:
        frame = None
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                frame = pd.read_csv(io.BytesIO(content), dtype=object, sep=None, engine="python", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:  # noqa: BLE001
                raise DatasetError(f"Could not read the CSV file: {exc}") from exc
        if frame is None:
            raise DatasetError("Could not read the CSV file. Save it as UTF-8 and try again.")
    else:
        raise DatasetError(f"Unsupported file type '{suffix or 'none'}'. Upload a CSV or Excel file.")

    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise DatasetError("The file has no data rows.")
    return frame.reset_index(drop=True)
