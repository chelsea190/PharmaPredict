"""Chronological train / validation / test windows. Never random."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class InsufficientDataError(ValueError):
    """The dataset is too short to evaluate forecasts honestly."""


@dataclass(frozen=True)
class Split:
    train_end: pd.Timestamp     # last training day
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window: int                 # days in validation and in test

    def as_dict(self) -> dict:
        return {k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in self.__dict__.items()}


def make_split(first: pd.Timestamp, last: pd.Timestamp, eval_days: int, min_eval_days: int, min_train_days: int) -> Split:
    """Last ``window`` days = test, the ``window`` days before = validation, the rest = training.

    The window shrinks for short datasets but never below ``min_eval_days``.
    """
    span = (last - first).days + 1
    window = eval_days
    while window > min_eval_days and span - 2 * window < min_train_days:
        window -= 1
    if span - 2 * window < min_train_days:
        need = min_train_days + 2 * min_eval_days
        raise InsufficientDataError(
            f"The data covers {span} days. At least {need} days are needed to train, validate and test "
            f"({min_train_days} training days plus two {min_eval_days}-day evaluation windows)."
        )
    test_start = last - pd.Timedelta(days=window - 1)
    val_start = test_start - pd.Timedelta(days=window)
    return Split(
        train_end=val_start - pd.Timedelta(days=1),
        val_start=val_start,
        val_end=test_start - pd.Timedelta(days=1),
        test_start=test_start,
        test_end=last,
        window=window,
    )
