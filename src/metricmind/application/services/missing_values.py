"""MissingValueHandler: explicit, causal repair of missing samples."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissingValueReport:
    missing_before: int
    filled: int
    rows_dropped: int
    missing_after: int


class MissingValueHandler:
    """Forward-fill up to ``max_ffill_steps`` consecutive samples, then drop rows still incomplete.

    Why forward-fill (and not interpolation)? It only uses PAST values ("last known reading"), so
    it cannot leak future information into earlier rows. Gaps longer than the limit are NOT
    invented - those rows are dropped and reported.
    """

    def __init__(self, max_ffill_steps: int = 4) -> None:
        self._limit = max_ffill_steps

    def handle(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, MissingValueReport]:
        before = int(frame.isna().sum().sum())
        filled_frame = frame.ffill(limit=self._limit) if self._limit > 0 else frame.copy()
        filled = before - int(filled_frame.isna().sum().sum())
        cleaned = filled_frame.dropna(how="any")
        dropped = len(filled_frame) - len(cleaned)
        report = MissingValueReport(before, filled, dropped, int(cleaned.isna().sum().sum()))
        logger.info("Missing values: %s", report)
        return cleaned, report
