"""MetricMerger: aligns separate metric series on timestamps into ONE multivariate frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from metricmind.domain.entities.metric import MetricSeries
from metricmind.domain.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    frame: pd.DataFrame  # DatetimeIndex 'timestamp'; one float column per metric
    n_timestamps: int
    n_inserted_gap_rows: int  # timestamps absent from every file, inserted as explicit NaN rows
    n_duplicates_resolved: int


class MetricMerger:
    """Strategy (documented, deterministic):

    1. Resolve duplicate timestamps within each series per ``duplicate_policy``.
    2. Sort chronologically and OUTER-join all series on timestamp (nothing is dropped).
    3. Build the regular sampling grid (first..last timestamp at ``interval_seconds``) and insert
       grid timestamps that no series contains as explicit NaN rows, so gaps are visible rather than
       silently skipped. Off-grid timestamps are kept, never discarded.
    Missing values are NOT filled here; that is a separate, explicit step.
    """

    def __init__(self, interval_seconds: int, duplicate_policy: str = "error") -> None:
        self._interval = pd.Timedelta(seconds=interval_seconds)
        self._policy = duplicate_policy

    def merge(self, series_list: list[MetricSeries]) -> MergeResult:
        if not series_list:
            raise DataValidationError("No metric series to merge")
        names = [s.metric.name for s in series_list]
        if len(set(names)) != len(names):
            raise DataValidationError(f"Duplicate metric names: {names}")

        n_dups = 0
        columns = []
        for s in series_list:
            col, dropped = self._to_series(s)
            n_dups += dropped
            columns.append(col)

        frame = pd.concat(columns, axis=1, join="outer").sort_index()
        frame.index.name = "timestamp"

        grid = pd.date_range(frame.index.min(), frame.index.max(), freq=self._interval)
        full_index = frame.index.union(grid)
        inserted = len(full_index) - len(frame.index)
        frame = frame.reindex(full_index)
        frame.index.name = "timestamp"

        logger.info(
            "Merged %d metrics -> %d timestamps (%d gap rows inserted, %d duplicates resolved)",
            len(columns),
            len(frame),
            inserted,
            n_dups,
        )
        return MergeResult(frame, len(frame), inserted, n_dups)

    def _to_series(self, s: MetricSeries) -> tuple[pd.Series, int]:
        col = pd.Series(s.values, index=pd.DatetimeIndex(s.timestamps), name=s.metric.name)
        col = col.sort_index(kind="stable")
        n_dup = int(col.index.duplicated().sum())
        if n_dup:
            if self._policy == "error":
                raise DataValidationError(f"{s.metric.name}: {n_dup} duplicate timestamps")
            if self._policy == "mean":
                col = col.groupby(level=0).mean()
            else:
                col = col[~col.index.duplicated(keep=self._policy)]  # 'first' | 'last'
        return col, n_dup
