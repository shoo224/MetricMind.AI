"""SeriesValidator: detects data-quality problems in a MetricSeries and reports them clearly."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from metricmind.domain.entities.metric import MetricSeries
from metricmind.domain.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesValidationReport:
    metric: str
    n_rows: int
    n_missing_values: int
    n_duplicate_timestamps: int
    is_sorted: bool
    dominant_interval_seconds: float
    n_irregular_intervals: int  # steps that differ from the expected interval


class SeriesValidator:
    """Fatal problems raise ``DataValidationError``; recoverable ones are logged and reported.

    * empty series ................................ fatal
    * duplicate timestamps ........................ fatal when ``duplicate_policy == 'error'``
    * missing values / unsorted / irregular steps .. reported (handled later by explicit steps)
    * dominant sampling interval != expected ....... fatal (the wrong file/frequency was supplied)
    """

    def __init__(self, expected_interval_seconds: int, duplicate_policy: str = "error") -> None:
        self._expected = expected_interval_seconds
        self._duplicate_policy = duplicate_policy

    def validate(self, series: MetricSeries) -> SeriesValidationReport:
        name = series.metric.name
        if len(series) == 0:
            raise DataValidationError(f"{name}: series is empty")

        ts = series.timestamps
        n_dup = int(len(ts) - len(np.unique(ts)))
        is_sorted = bool(np.all(ts[1:] >= ts[:-1]))
        steps = np.diff(np.sort(np.unique(ts))).astype("timedelta64[s]").astype(float)
        dominant = float(np.median(steps)) if len(steps) else float(self._expected)
        irregular = int(np.sum(steps != self._expected))
        n_missing = int(np.isnan(series.values).sum())

        report = SeriesValidationReport(
            name, len(series), n_missing, n_dup, is_sorted, dominant, irregular
        )

        if n_dup and self._duplicate_policy == "error":
            raise DataValidationError(
                f"{name}: {n_dup} duplicate timestamp(s). Set DUPLICATE_POLICY to first/last/mean "
                f"to resolve them explicitly."
            )
        if dominant != self._expected:
            raise DataValidationError(
                f"{name}: sampling interval is {dominant:g}s, expected {self._expected}s"
            )
        if n_dup or not is_sorted or irregular:
            logger.warning("Validation issues: %s", report)  # structural problems
        else:  # clean, or only missing values - which are reported and repaired explicitly later
            logger.info("Validation passed: %s", report)
        return report
