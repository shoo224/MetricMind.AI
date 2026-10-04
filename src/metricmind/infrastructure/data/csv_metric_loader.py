"""CSVMetricDataLoader: reads one metric CSV and returns a validated domain MetricSeries."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from metricmind.domain.entities.metric import Metric, MetricSeries
from metricmind.domain.exceptions import DataValidationError
from metricmind.domain.ports.data_loader import MetricDataLoader

logger = logging.getLogger(__name__)


class CSVMetricDataLoader(MetricDataLoader):
    """All pandas / CSV knowledge for metrics lives here (infrastructure layer).

    Responsibilities: locate file, read, check columns, parse timestamps strictly, coerce values,
    and turn the source's "probe failed" sentinel (e.g. -99999) into NaN so it is never mistaken
    for a real measurement. Duplicates / ordering / gaps are deliberately NOT fixed here: they
    are reported by the validation service.
    """

    def __init__(
        self,
        raw_dir: Path,
        timestamp_column: str = "Time",
        value_column: str = "Value",
        timestamp_format: str | None = "%Y-%m-%d %H:%M:%S",
        missing_sentinels: tuple[float, ...] = (),
    ) -> None:
        self._raw_dir = raw_dir
        self._ts_col = timestamp_column
        self._val_col = value_column
        self._ts_format = timestamp_format
        self._sentinels = missing_sentinels

    def load(self, metric: Metric) -> MetricSeries:
        path = self._raw_dir / metric.source
        if not path.is_file():
            raise DataValidationError(f"{metric.name}: file not found: {path}")

        frame = pd.read_csv(path, dtype=str, skipinitialspace=True)
        if frame.empty:
            raise DataValidationError(f"{metric.name}: file is empty: {path.name}")
        missing = {self._ts_col, self._val_col} - set(frame.columns)
        if missing:
            raise DataValidationError(
                f"{metric.name}: missing column(s) {sorted(missing)}; found {list(frame.columns)}"
            )

        timestamps = self._parse_timestamps(frame[self._ts_col], metric.name)
        values = self._parse_values(frame[self._val_col], metric.name)

        logger.info("Loaded %s: %d rows from %s", metric.name, len(frame), path.name)
        return MetricSeries(
            metric=metric,
            timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
            values=values.to_numpy(dtype=np.float64),
        )

    def _parse_timestamps(self, column: pd.Series, name: str) -> pd.Series:
        parsed = pd.to_datetime(column, format=self._ts_format, errors="coerce")
        bad = parsed.isna()
        if bad.any():
            examples = column[bad].head(3).tolist()
            raise DataValidationError(
                f"{name}: {int(bad.sum())} missing/invalid timestamp(s), e.g. {examples}"
            )
        return parsed

    def _parse_values(self, column: pd.Series, name: str) -> pd.Series:
        numeric = pd.to_numeric(column, errors="coerce")
        was_blank = column.isna() | (column.str.strip() == "")
        non_numeric = numeric.isna() & ~was_blank
        if non_numeric.any():
            examples = column[non_numeric].head(3).tolist()
            raise DataValidationError(
                f"{name}: {int(non_numeric.sum())} non-numeric value(s), e.g. {examples}"
            )
        for sentinel in self._sentinels:
            hits = numeric == sentinel
            if hits.any():
                logger.info(
                    "%s: %d sentinel value(s) %s treated as missing",
                    name,
                    int(hits.sum()),
                    sentinel,
                )
                numeric = numeric.mask(hits)
        return numeric
