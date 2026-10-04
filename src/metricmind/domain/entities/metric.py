"""Metric and MetricSeries: what is being measured and its observations over time."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from metricmind.domain.exceptions import DataValidationError


@dataclass(frozen=True)
class Metric:
    """Definition of one monitored metric (identity + meaning, not data)."""

    name: str  # canonical column name, e.g. "cpu_usage"
    source: str  # where its data lives (a file name for CSV; could be a topic/table later)
    category: str  # "cpu" | "memory" | "network" - links a metric to its fault-event category
    unit: str = "unspecified"
    description: str = ""


@dataclass(frozen=True)
class MetricSeries:
    """Observations of a single metric: parallel arrays of timestamps and values.

    Values may contain NaN (= missing). Timestamps are NOT guaranteed unique or sorted here;
    checking that is the job of the validation service, so problems are reported, not hidden.
    """

    metric: Metric
    timestamps: NDArray[np.datetime64] = field(repr=False)
    values: NDArray[np.float64] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.values):
            raise DataValidationError(
                f"{self.metric.name}: {len(self.timestamps)} timestamps vs "
                f"{len(self.values)} values"
            )

    def __len__(self) -> int:
        return len(self.values)
