"""Ports for reading raw data. Infrastructure implements them (e.g. from CSV files)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from metricmind.domain.entities.event import LabeledEvent
from metricmind.domain.entities.metric import Metric, MetricSeries


class MetricDataLoader(ABC):
    """Loads the observations of one metric from some source."""

    @abstractmethod
    def load(self, metric: Metric) -> MetricSeries:
        """Return a clean MetricSeries or raise DataValidationError."""


class EventLabelLoader(ABC):
    """Loads ground-truth fault events (may legitimately return an empty list)."""

    @abstractmethod
    def load_events(self) -> list[LabeledEvent]: ...
