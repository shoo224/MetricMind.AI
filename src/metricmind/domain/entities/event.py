"""LabeledEvent: a known (injected) fault interval - the dataset's ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from metricmind.domain.exceptions import DataValidationError


@dataclass(frozen=True)
class LabeledEvent:
    """A period during which a fault of ``category`` was deliberately active."""

    event_id: str
    category: str  # "cpu" | "memory" | "network"
    start: datetime  # timezone-naive, in the same timezone as the metric timestamps
    end: datetime
    mode: str = ""  # e.g. burst / gradual / stable / fluctuation
    intensity: float = 0.0

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise DataValidationError(f"Event {self.event_id}: end is before start")
