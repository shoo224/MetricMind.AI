"""ChronologicalSplit: the rule for dividing a time series into train/test without leakage."""

from __future__ import annotations

from dataclasses import dataclass

from metricmind.domain.exceptions import DataValidationError


@dataclass(frozen=True)
class ChronologicalSplit:
    """Earliest ``train_fraction`` of rows -> train; the remainder (later in time) -> test.

    Deliberately NOT random: shuffling a time series would let the model see the "future".
    """

    train_fraction: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise DataValidationError("train_fraction must be strictly between 0 and 1")

    def boundary(self, n_rows: int) -> int:
        """Index of the first test row. Rows ``[0, boundary)`` are train."""
        boundary = int(n_rows * self.train_fraction)
        if boundary < 1 or boundary >= n_rows:
            raise DataValidationError(f"Cannot split {n_rows} rows with {self.train_fraction}")
        return boundary
