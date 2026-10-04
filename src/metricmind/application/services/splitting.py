"""ChronologicalSplitter: applies a ChronologicalSplit rule to a time-ordered frame."""

from __future__ import annotations

import pandas as pd

from metricmind.domain.exceptions import DataValidationError
from metricmind.domain.value_objects.chronological_split import ChronologicalSplit


class ChronologicalSplitter:
    def split(
        self, frame: pd.DataFrame, rule: ChronologicalSplit
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not frame.index.is_monotonic_increasing:
            raise DataValidationError("Frame must be sorted by time before splitting")
        cut = rule.boundary(len(frame))
        train, test = frame.iloc[:cut], frame.iloc[cut:]
        assert train.index.max() < test.index.min()  # no temporal overlap = no leakage
        return train, test
