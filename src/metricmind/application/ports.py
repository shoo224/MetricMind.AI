"""Application-level ports for tabular (pandas) I/O and figures.

The application layer uses pandas DataFrames as its tabular type (the domain does not). These
ports keep file formats and plotting libraries OUT of the application layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class TabularStore(ABC):
    """Persist/reload DataFrames and JSON documents under one base location."""

    @abstractmethod
    def save_frame(self, name: str, frame: pd.DataFrame) -> Path: ...

    @abstractmethod
    def load_frame(self, name: str) -> pd.DataFrame: ...

    @abstractmethod
    def save_json(self, name: str, document: dict[str, Any]) -> Path: ...

    @abstractmethod
    def load_json(self, name: str) -> dict[str, Any]: ...


class FigureRenderer(ABC):
    """Draws report figures and returns the saved file path."""

    @abstractmethod
    def metric_timeseries(
        self, frame: pd.DataFrame, columns: Sequence[str], title: str, filename: str
    ) -> Path:
        """Line plot per column over time; labelled fault windows are shaded if available."""

    @abstractmethod
    def correlation_heatmap(
        self, frame: pd.DataFrame, columns: Sequence[str], filename: str
    ) -> Path: ...

    @abstractmethod
    def confusion_matrix(
        self, matrix: Sequence[Sequence[int]], title: str, filename: str
    ) -> Path: ...

    @abstractmethod
    def anomaly_overlay(
        self,
        frame: pd.DataFrame,
        metric: str,
        predictions: NDArray[np.int_],
        title: str,
        filename: str,
    ) -> Path:
        """Metric line + detected anomaly points, with true fault windows shaded."""

    @abstractmethod
    def score_timeline(
        self,
        frame: pd.DataFrame,
        scores: NDArray[np.float64],
        threshold: float,
        title: str,
        filename: str,
    ) -> Path: ...


class ProgressReporter(ABC):
    """Lets a use case announce progress without knowing whether it is a console, GUI or API."""

    @abstractmethod
    def section(self, title: str) -> None: ...

    @abstractmethod
    def step(self, message: str) -> None: ...


class NullReporter(ProgressReporter):
    def section(self, title: str) -> None:  # noqa: D102
        pass

    def step(self, message: str) -> None:  # noqa: D102
        pass
