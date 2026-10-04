"""Anomaly and AnomalyResult: the output of a detector, expressed without pandas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Anomaly:
    """One observation flagged as anomalous."""

    timestamp: np.datetime64
    score: float  # higher = more anomalous


@dataclass(frozen=True)
class AnomalyResult:
    """Detector output for a batch of observations.

    ``scores``: continuous anomaly score (higher = more anomalous).
    ``predictions``: 1 = anomaly, 0 = normal, obtained by comparing ``scores`` to ``threshold``.
    """

    detector_name: str
    threshold: float
    timestamps: NDArray[np.datetime64] = field(repr=False)
    scores: NDArray[np.float64] = field(repr=False)
    predictions: NDArray[np.int_] = field(repr=False)

    @property
    def anomaly_count(self) -> int:
        return int(self.predictions.sum())

    @property
    def anomaly_rate(self) -> float:
        return self.anomaly_count / len(self.predictions) if len(self.predictions) else 0.0

    @property
    def anomalies(self) -> list[Anomaly]:
        idx = np.flatnonzero(self.predictions == 1)
        return [Anomaly(self.timestamps[i], float(self.scores[i])) for i in idx]
