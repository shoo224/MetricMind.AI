"""AnomalyDetector port. Isolation Forest today; an LSTM autoencoder later - same interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


class AnomalyDetector(ABC):
    """Unsupervised anomaly detector over a 2-D feature matrix (rows = time steps).

    Convention used project-wide: a HIGHER score means MORE anomalous, and prediction 1 = anomaly.
    (scikit-learn uses the opposite sign; the adapter converts so callers never have to care.)
    """

    name: str

    @abstractmethod
    def fit(self, features: NDArray[np.float64]) -> Self:
        """Learn what 'normal' looks like from ``features``. Labels are never used."""

    @abstractmethod
    def score_samples(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continuous anomaly score per row (higher = more anomalous)."""

    @abstractmethod
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        """1 = anomaly, 0 = normal, using the threshold learned during ``fit``."""

    @property
    @abstractmethod
    def threshold(self) -> float:
        """Score above which a row is classified as an anomaly."""

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """Hyper-parameters, for the saved model metadata."""
