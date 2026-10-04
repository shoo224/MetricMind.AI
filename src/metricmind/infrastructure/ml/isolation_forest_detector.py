"""IsolationForestDetector: scikit-learn Isolation Forest behind the AnomalyDetector port."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from metricmind.domain.exceptions import NotFittedError
from metricmind.domain.ports.anomaly_detector import AnomalyDetector


def parse_max_samples(text: str | int | float) -> str | int | float:
    """'auto' stays 'auto'; '256' -> 256 (rows); '0.5' -> 0.5 (fraction)."""
    if isinstance(text, (int, float)):
        return text
    text = text.strip().lower()
    if text == "auto":
        return "auto"
    return float(text) if "." in text else int(text)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest isolates points with random axis-parallel splits; anomalies are 'few and
    different' so they are isolated in FEWER splits (shorter average path length => higher score).

    Sign convention: scikit-learn's ``score_samples`` is negative-is-anomalous; we return its
    negation so that, project-wide, HIGHER = MORE ANOMALOUS. ``contamination`` is the expected
    anomaly fraction; at fit time it fixes the score threshold at that quantile of the training
    scores.
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: str | int | float = "auto",
        random_state: int | None = 42,
        n_jobs: int = -1,
    ) -> None:
        self._params: dict[str, object] = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": parse_max_samples(max_samples),
            "random_state": random_state,
        }
        self._model = IsolationForest(**self._params, n_jobs=n_jobs)
        self._fitted = False

    def fit(self, features: NDArray[np.float64]) -> Self:
        self._model.fit(features)
        self._fitted = True
        return self

    def score_samples(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        self._require_fitted()
        return -self._model.score_samples(features)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        # sklearn returns -1 (anomaly) / +1 (normal); map to project convention 1 / 0.
        self._require_fitted()
        return (self._model.predict(features) == -1).astype(int)

    @property
    def threshold(self) -> float:
        self._require_fitted()
        return float(-self._model.offset_)

    def get_params(self) -> dict[str, object]:
        return dict(self._params)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("IsolationForestDetector used before fit()")
