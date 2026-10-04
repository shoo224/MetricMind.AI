"""StandardScaler adapter implementing the FeatureScaler port."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from metricmind.domain.exceptions import NotFittedError
from metricmind.domain.ports.feature_scaler import FeatureScaler


class SklearnStandardScaler(FeatureScaler):
    """z = (x - mean) / std, with mean/std learned from the data given to ``fit`` only."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, features: NDArray[np.float64]) -> Self:
        self._scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted:
            raise NotFittedError("Scaler used before fit()")
        return self._scaler.transform(features)
