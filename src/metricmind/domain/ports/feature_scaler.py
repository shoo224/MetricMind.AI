"""FeatureScaler port: fit on training data only, then reuse unchanged at inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


class FeatureScaler(ABC):
    @abstractmethod
    def fit(self, features: NDArray[np.float64]) -> Self: ...

    @abstractmethod
    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...
