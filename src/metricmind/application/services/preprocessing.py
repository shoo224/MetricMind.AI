"""FeaturePreprocessor: select model features and scale them, without data leakage."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from metricmind.domain.exceptions import DataValidationError
from metricmind.domain.ports.feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """``fit`` learns scaling parameters from TRAIN rows only; ``transform`` re-applies them to any
    later data (test set, or live data at inference). Never call ``fit`` on test data.

    No winsorising/clipping on purpose: extreme values ARE the anomalies we want to detect, so
    clipping them would hide the signal.
    """

    def __init__(self, scaler: FeatureScaler, feature_columns: Sequence[str]) -> None:
        self._scaler = scaler
        self.feature_columns = list(feature_columns)

    @property
    def scaler(self) -> FeatureScaler:
        """The (fitted) scaler, so it can be persisted and reloaded for inference."""
        return self._scaler

    def fit(self, train_frame: pd.DataFrame) -> FeaturePreprocessor:
        self._scaler.fit(self._select(train_frame))
        logger.info(
            "Scaler fitted on %d train rows, features=%s", len(train_frame), self.feature_columns
        )
        return self

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        return self._scaler.transform(self._select(frame))

    def _select(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise DataValidationError(f"Missing feature column(s): {missing}")
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataValidationError("Features contain NaN/inf; run missing-value handling first")
        return values
