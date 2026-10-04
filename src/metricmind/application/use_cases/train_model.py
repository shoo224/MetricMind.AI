"""TrainModelUseCase: fit scaler + anomaly detector on TRAIN data only and persist them."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from metricmind.application.ports import NullReporter, ProgressReporter, TabularStore
from metricmind.application.services.preprocessing import FeaturePreprocessor
from metricmind.application.use_cases.prepare_data import TRAIN
from metricmind.domain.ports.anomaly_detector import AnomalyDetector
from metricmind.domain.ports.model_repository import ModelRepository

logger = logging.getLogger(__name__)

SCALER_ARTEFACT = "scaler"
DETECTOR_ARTEFACT = "isolation_forest"
METADATA_ARTEFACT = "model_metadata"


class TrainModelUseCase:
    def __init__(
        self,
        processed_store: TabularStore,
        preprocessor: FeaturePreprocessor,
        detector: AnomalyDetector,
        models: ModelRepository,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._store = processed_store
        self._pre = preprocessor
        self._detector = detector
        self._models = models
        self._out = reporter or NullReporter()

    def execute(self) -> dict[str, Any]:
        out = self._out
        train = self._store.load_frame(TRAIN)
        out.section("Training data")
        out.step(f"{len(train):,} rows, features: {', '.join(self._pre.feature_columns)}")
        out.step("labels are NOT used for training (unsupervised)")

        out.section("Fitting scaler (train rows only) and detector")
        features = self._pre.fit(train).transform(train)
        self._detector.fit(features)
        flagged = float(self._detector.predict(features).mean())
        out.step(f"{self._detector.name} fitted; score threshold = {self._detector.threshold:.4f}")
        out.step(
            f"{100 * flagged:.2f}% of training rows flagged "
            f"(contamination = {self._detector.get_params()['contamination']})"
        )

        out.section("Saving artefacts")
        paths = [
            self._models.save(SCALER_ARTEFACT, self._pre.scaler),
            self._models.save(DETECTOR_ARTEFACT, self._detector),
        ]
        metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "detector": self._detector.name,
            "hyperparameters": self._detector.get_params(),
            "threshold": self._detector.threshold,
            "feature_columns": self._pre.feature_columns,
            "train_rows": len(train),
            "train_start": train.index.min(),
            "train_end": train.index.max(),
            "train_flagged_fraction": flagged,
        }
        paths.append(self._models.save_metadata(METADATA_ARTEFACT, metadata))
        for p in paths:
            out.step(str(p))
        return metadata
