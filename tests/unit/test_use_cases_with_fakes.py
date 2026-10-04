"""Use cases tested with in-memory fakes: proves they depend on abstractions, not on sklearn/files,
and guards the key anti-leakage rule: evaluation must NEVER re-fit the scaler."""

from pathlib import Path

import numpy as np
import pandas as pd

from metricmind.application.ports import FigureRenderer, TabularStore
from metricmind.application.services.evaluation import AnomalyEvaluator
from metricmind.application.use_cases.evaluate_model import EvaluateModelUseCase
from metricmind.domain.ports.anomaly_detector import AnomalyDetector
from metricmind.domain.ports.feature_scaler import FeatureScaler
from metricmind.domain.ports.model_repository import ModelRepository


class ExplodingScaler(FeatureScaler):
    def fit(self, features):
        raise AssertionError("evaluation must not fit the scaler (data leakage)")

    def transform(self, features):
        return features


class ThresholdDetector(AnomalyDetector):
    name = "fake"

    def fit(self, features):
        return self

    def score_samples(self, features):
        return features[:, 0]

    def predict(self, features):
        return (features[:, 0] > self.threshold).astype(int)

    @property
    def threshold(self):
        return 5.0

    def get_params(self):
        return {}


class MemoryStore(TabularStore):
    def __init__(self, frames=None):
        self.frames, self.docs = dict(frames or {}), {}

    def save_frame(self, name, frame):
        self.frames[name] = frame
        return Path(name)

    def load_frame(self, name):
        return self.frames[name]

    def save_json(self, name, document):
        self.docs[name] = document
        return Path(name)

    def load_json(self, name):
        return self.docs[name]


class MemoryModels(ModelRepository):
    def __init__(self):
        self.items = {"scaler": ExplodingScaler(), "isolation_forest": ThresholdDetector()}
        self.meta = {"trained_at": "t", "feature_columns": ["x"]}

    def save(self, name, artefact):
        self.items[name] = artefact
        return Path(name)

    def load(self, name):
        return self.items[name]

    def save_metadata(self, name, metadata):
        return Path(name)

    def load_metadata(self, name):
        return self.meta


class NoopRenderer(FigureRenderer):
    def __getattribute__(self, item):
        if item.startswith("_") or item in {"metric_timeseries"}:
            return object.__getattribute__(self, item)
        return lambda *a, **k: Path(item)

    def metric_timeseries(self, *a, **k):
        return Path("m")

    def correlation_heatmap(self, *a, **k): ...
    def confusion_matrix(self, *a, **k): ...
    def anomaly_overlay(self, *a, **k): ...
    def score_timeline(self, *a, **k): ...


def test_evaluate_uses_only_the_saved_scaler_and_reports_correct_counts():
    idx = pd.date_range("2025-01-01", periods=6, freq="15s", name="timestamp")
    test = pd.DataFrame(
        {
            "x": [1.0, 9.0, 9.0, 1.0, 1.0, 1.0],
            "is_anomaly": [0, 1, 1, 0, 1, 0],
            "anomaly_category": ["none", "cpu", "cpu", "none", "cpu", "none"],
            "event_id": ["", "e1", "e1", "", "e2", ""],
        },
        index=idx,
    )
    results = MemoryStore()
    doc = EvaluateModelUseCase(
        MemoryStore({"test": test}), results, MemoryModels(), AnomalyEvaluator(), NoopRenderer()
    ).execute()
    ev = doc["evaluation"]
    assert ev["confusion_matrix"] == [[3, 0], [1, 2]]  # TN FP / FN TP
    assert (ev["precision"], round(ev["recall"], 3)) == (1.0, 0.667)
    assert (ev["events_total"], ev["events_detected"]) == (2, 1)
    assert "evaluation_metrics" in results.docs and "test_predictions" in results.frames
    assert np.isclose(results.frames["test_predictions"]["anomaly_score"].iloc[1], 9.0)
