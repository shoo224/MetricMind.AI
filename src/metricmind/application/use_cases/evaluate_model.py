"""EvaluateModelUseCase: score the held-out (later-in-time) test set and measure quality."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from metricmind.application.ports import (
    FigureRenderer,
    NullReporter,
    ProgressReporter,
    TabularStore,
)
from metricmind.application.services.evaluation import AnomalyEvaluator
from metricmind.application.services.preprocessing import FeaturePreprocessor
from metricmind.application.use_cases.prepare_data import TEST
from metricmind.application.use_cases.train_model import (
    DETECTOR_ARTEFACT,
    METADATA_ARTEFACT,
    SCALER_ARTEFACT,
)
from metricmind.domain.entities.anomaly import AnomalyResult
from metricmind.domain.ports.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class EvaluateModelUseCase:
    def __init__(
        self,
        processed_store: TabularStore,
        results_store: TabularStore,
        models: ModelRepository,
        evaluator: AnomalyEvaluator,
        renderer: FigureRenderer,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._processed = processed_store
        self._results = results_store
        self._models = models
        self._evaluator = evaluator
        self._renderer = renderer
        self._out = reporter or NullReporter()

    def execute(self) -> dict[str, Any]:
        out = self._out
        out.section("Loading trained model (no retraining)")
        meta = self._models.load_metadata(METADATA_ARTEFACT)
        detector = self._models.load(DETECTOR_ARTEFACT)
        scaler = self._models.load(SCALER_ARTEFACT)
        features_cols = meta["feature_columns"]  # the model's own feature list, not today's config
        pre = FeaturePreprocessor(scaler, features_cols)
        out.step(
            f"{detector.name}, trained {meta['trained_at']}, features: {', '.join(features_cols)}"
        )

        out.section("Scoring test set")
        test = self._processed.load_frame(TEST)
        features = pre.transform(test)  # transform ONLY: scaler learned from train
        scores = detector.score_samples(features)
        result = AnomalyResult(
            detector.name,
            detector.threshold,
            test.index.to_numpy(),
            scores,
            detector.predict(features),
        )
        out.step(
            f"{len(test):,} rows; {result.anomaly_count:,} flagged "
            f"({100 * result.anomaly_rate:.2f}%)"
        )

        out.section("Evaluating against labelled fault events")
        report = self._evaluator.evaluate(
            test["is_anomaly"].to_numpy(),
            result.predictions,
            scores,
            categories=test["anomaly_category"].to_numpy(),
            event_ids=test["event_id"].to_numpy(),
        )
        out.step(
            f"precision {report.precision:.3f} | recall {report.recall:.3f} | F1 {report.f1:.3f}"
        )
        out.step(
            f"ROC-AUC {report.roc_auc:.3f} | avg precision {report.average_precision:.3f} "
            f"(true anomaly rate {100 * report.true_anomaly_rate:.1f}%)"
        )
        out.step(
            f"events detected: {report.events_detected}/{report.events_total}; "
            f"recall by category: "
            + ", ".join(f"{k} {v:.2f}" for k, v in report.recall_by_category.items())
        )

        sensitivity = self._evaluator.flag_rate_sensitivity(test["is_anomaly"].to_numpy(), scores)

        out.section("Saving results and figures")
        document = {
            "model": meta,
            "evaluation": report.to_dict(),
            "threshold": result.threshold,
            "flag_rate_sensitivity_analysis_only": sensitivity.reset_index().to_dict("records"),
        }
        paths = [self._results.save_json("evaluation_metrics", document)]
        predictions = pd.DataFrame(
            {
                "anomaly_score": scores,
                "predicted_anomaly": result.predictions,
                "is_anomaly": test["is_anomaly"],
                "anomaly_category": test["anomaly_category"],
                "event_id": test["event_id"],
            },
            index=test.index,
        )
        predictions.index.name = "timestamp"
        paths.append(self._results.save_frame("test_predictions", predictions))
        paths.append(self._results.save_frame("flag_rate_sensitivity", sensitivity))

        r = self._renderer
        paths.append(
            r.confusion_matrix(
                report.confusion_matrix,
                "Confusion matrix (test set, point level)",
                "confusion_matrix.png",
            )
        )
        paths.append(
            r.score_timeline(
                test,
                scores,
                result.threshold,
                "Anomaly score over time (test set)",
                "anomaly_scores.png",
            )
        )
        for col in features_cols:
            paths.append(
                r.anomaly_overlay(
                    test,
                    col,
                    result.predictions,
                    f"Detected anomalies on {col} (test set)",
                    f"anomalies_{col}.png",
                )
            )
        for p in paths:
            out.step(str(p))
        return document
