"""EvaluationReport: immutable summary of how predictions compare with ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationReport:
    """Point-level metrics (positive class = anomaly) plus threshold-free and event-level views."""

    n_samples: int
    n_true_anomalies: int
    n_detected_anomalies: int
    detected_anomaly_rate: float
    true_anomaly_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float  # reported for completeness only: misleading when anomalies are rare
    roc_auc: float | None = None  # threshold-free ranking quality of the scores
    average_precision: float | None = None  # area under precision-recall curve
    event_recall: float | None = None  # fraction of labelled events with >= 1 detection
    events_total: int | None = None
    events_detected: int | None = None
    recall_by_category: dict[str, float] = field(default_factory=dict)

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """[[TN, FP], [FN, TP]] - rows are the actual class, columns the predicted class."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confusion_matrix"] = self.confusion_matrix
        return data
