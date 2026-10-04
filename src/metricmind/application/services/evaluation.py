"""AnomalyEvaluator: computes detection metrics from first principles (no sklearn needed).

Metrics are derived from the confusion matrix so the formulas are visible and explainable;
the unit tests cross-check them against scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from metricmind.domain.value_objects.evaluation import EvaluationReport


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def roc_auc(y_true: NDArray, scores: NDArray) -> float | None:
    """AUC = P(random anomaly scores higher than random normal), via the rank-sum identity."""
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: NDArray, scores: NDArray) -> float | None:
    """Area under the precision-recall curve (step-wise): mean precision at each true anomaly.

    Rows are ranked by descending score; tied scores are processed as one block."""
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    frame = (
        pd.DataFrame({"y": y_true, "s": scores}).groupby("s").agg(pos=("y", "sum"), n=("y", "size"))
    )
    frame = frame.sort_index(ascending=False)
    tp = frame["pos"].cumsum()
    precision = tp / frame["n"].cumsum()
    return float((precision * frame["pos"]).sum() / n_pos)


class AnomalyEvaluator:
    def evaluate(
        self,
        y_true: NDArray[np.int_],
        y_pred: NDArray[np.int_],
        scores: NDArray[np.float64] | None = None,
        categories: NDArray | None = None,
        event_ids: NDArray | None = None,
    ) -> EvaluationReport:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        n = len(y_true)

        ev_total = ev_hit = ev_recall = None
        if event_ids is not None:
            ev = pd.DataFrame({"id": event_ids, "hit": y_pred})
            ev = ev[ev["id"] != ""]
            ev_total = int(ev["id"].nunique())
            ev_hit = int((ev.groupby("id")["hit"].max() == 1).sum())
            ev_recall = _safe_div(ev_hit, ev_total) if ev_total else None

        by_cat: dict[str, float] = {}
        if categories is not None:
            cats = np.asarray(categories)
            for cat in sorted(set(cats) - {"none"}):
                mask = cats == cat
                by_cat[str(cat)] = _safe_div(int((y_pred[mask] == 1).sum()), int(mask.sum()))

        return EvaluationReport(
            n_samples=n,
            n_true_anomalies=int(y_true.sum()),
            n_detected_anomalies=int(y_pred.sum()),
            detected_anomaly_rate=_safe_div(int(y_pred.sum()), n),
            true_anomaly_rate=_safe_div(int(y_true.sum()), n),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=_safe_div(2 * precision * recall, precision + recall),
            accuracy=_safe_div(tp + tn, n),
            roc_auc=roc_auc(y_true, scores) if scores is not None else None,
            average_precision=average_precision(y_true, scores) if scores is not None else None,
            event_recall=ev_recall,
            events_total=ev_total,
            events_detected=ev_hit,
            recall_by_category=by_cat,
        )

    def flag_rate_sensitivity(
        self,
        y_true: NDArray[np.int_],
        scores: NDArray[np.float64],
        rates: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
    ) -> pd.DataFrame:
        """Precision/recall/F1 if the threshold were set so that ``rate`` of rows are flagged.

        FOR ANALYSIS ONLY. It uses test labels, so the best row must NOT be used to choose the
        deployed setting (that would be tuning on the test set). It shows the trade-off the
        single ``contamination`` hyper-parameter controls.
        """
        rows = []
        for rate in rates:
            threshold = float(np.quantile(scores, 1 - rate))
            pred = (scores > threshold).astype(int)
            rep = self.evaluate(y_true, pred)
            rows.append(
                {
                    "flag_rate": rate,
                    "precision": rep.precision,
                    "recall": rep.recall,
                    "f1": rep.f1,
                    "detected": rep.n_detected_anomalies,
                }
            )
        return pd.DataFrame(rows).set_index("flag_rate")
