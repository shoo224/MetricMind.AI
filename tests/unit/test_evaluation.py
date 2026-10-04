import numpy as np
import pytest
from sklearn import metrics as skm

from metricmind.application.services.evaluation import AnomalyEvaluator, average_precision, roc_auc


@pytest.fixture
def case():
    rng = np.random.default_rng(7)
    y = (rng.random(400) < 0.2).astype(int)
    scores = y * 0.6 + rng.random(400)  # informative but imperfect
    pred = (scores > 0.9).astype(int)
    return y, pred, scores


def test_matches_scikit_learn(case):
    y, pred, scores = case
    r = AnomalyEvaluator().evaluate(y, pred, scores)
    assert r.precision == pytest.approx(skm.precision_score(y, pred))
    assert r.recall == pytest.approx(skm.recall_score(y, pred))
    assert r.f1 == pytest.approx(skm.f1_score(y, pred))
    assert r.accuracy == pytest.approx(skm.accuracy_score(y, pred))
    assert r.roc_auc == pytest.approx(skm.roc_auc_score(y, scores))
    assert r.average_precision == pytest.approx(skm.average_precision_score(y, scores))
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert r.confusion_matrix == [[tn, fp], [fn, tp]]


def test_ties_in_scores_are_handled_like_scikit_learn():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.5, 0.5, 0.9, 0.1, 0.5, 0.1])
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))


def test_counts_and_rates():
    r = AnomalyEvaluator().evaluate(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert (r.n_detected_anomalies, r.n_true_anomalies) == (1, 2)
    assert r.detected_anomaly_rate == 0.25 and r.true_anomaly_rate == 0.5
    assert r.roc_auc is None  # no scores supplied


def test_zero_division_is_safe():
    r = AnomalyEvaluator().evaluate(np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)
    only_normal = AnomalyEvaluator().evaluate(np.zeros(3, int), np.zeros(3, int), np.arange(3.0))
    assert only_normal.roc_auc is None and only_normal.average_precision is None


def test_event_level_and_category_recall():
    y = np.array([1, 1, 1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0, 0, 0])
    ids = np.array(["e1", "e1", "e2", "e2", "", ""], dtype=object)
    cats = np.array(["cpu", "cpu", "memory", "memory", "none", "none"], dtype=object)
    r = AnomalyEvaluator().evaluate(y, pred, event_ids=ids, categories=cats)
    assert (r.events_total, r.events_detected, r.event_recall) == (2, 1, 0.5)
    assert r.recall_by_category == {"cpu": 0.5, "memory": 0.0}


def test_flag_rate_sensitivity_flags_requested_fraction(case):
    y, _, scores = case
    table = AnomalyEvaluator().flag_rate_sensitivity(y, scores, rates=(0.1, 0.5))
    assert table.loc[0.1, "detected"] == pytest.approx(40, abs=2)
    assert table.loc[0.5, "recall"] >= table.loc[0.1, "recall"]  # flag more -> recall cannot drop
