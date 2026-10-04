import numpy as np
import pytest

from metricmind.domain.exceptions import ModelNotFoundError, NotFittedError
from metricmind.infrastructure.ml.isolation_forest_detector import (
    IsolationForestDetector,
    parse_max_samples,
)
from metricmind.infrastructure.persistence.joblib_model_repository import JoblibModelRepository


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, (500, 3))
    outliers = rng.normal(8, 0.5, (10, 3))
    return normal, outliers


def test_outliers_score_higher_and_are_flagged(data):
    normal, outliers = data
    det = IsolationForestDetector(contamination=0.02, random_state=0).fit(normal)
    assert det.score_samples(outliers).mean() > det.score_samples(normal).mean()
    assert det.predict(outliers).mean() > 0.9  # convention: 1 = anomaly


def test_predict_is_consistent_with_score_and_threshold(data):
    normal, outliers = data
    det = IsolationForestDetector(random_state=0).fit(normal)
    x = np.vstack([normal, outliers])
    assert np.array_equal(det.predict(x), (det.score_samples(x) > det.threshold).astype(int))


def test_contamination_controls_training_flag_rate(data):
    normal, _ = data
    for c in (0.02, 0.10):
        det = IsolationForestDetector(contamination=c, random_state=0).fit(normal)
        assert abs(det.predict(normal).mean() - c) < 0.02


def test_same_seed_is_deterministic(data):
    normal, _ = data
    a = IsolationForestDetector(random_state=3).fit(normal).score_samples(normal)
    b = IsolationForestDetector(random_state=3).fit(normal).score_samples(normal)
    assert np.allclose(a, b)


def test_use_before_fit_raises():
    with pytest.raises(NotFittedError):
        IsolationForestDetector().predict(np.zeros((2, 2)))


def test_hyperparameters_are_configurable_and_exposed():
    p = IsolationForestDetector(n_estimators=10, contamination=0.1, max_samples="64").get_params()
    assert p["n_estimators"] == 10 and p["contamination"] == 0.1 and p["max_samples"] == 64
    assert parse_max_samples("auto") == "auto" and parse_max_samples("0.5") == 0.5


def test_model_roundtrip_gives_identical_predictions(tmp_path, data):
    normal, outliers = data
    repo = JoblibModelRepository(tmp_path / "models")
    det = IsolationForestDetector(random_state=0).fit(normal)
    repo.save("det", det)
    repo.save_metadata("meta", {"feature_columns": ["a", "b", "c"]})
    loaded = repo.load("det")
    x = np.vstack([normal, outliers])
    assert np.array_equal(det.predict(x), loaded.predict(x))
    assert np.allclose(det.score_samples(x), loaded.score_samples(x))
    assert repo.load_metadata("meta")["library_versions"]["scikit-learn"]


def test_missing_artefact_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train_model"):
        JoblibModelRepository(tmp_path).load("nothing")
