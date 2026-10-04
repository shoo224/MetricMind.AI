import numpy as np
import pandas as pd
import pytest

from metricmind.application.services.preprocessing import FeaturePreprocessor
from metricmind.domain.exceptions import DataValidationError, NotFittedError
from metricmind.infrastructure.ml.sklearn_scaler import SklearnStandardScaler


def test_scaler_is_fitted_on_train_only_no_leakage(small_frame):
    train, test = small_frame.iloc[:70], small_frame.iloc[70:].copy()
    test["cpu_usage"] += 100  # a shifted test set must NOT change the learned scaling
    pre = FeaturePreprocessor(SklearnStandardScaler(), ["cpu_usage", "memory_usage"]).fit(train)
    assert np.allclose(pre.transform(train).mean(axis=0), 0, atol=1e-9)
    assert pre.transform(test)[:, 0].mean() > 50  # still huge: scaler did not adapt to test
    assert np.isclose(pre.scaler._scaler.mean_[0], train["cpu_usage"].mean())


def test_transform_before_fit_fails(small_frame):
    pre = FeaturePreprocessor(SklearnStandardScaler(), ["cpu_usage"])
    with pytest.raises(NotFittedError):
        pre.transform(small_frame)


def test_missing_column_and_nan_rejected(small_frame):
    pre = FeaturePreprocessor(SklearnStandardScaler(), ["nope"])
    with pytest.raises(DataValidationError, match="Missing feature"):
        pre.fit(small_frame)
    bad = small_frame.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        FeaturePreprocessor(SklearnStandardScaler(), ["cpu_usage"]).fit(bad)


def test_only_selected_features_are_used(small_frame):
    pre = FeaturePreprocessor(SklearnStandardScaler(), ["memory_usage"]).fit(small_frame)
    assert pre.transform(small_frame).shape == (len(small_frame), 1)
    assert isinstance(small_frame, pd.DataFrame)
