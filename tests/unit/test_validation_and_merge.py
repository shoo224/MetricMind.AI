import numpy as np
import pandas as pd
import pytest

from metricmind.application.services.metric_merger import MetricMerger
from metricmind.application.services.validation import SeriesValidator
from metricmind.domain.entities.metric import MetricSeries
from metricmind.domain.exceptions import DataValidationError
from tests.conftest import T0, make_series


def test_validator_reports_clean_series():
    r = SeriesValidator(15).validate(make_series(n=5))
    assert (r.n_duplicate_timestamps, r.n_missing_values, r.is_sorted) == (0, 0, True)
    assert r.dominant_interval_seconds == 15


def test_validator_rejects_duplicates_by_default():
    s = make_series(values=[1, 2, 3])
    dup = MetricSeries(s.metric, np.append(s.timestamps, s.timestamps[-1]), np.append(s.values, 9))
    with pytest.raises(DataValidationError, match="duplicate"):
        SeriesValidator(15).validate(dup)
    assert SeriesValidator(15, "first").validate(dup).n_duplicate_timestamps == 1


def test_validator_rejects_wrong_interval_and_empty():
    with pytest.raises(DataValidationError, match="interval"):
        SeriesValidator(15).validate(make_series(step=60))
    with pytest.raises(DataValidationError, match="empty"):
        SeriesValidator(15).validate(make_series(values=[]))


def test_merge_aligns_on_timestamp():
    a, b = make_series("a", [1, 2, 3]), make_series("b", [10, 20, 30])
    r = MetricMerger(15).merge([a, b])
    assert list(r.frame.columns) == ["a", "b"] and len(r.frame) == 3
    assert r.frame["b"].tolist() == [10, 20, 30]
    assert r.frame.index.is_monotonic_increasing


def test_merge_outer_join_keeps_nonoverlapping_rows_as_nan():
    a = make_series("a", [1, 2, 3])
    b = make_series("b", [7, 8], start=T0 + pd.Timedelta(seconds=15))
    f = MetricMerger(15).merge([a, b]).frame
    assert len(f) == 3 and np.isnan(f["b"].iloc[0]) and f["b"].iloc[1] == 7


def test_merge_inserts_explicit_gap_rows():
    ts = np.array(
        [T0, T0 + pd.Timedelta(seconds=15), T0 + pd.Timedelta(seconds=60)], dtype="datetime64[ns]"
    )
    s = MetricSeries(make_series("a").metric, ts, np.array([1.0, 2.0, 3.0]))
    r = MetricMerger(15).merge([s])
    assert r.n_inserted_gap_rows == 2 and len(r.frame) == 5
    assert r.frame["a"].isna().sum() == 2


def test_merge_sorts_unsorted_input():
    s = make_series("a", [3, 2, 1])
    shuffled = MetricSeries(s.metric, s.timestamps[::-1], s.values[::-1])
    assert MetricMerger(15).merge([shuffled]).frame["a"].tolist() == [3, 2, 1]


@pytest.mark.parametrize("policy, expected", [("first", 1.0), ("last", 9.0), ("mean", 5.0)])
def test_merge_duplicate_policies(policy, expected):
    s = make_series("a", [1, 2])
    dup = MetricSeries(
        s.metric, np.insert(s.timestamps, 1, s.timestamps[0]), np.array([1.0, 9.0, 2.0])
    )
    r = MetricMerger(15, policy).merge([dup])
    assert r.frame["a"].iloc[0] == expected and r.n_duplicates_resolved == 1


def test_merge_duplicate_error_and_name_clash():
    s = make_series("a", [1, 2])
    dup = MetricSeries(
        s.metric, np.append(s.timestamps, s.timestamps[0]), np.array([1.0, 2.0, 3.0])
    )
    with pytest.raises(DataValidationError):
        MetricMerger(15, "error").merge([dup])
    with pytest.raises(DataValidationError, match="Duplicate metric names"):
        MetricMerger(15).merge([s, s])
