from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from metricmind.application.services.labeling import GroundTruthLabeler
from metricmind.application.services.missing_values import MissingValueHandler
from metricmind.application.services.splitting import ChronologicalSplitter
from metricmind.domain.entities.event import LabeledEvent
from metricmind.domain.exceptions import DataValidationError
from metricmind.domain.value_objects.chronological_split import ChronologicalSplit
from tests.conftest import T0


def frame(values):
    idx = pd.date_range(T0, periods=len(values), freq="15s", name="timestamp")
    return pd.DataFrame({"x": values}, index=idx)


def test_ffill_respects_limit_and_drops_remaining():
    f, rep = MissingValueHandler(max_ffill_steps=2).handle(frame([1, np.nan, np.nan, np.nan, 5]))
    assert f["x"].tolist() == [1, 1, 1, 5]  # 3rd NaN not filled -> row dropped
    assert (rep.missing_before, rep.filled, rep.rows_dropped, rep.missing_after) == (3, 2, 1, 0)


def test_ffill_is_causal_leading_nan_is_dropped_not_backfilled():
    f, rep = MissingValueHandler(4).handle(frame([np.nan, 2, 3]))
    assert f["x"].tolist() == [2, 3] and rep.rows_dropped == 1


def test_no_missing_is_untouched():
    f, rep = MissingValueHandler(4).handle(frame([1, 2, 3]))
    assert f["x"].tolist() == [1, 2, 3] and rep.missing_before == 0


def test_labeler_marks_inclusive_interval_and_records_category():
    ev = LabeledEvent("e1", "cpu", datetime(2025, 1, 1, 0, 0, 15), datetime(2025, 1, 1, 0, 0, 45))
    out = GroundTruthLabeler().label(frame(range(6)), [ev])
    assert out["is_anomaly"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (
        out["anomaly_category"].tolist()[1] == "cpu" and out["anomaly_category"].iloc[0] == "none"
    )
    assert out["event_id"].iloc[2] == "e1" and out["event_id"].iloc[5] == ""


def test_labeler_event_between_samples_and_outside_range():
    inside = LabeledEvent(
        "a", "cpu", datetime(2025, 1, 1, 0, 0, 16), datetime(2025, 1, 1, 0, 0, 29)
    )
    outside = LabeledEvent("b", "cpu", datetime(2030, 1, 1), datetime(2030, 1, 2))
    out = GroundTruthLabeler().label(frame(range(4)), [inside, outside])
    assert out["is_anomaly"].sum() == 0  # no invented labels


def test_event_end_before_start_rejected():
    with pytest.raises(DataValidationError):
        LabeledEvent("x", "cpu", datetime(2025, 1, 2), datetime(2025, 1, 1))


def test_split_is_chronological_with_no_overlap():
    f = frame(range(10))
    train, test = ChronologicalSplitter().split(f, ChronologicalSplit(0.7))
    assert len(train) == 7 and len(test) == 3
    assert train.index.max() < test.index.min()
    assert train["x"].tolist() == list(range(7))  # not shuffled


def test_split_rejects_bad_fraction_and_unsorted_input():
    with pytest.raises(DataValidationError):
        ChronologicalSplit(1.0)
    with pytest.raises(DataValidationError):
        ChronologicalSplitter().split(frame(range(5)).iloc[::-1], ChronologicalSplit(0.5))
