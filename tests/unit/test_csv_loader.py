import numpy as np
import pytest

from metricmind.domain.entities.metric import Metric
from metricmind.domain.exceptions import DataValidationError
from metricmind.infrastructure.data.csv_metric_loader import CSVMetricDataLoader

METRIC = Metric("cpu_usage", "cpu.csv", "cpu")


def loader(tmp_path, sentinels=()):
    return CSVMetricDataLoader(tmp_path, "Time", "Value", "%Y-%m-%d %H:%M:%S", sentinels)


def test_loads_valid_file_with_quoted_header(tmp_path, write_csv):
    write_csv("cpu.csv", '"Time","Value"\r\n2025-01-01 00:00:00,2\r\n2025-01-01 00:00:15,3\r\n')
    s = loader(tmp_path).load(METRIC)
    assert len(s) == 2
    assert s.values.tolist() == [2.0, 3.0]
    assert str(s.timestamps[1]).startswith("2025-01-01T00:00:15")


def test_sentinel_becomes_nan(tmp_path, write_csv):
    write_csv("cpu.csv", "Time,Value\n2025-01-01 00:00:00,5\n2025-01-01 00:00:15,-99999\n")
    s = loader(tmp_path, (-99999.0,)).load(METRIC)
    assert np.isnan(s.values[1]) and s.values[0] == 5


def test_blank_value_is_missing_not_error(tmp_path, write_csv):
    write_csv("cpu.csv", "Time,Value\n2025-01-01 00:00:00,\n2025-01-01 00:00:15,1\n")
    assert np.isnan(loader(tmp_path).load(METRIC).values[0])


@pytest.mark.parametrize(
    "text, message",
    [
        ("Time,Value\n", "empty"),
        ("Time,Other\n2025-01-01 00:00:00,1\n", "missing column"),
        ("Time,Value\nnot-a-date,1\n", "timestamp"),
        ("Time,Value\n2025-01-01 00:00:00,abc\n", "non-numeric"),
        ("Time,Value\n,1\n", "timestamp"),
    ],
)
def test_invalid_files_raise_clear_errors(tmp_path, write_csv, text, message):
    write_csv("cpu.csv", text)
    with pytest.raises(DataValidationError, match=message):
        loader(tmp_path).load(METRIC)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        loader(tmp_path).load(METRIC)
