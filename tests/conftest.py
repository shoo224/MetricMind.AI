"""Shared synthetic fixtures. Tests never touch the real OS-Metrics dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metricmind.domain.entities.metric import Metric, MetricSeries

T0 = pd.Timestamp("2025-01-01 00:00:00")


def make_series(name="cpu_usage", values=None, n=10, step=15, start=T0, category="cpu"):
    values = np.arange(n, dtype=float) if values is None else np.asarray(values, dtype=float)
    ts = (start + pd.to_timedelta(np.arange(len(values)) * step, unit="s")).to_numpy(
        "datetime64[ns]"
    )
    return MetricSeries(Metric(name, f"{name}.csv", category), ts, values)


@pytest.fixture
def write_csv(tmp_path: Path):
    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def small_frame() -> pd.DataFrame:
    idx = pd.date_range(T0, periods=100, freq="15s", name="timestamp")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"cpu_usage": rng.normal(2, 0.5, 100), "memory_usage": rng.normal(18, 0.2, 100)}, index=idx
    )
