"""Whole pipeline (prepare -> eda -> train -> evaluate) on a small SYNTHETIC dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from metricmind.application.ports import NullReporter
from metricmind.config.settings import DatasetSpec, Settings
from metricmind.interfaces.cli import wiring

N = 1200
METRICS = {
    "cpu_usage": ("cpu_usage.csv", "cpu"),
    "memory_usage": ("memory_usage.csv", "memory"),
    "memory_used": ("memory_used.csv", "memory"),
    "response_time_a": ("rt_a.csv", "network"),
    "response_time_b": ("rt_b.csv", "network"),
    "response_time_diff": ("rt_diff.csv", "network"),
}


@pytest.fixture
def workspace(tmp_path):
    raw = tmp_path / "raw"
    (raw / "event_logs").mkdir(parents=True)
    ts = pd.date_range("2025-01-01", periods=N, freq="15s")
    rng = np.random.default_rng(0)
    fault = np.zeros(N, bool)
    fault[900:960] = True  # one fault in the later (test) part of the series
    fault[200:240] = True  # and one in the training part
    cpu = rng.normal(2, 0.5, N) + fault * 60
    mem = rng.normal(18, 0.2, N) + fault * 20
    rta, rtb = rng.normal(185, 2, N), rng.normal(550, 20, N)
    rtb[5] = -99999  # a failed probe, as in the real data
    data = {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "memory_used": mem * 77,
        "response_time_a": rta,
        "response_time_b": rtb,
        "response_time_diff": rta - rtb,
    }
    for name, (fname, _) in METRICS.items():
        pd.DataFrame({"Time": ts.strftime("%Y-%m-%d %H:%M:%S"), "Value": data[name]}).to_csv(
            raw / fname, index=False
        )
    ev = pd.DataFrame(
        {
            "id": ["c1", "c2"],
            "flag": "completed",
            "type": "cpu",
            "mode": "stable",
            "intensity": 50,
            "start_time": [
                ts[200].strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                ts[900].strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            ],
            "end_time": [
                ts[239].strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                ts[959].strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            ],
        }
    )
    ev.to_csv(raw / "event_logs" / "cpu.csv", index=False)

    toml = [
        "[format]",
        'timestamp_column="Time"',
        'value_column="Value"',
        'timestamp_format="%Y-%m-%d %H:%M:%S"',
        "expected_interval_seconds=15",
        "missing_sentinels=[-99999.0]",
    ]
    for name, (fname, cat) in METRICS.items():
        toml += ["[[metrics]]", f'name="{name}"', f'file="{fname}"', f'category="{cat}"']
    toml += ["[events]", 'files=["event_logs/cpu.csv"]', 'timezone="UTC"']
    cfg = tmp_path / "metrics.toml"
    cfg.write_text("\n".join(toml))

    settings = Settings(
        _env_file=None,
        data_raw_dir=raw,
        data_processed_dir=tmp_path / "processed",
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        metrics_config_path=cfg,
        if_n_estimators=50,
        if_contamination=0.08,
    )
    return settings, DatasetSpec.from_file(cfg)


def test_full_pipeline_produces_all_artefacts_and_detects_the_fault(workspace):
    settings, spec = workspace
    summary = wiring.build_prepare(settings, spec, NullReporter()).execute()
    assert (
        summary["rows"] == N and summary["missing_values"]["filled"] == 1
    )  # the one failed probe was repaired
    assert summary["split"]["train_end"] < summary["split"]["test_start"]

    eda = wiring.build_eda(settings, NullReporter()).execute()
    assert (settings.figures_dir / "eda_correlation_heatmap.png").is_file()
    assert any(
        {"memory_usage", "memory_used"} == {a, b} for a, b, _ in eda["highly_correlated_pairs"]
    )

    wiring.build_train(settings, NullReporter()).execute()
    for f in ("scaler.joblib", "isolation_forest.joblib", "model_metadata.json"):
        assert (settings.model_dir / f).is_file()

    doc = wiring.build_evaluate(settings, NullReporter()).execute()
    ev = doc["evaluation"]
    assert ev["recall"] > 0.8 and ev["precision"] > 0.8  # an obvious fault must be found
    assert ev["events_detected"] == ev["events_total"] == 1
    saved = json.loads((settings.results_dir / "evaluation_metrics.json").read_text())
    assert saved["evaluation"]["confusion_matrix"] == ev["confusion_matrix"]
    assert (settings.figures_dir / "confusion_matrix.png").is_file()


def test_evaluation_uses_saved_model_without_retraining(workspace):
    settings, spec = workspace
    wiring.build_prepare(settings, spec, NullReporter()).execute()
    wiring.build_train(settings, NullReporter()).execute()
    a = wiring.build_evaluate(settings, NullReporter()).execute()
    b = wiring.build_evaluate(settings, NullReporter()).execute()
    assert a["evaluation"] == b["evaluation"]
