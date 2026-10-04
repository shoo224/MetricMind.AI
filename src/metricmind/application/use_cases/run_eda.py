"""RunEdaUseCase: exploratory data analysis - statistics, correlation and time-series figures."""

from __future__ import annotations

from typing import Any

from metricmind.application.ports import (
    FigureRenderer,
    NullReporter,
    ProgressReporter,
    TabularStore,
)
from metricmind.application.use_cases.prepare_data import FULL

LABEL_COLUMNS = ("is_anomaly", "anomaly_category", "event_id")


class RunEdaUseCase:
    def __init__(
        self,
        processed_store: TabularStore,
        results_store: TabularStore,
        renderer: FigureRenderer,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._processed = processed_store
        self._results = results_store
        self._renderer = renderer
        self._out = reporter or NullReporter()

    def execute(self) -> dict[str, Any]:
        frame = self._processed.load_frame(FULL)
        metrics = [c for c in frame.columns if c not in LABEL_COLUMNS]
        out = self._out

        out.section("Summary statistics")
        stats = frame[metrics].describe().T
        stats["labelled_anomaly_mean"] = [
            frame.loc[frame["is_anomaly"] == 1, c].mean() for c in metrics
        ]
        stats["normal_mean"] = [frame.loc[frame["is_anomaly"] == 0, c].mean() for c in metrics]
        stats.index.name = "metric"
        self._results.save_frame("eda_summary_statistics", stats)
        corr = frame[metrics].corr()
        self._results.save_frame("eda_correlation_matrix", corr)
        out.step(f"{len(metrics)} metrics, {len(frame):,} rows summarised")

        out.section("Figures")
        r = self._renderer
        figures = [
            r.metric_timeseries(
                frame,
                ["cpu_usage"],
                "CPU usage over time (shaded = labelled faults)",
                "eda_cpu_usage.png",
            ),
            r.metric_timeseries(
                frame,
                ["memory_usage", "memory_used"],
                "Memory over time (shaded = labelled faults)",
                "eda_memory.png",
            ),
            r.metric_timeseries(
                frame,
                ["response_time_a", "response_time_b", "response_time_diff"],
                "Network response times (shaded = labelled faults)",
                "eda_response_time.png",
            ),
            r.correlation_heatmap(frame, metrics, "eda_correlation_heatmap.png"),
        ]
        for path in figures:
            out.step(str(path))

        high = [
            (a, b, float(corr.loc[a, b]))
            for i, a in enumerate(metrics)
            for b in metrics[i + 1 :]
            if abs(corr.loc[a, b]) > 0.95
        ]
        return {
            "metrics": metrics,
            "highly_correlated_pairs": high,
            "figures": [str(p) for p in figures],
        }
