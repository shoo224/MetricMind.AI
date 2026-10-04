"""PrepareDataUseCase: raw metric files -> validated, merged, cleaned, labelled, split datasets."""

from __future__ import annotations

import logging
from typing import Any

from metricmind.application.ports import NullReporter, ProgressReporter, TabularStore
from metricmind.application.services.labeling import GroundTruthLabeler
from metricmind.application.services.metric_merger import MetricMerger
from metricmind.application.services.missing_values import MissingValueHandler
from metricmind.application.services.splitting import ChronologicalSplitter
from metricmind.application.services.validation import SeriesValidator
from metricmind.domain.entities.metric import Metric
from metricmind.domain.ports.data_loader import EventLabelLoader, MetricDataLoader
from metricmind.domain.value_objects.chronological_split import ChronologicalSplit

logger = logging.getLogger(__name__)

FULL, TRAIN, TEST, SUMMARY = "metrics", "train", "test", "dataset_summary"


class PrepareDataUseCase:
    def __init__(
        self,
        metrics: list[Metric],
        loader: MetricDataLoader,
        event_loader: EventLabelLoader,
        validator: SeriesValidator,
        merger: MetricMerger,
        missing_handler: MissingValueHandler,
        labeler: GroundTruthLabeler,
        splitter: ChronologicalSplitter,
        split_rule: ChronologicalSplit,
        store: TabularStore,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._metrics = metrics
        self._loader = loader
        self._event_loader = event_loader
        self._validator = validator
        self._merger = merger
        self._missing = missing_handler
        self._labeler = labeler
        self._splitter = splitter
        self._split_rule = split_rule
        self._store = store
        self._out = reporter or NullReporter()

    def execute(self) -> dict[str, Any]:
        out = self._out

        out.section("Loading metrics")
        series_list, validation = [], {}
        for metric in self._metrics:
            series = self._loader.load(metric)
            report = self._validator.validate(series)
            series_list.append(series)
            validation[metric.name] = report.__dict__
            out.step(f"{metric.name}: {len(series):,} rows, {report.n_missing_values} missing")

        out.section("Merging metrics on timestamp")
        merged = self._merger.merge(series_list)
        out.step(
            f"{merged.n_timestamps:,} timestamps aligned "
            f"({merged.n_inserted_gap_rows} gap rows inserted)"
        )
        out.step(f"{merged.n_duplicates_resolved} duplicate timestamps")

        out.section("Handling missing values")
        frame, miss = self._missing.handle(merged.frame)
        out.step(
            f"{miss.missing_before} missing -> {miss.filled} forward-filled, "
            f"{miss.rows_dropped} rows dropped"
        )

        out.section("Attaching ground-truth labels")
        events = self._event_loader.load_events()
        frame = self._labeler.label(frame, events)
        n_anom = int(frame["is_anomaly"].sum())
        out.step(
            f"{len(events)} fault events -> {n_anom:,} labelled rows "
            f"({100 * n_anom / len(frame):.1f}%)"
        )

        out.section("Chronological train/test split")
        train, test = self._splitter.split(frame, self._split_rule)
        out.step(f"train: {len(train):,} rows  [{train.index.min()} -> {train.index.max()}]")
        out.step(f"test : {len(test):,} rows  [{test.index.min()} -> {test.index.max()}]")

        out.section("Saving")
        paths = [
            self._store.save_frame(n, f) for n, f in ((FULL, frame), (TRAIN, train), (TEST, test))
        ]

        label_cols = ["is_anomaly", "anomaly_category", "event_id"]
        summary = {
            "rows": len(frame),
            "metric_columns": [c for c in frame.columns if c not in label_cols],
            "start": frame.index.min(),
            "end": frame.index.max(),
            "interval_seconds": self._merger_interval(frame),
            "validation": validation,
            "merge": {k: v for k, v in merged.__dict__.items() if k != "frame"},
            "missing_values": miss.__dict__,
            "labels": {
                "events": len(events),
                "anomalous_rows": n_anom,
                "anomalous_fraction": n_anom / len(frame),
                "rows_by_category": frame["anomaly_category"].value_counts().to_dict(),
            },
            "split": {
                "train_fraction": self._split_rule.train_fraction,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_anomalous_fraction": float(train["is_anomaly"].mean()),
                "test_anomalous_fraction": float(test["is_anomaly"].mean()),
                "train_end": train.index.max(),
                "test_start": test.index.min(),
            },
            "files": [str(p) for p in paths],
        }
        self._store.save_json(SUMMARY, summary)
        for p in paths:
            out.step(str(p))
        return summary

    @staticmethod
    def _merger_interval(frame) -> float:
        return float(frame.index.to_series().diff().dt.total_seconds().median())
