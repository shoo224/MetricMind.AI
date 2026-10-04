"""Composition root: the ONE place where concrete classes are chosen and connected.

Everything else depends on abstractions; swapping Isolation Forest for an LSTM autoencoder (or CSV
for TimescaleDB) means changing this file, not the use cases.
"""

from __future__ import annotations

from metricmind.application.ports import ProgressReporter
from metricmind.application.services.evaluation import AnomalyEvaluator
from metricmind.application.services.labeling import GroundTruthLabeler
from metricmind.application.services.metric_merger import MetricMerger
from metricmind.application.services.missing_values import MissingValueHandler
from metricmind.application.services.preprocessing import FeaturePreprocessor
from metricmind.application.services.splitting import ChronologicalSplitter
from metricmind.application.services.validation import SeriesValidator
from metricmind.application.use_cases.evaluate_model import EvaluateModelUseCase
from metricmind.application.use_cases.prepare_data import PrepareDataUseCase
from metricmind.application.use_cases.run_eda import RunEdaUseCase
from metricmind.application.use_cases.train_model import TrainModelUseCase
from metricmind.config.logging_config import setup_logging
from metricmind.config.settings import DatasetSpec, Settings, get_settings
from metricmind.domain.value_objects.chronological_split import ChronologicalSplit
from metricmind.infrastructure.data.csv_event_loader import CSVEventLabelLoader
from metricmind.infrastructure.data.csv_metric_loader import CSVMetricDataLoader
from metricmind.infrastructure.ml.isolation_forest_detector import IsolationForestDetector
from metricmind.infrastructure.ml.sklearn_scaler import SklearnStandardScaler
from metricmind.infrastructure.persistence.csv_json_store import CsvJsonStore
from metricmind.infrastructure.persistence.joblib_model_repository import JoblibModelRepository
from metricmind.infrastructure.visualization.matplotlib_renderer import MatplotlibFigureRenderer


def bootstrap(verbose: bool = False) -> tuple[Settings, DatasetSpec]:
    settings = get_settings()
    setup_logging("INFO" if verbose else settings.log_level, settings.logs_dir / "metricmind.log")
    return settings, DatasetSpec.from_file(settings.metrics_config_path)


def build_prepare(
    settings: Settings, spec: DatasetSpec, out: ProgressReporter
) -> PrepareDataUseCase:
    return PrepareDataUseCase(
        metrics=spec.metrics,
        loader=CSVMetricDataLoader(
            settings.data_raw_dir,
            spec.timestamp_column,
            spec.value_column,
            spec.timestamp_format,
            spec.missing_sentinels,
        ),
        event_loader=CSVEventLabelLoader(settings.data_raw_dir, spec.events),
        validator=SeriesValidator(spec.expected_interval_seconds, settings.duplicate_policy),
        merger=MetricMerger(spec.expected_interval_seconds, settings.duplicate_policy),
        missing_handler=MissingValueHandler(settings.max_ffill_steps),
        labeler=GroundTruthLabeler(),
        splitter=ChronologicalSplitter(),
        split_rule=ChronologicalSplit(settings.train_fraction),
        store=CsvJsonStore(settings.data_processed_dir),
        reporter=out,
    )


def build_eda(settings: Settings, out: ProgressReporter) -> RunEdaUseCase:
    return RunEdaUseCase(
        CsvJsonStore(settings.data_processed_dir),
        CsvJsonStore(settings.results_dir),
        MatplotlibFigureRenderer(settings.figures_dir),
        out,
    )


def build_train(settings: Settings, out: ProgressReporter) -> TrainModelUseCase:
    return TrainModelUseCase(
        processed_store=CsvJsonStore(settings.data_processed_dir),
        preprocessor=FeaturePreprocessor(SklearnStandardScaler(), settings.model_features),
        detector=IsolationForestDetector(
            n_estimators=settings.if_n_estimators,
            contamination=settings.if_contamination,
            max_samples=settings.if_max_samples,
            random_state=settings.if_random_state,
        ),
        models=JoblibModelRepository(settings.model_dir),
        reporter=out,
    )


def build_evaluate(settings: Settings, out: ProgressReporter) -> EvaluateModelUseCase:
    return EvaluateModelUseCase(
        processed_store=CsvJsonStore(settings.data_processed_dir),
        results_store=CsvJsonStore(settings.results_dir),
        models=JoblibModelRepository(settings.model_dir),
        evaluator=AnomalyEvaluator(),
        renderer=MatplotlibFigureRenderer(settings.figures_dir),
        reporter=out,
    )
