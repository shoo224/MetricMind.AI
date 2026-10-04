"""Application settings, read from environment variables / .env (see .env.example).

Relative paths are resolved against the project root, so nothing is hard-coded to one machine.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from metricmind.domain.entities.metric import Metric


def _find_project_root() -> Path:
    if env := os.environ.get("PROJECT_ROOT"):
        return Path(env).resolve()
    # src/metricmind/config/settings.py -> parents[3] is the repository root
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """All tunables in one place. Field NAME == environment variable name (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_raw_dir: Path = Path("data/raw/OSMetrics")
    data_processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    metrics_config_path: Path = Path("configs/metrics.toml")

    log_level: str = "WARNING"

    train_fraction: float = Field(0.7, gt=0, lt=1)
    max_ffill_steps: int = Field(4, ge=0)
    duplicate_policy: Literal["error", "first", "last", "mean"] = "error"

    # Redundant columns (memory_used ~ 77*memory_usage; response_time_diff = a - b) are excluded.
    model_features: list[str] = [
        "cpu_usage",
        "memory_usage",
        "response_time_a",
        "response_time_b",
    ]

    if_n_estimators: int = Field(200, gt=0)
    if_contamination: float = Field(0.05, gt=0, le=0.5)
    if_max_samples: str = "auto"  # "auto" or an integer / fraction given as text
    if_random_state: int = 42

    @field_validator(
        "data_raw_dir", "data_processed_dir", "model_dir", "report_dir", "metrics_config_path"
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def figures_dir(self) -> Path:
        return self.report_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.report_dir / "results"

    @property
    def logs_dir(self) -> Path:
        return self.report_dir / "logs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class DatasetSpec:
    """Dataset description parsed from ``configs/metrics.toml`` (dataset facts, not tunables)."""

    def __init__(self, raw: dict) -> None:
        fmt = raw["format"]
        self.timestamp_column: str = fmt["timestamp_column"]
        self.value_column: str = fmt["value_column"]
        self.timestamp_format: str = fmt["timestamp_format"]
        self.expected_interval_seconds: int = int(fmt["expected_interval_seconds"])
        self.missing_sentinels: tuple[float, ...] = tuple(fmt.get("missing_sentinels", []))
        self.metrics: list[Metric] = [
            Metric(
                name=m["name"],
                source=m["file"],
                category=m["category"],
                unit=m.get("unit", "unspecified"),
                description=m.get("description", ""),
            )
            for m in raw["metrics"]
        ]
        self.events: dict = raw.get("events", {})

    @classmethod
    def from_file(cls, path: Path) -> DatasetSpec:
        with path.open("rb") as handle:
            return cls(tomllib.load(handle))
