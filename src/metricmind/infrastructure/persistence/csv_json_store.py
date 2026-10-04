"""CsvJsonStore: TabularStore adapter that keeps DataFrames as CSV and documents as JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from metricmind.application.ports import TabularStore
from metricmind.domain.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_STRING_DEFAULTS = {"event_id": "", "anomaly_category": "none"}  # blank CSV cells -> these


class CsvJsonStore(TabularStore):
    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir

    def save_frame(self, name: str, frame: pd.DataFrame) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.csv"
        frame.to_csv(path, index_label=frame.index.name or "timestamp")
        logger.info("Saved %d rows x %d cols to %s", len(frame), frame.shape[1], path)
        return path

    def load_frame(self, name: str) -> pd.DataFrame:
        path = self._dir / f"{name}.csv"
        if not path.is_file():
            raise DataValidationError(f"{path} not found - run the previous pipeline step first")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        for column, default in _STRING_DEFAULTS.items():
            if column in frame.columns:
                frame[column] = frame[column].fillna(default).astype(str)
        return frame

    def save_json(self, name: str, document: dict[str, Any]) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        path = self._dir / f"{name}.json"
        if not path.is_file():
            raise DataValidationError(f"{path} not found")
        return json.loads(path.read_text(encoding="utf-8"))
