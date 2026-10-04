"""CSVEventLabelLoader: reads the fault-injection event logs (ground truth)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from metricmind.domain.entities.event import LabeledEvent
from metricmind.domain.exceptions import DataValidationError
from metricmind.domain.ports.data_loader import EventLabelLoader

logger = logging.getLogger(__name__)


class CSVEventLabelLoader(EventLabelLoader):
    """Event start/end are ISO-8601 UTC strings; they are converted to the metrics' timezone and
    made timezone-naive so they compare directly with the (naive) metric timestamps."""

    def __init__(self, raw_dir: Path, spec: dict) -> None:
        self._raw_dir = raw_dir
        self._files: list[str] = spec.get("files", [])
        self._cols = {
            "id": spec.get("id_column", "id"),
            "category": spec.get("category_column", "type"),
            "mode": spec.get("mode_column", "mode"),
            "intensity": spec.get("intensity_column", "intensity"),
            "start": spec.get("start_column", "start_time"),
            "end": spec.get("end_column", "end_time"),
        }
        self._timezone = spec.get("timezone", "UTC")

    def load_events(self) -> list[LabeledEvent]:
        events: list[LabeledEvent] = []
        for name in self._files:
            path = self._raw_dir / name
            if not path.is_file():
                raise DataValidationError(f"Event log not found: {path}")
            events.extend(self._read(path))
        events.sort(key=lambda e: e.start)
        logger.info("Loaded %d labelled events from %d file(s)", len(events), len(self._files))
        return events

    def _read(self, path: Path) -> list[LabeledEvent]:
        frame = pd.read_csv(path)
        missing = set(self._cols.values()) - set(frame.columns)
        if missing:
            raise DataValidationError(f"{path.name}: missing column(s) {sorted(missing)}")
        c = self._cols
        start = self._to_local_naive(frame[c["start"]], path.name)
        end = self._to_local_naive(frame[c["end"]], path.name)
        return [
            LabeledEvent(
                event_id=str(row[c["id"]]),
                category=str(row[c["category"]]),
                start=start.iloc[i].to_pydatetime(),
                end=end.iloc[i].to_pydatetime(),
                mode=str(row[c["mode"]]),
                intensity=float(row[c["intensity"]]),
            )
            for i, (_, row) in enumerate(frame.iterrows())
        ]

    def _to_local_naive(self, column: pd.Series, source: str) -> pd.Series:
        parsed = pd.to_datetime(column, utc=True, errors="coerce")
        if parsed.isna().any():
            raise DataValidationError(f"{source}: invalid event timestamp(s)")
        return parsed.dt.tz_convert(self._timezone).dt.tz_localize(None)
