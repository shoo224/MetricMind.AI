"""GroundTruthLabeler: converts labelled fault INTERVALS into per-row labels."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from metricmind.domain.entities.event import LabeledEvent

logger = logging.getLogger(__name__)

NORMAL_CATEGORY = "none"


class GroundTruthLabeler:
    """Adds three columns to a time-indexed frame:

    ``is_anomaly`` (1 if the row's timestamp lies inside any event, bounds inclusive),
    ``anomaly_category`` (cpu/memory/network/none) and ``event_id`` ('' if none).

    LIMITATION (important for the viva): labels are interval-level ground truth of INJECTED faults.
    A row inside a window is labelled anomalous even if the metric momentarily looks normal, and
    natural spikes outside any window are labelled normal. No label is invented.
    """

    def label(self, frame: pd.DataFrame, events: list[LabeledEvent]) -> pd.DataFrame:
        out = frame.copy()
        index = out.index
        is_anomaly = np.zeros(len(out), dtype=np.int8)
        category = np.full(len(out), NORMAL_CATEGORY, dtype=object)
        event_ids = np.full(len(out), "", dtype=object)
        overlaps = 0

        for event in events:
            lo = index.searchsorted(pd.Timestamp(event.start), side="left")
            hi = index.searchsorted(pd.Timestamp(event.end), side="right")
            if hi <= lo:
                logger.warning("Event %s covers no rows in the dataset", event.event_id)
                continue
            overlaps += int(is_anomaly[lo:hi].sum())
            is_anomaly[lo:hi] = 1
            category[lo:hi] = event.category
            event_ids[lo:hi] = event.event_id

        if overlaps:
            logger.warning("%d rows belong to overlapping events (later event wins)", overlaps)
        out["is_anomaly"] = is_anomaly
        out["anomaly_category"] = category
        out["event_id"] = event_ids
        logger.info(
            "Labelled %d/%d rows (%.2f%%) as anomalous from %d events",
            int(is_anomaly.sum()),
            len(out),
            100 * is_anomaly.mean(),
            len(events),
        )
        return out
