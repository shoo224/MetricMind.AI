from datetime import datetime

from metricmind.infrastructure.data.csv_event_loader import CSVEventLabelLoader

SPEC = {"files": ["ev.csv"], "timezone": "UTC"}
HEADER = "id,flag,start_time,end_time,start_point,end_point,type,mode,intensity\n"


def test_utc_iso_timestamps_become_naive_and_sorted(tmp_path):
    (tmp_path / "ev.csv").write_text(
        HEADER
        + "b,completed,2025-01-01T01:00:00.500Z,2025-01-01T01:10:00.000Z,1,2,memory,burst,5\n"
        "a,completed,2025-01-01T00:00:00.000Z,2025-01-01T00:05:00.000Z,1,2,cpu,stable,9\n"
    )
    events = CSVEventLabelLoader(tmp_path, SPEC).load_events()
    assert [e.event_id for e in events] == ["a", "b"]
    assert (
        events[1].start == datetime(2025, 1, 1, 1, 0, 0, 500000) and events[1].start.tzinfo is None
    )
    assert events[0].category == "cpu" and events[0].intensity == 9.0
