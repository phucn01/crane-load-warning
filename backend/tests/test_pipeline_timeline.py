import json
import logging
from pathlib import Path

import pytest
from pipeline_timeline import (
    PipelineTimeline,
    TimelineStatus,
    log_pipeline_operation,
)


def test_pipeline_log_uses_visible_start_and_end_markers(caplog):
    with caplog.at_level(logging.INFO), log_pipeline_operation(
        "geometry",
        "depth_normalization",
        frame_id="frame-1",
        entity_id="load-1",
    ):
        pass

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == (
        "=== START ===\n"
        "    COMPONENT    : GEOMETRY\n"
        "    OPERATION    : DEPTH_NORMALIZATION\n"
        "    FRAME_ID     : frame-1\n"
        "    ENTITY_ID    : load-1"
    )
    assert messages[1].startswith(
        "=== END ===\n"
        "    COMPONENT    : GEOMETRY\n"
        "    OPERATION    : DEPTH_NORMALIZATION\n"
        "    FRAME_ID     : frame-1\n"
        "    ENTITY_ID    : load-1\n"
        "    DURATION_MS  : "
    )


def test_tracks_running_and_completed_operation():
    timeline = PipelineTimeline()

    with timeline.track("vision", "process", frame_id="frame-000001"):
        running = timeline.snapshot()[0]
        assert running.status is TimelineStatus.RUNNING
        assert running.started_at.endswith("+07:00")
        assert running.completed_at is None
        assert running.duration_ms is None

    completed = timeline.snapshot()[0]
    assert completed.record_id == 1
    assert completed.component == "vision"
    assert completed.operation == "process"
    assert completed.frame_id == "frame-000001"
    assert completed.status is TimelineStatus.COMPLETED
    assert completed.completed_at is not None
    assert completed.completed_at.endswith("+07:00")
    assert completed.duration_ms is not None and completed.duration_ms >= 0.0
    assert completed.error_type is None


def test_tracks_failure_and_preserves_original_exception():
    timeline = PipelineTimeline()

    with (
        pytest.raises(ValueError, match="invalid frame"),
        timeline.track("geometry", "process", frame_id="frame-bad"),
    ):
        raise ValueError("invalid frame")

    failed = timeline.snapshot()[0]
    assert failed.status is TimelineStatus.FAILED
    assert failed.error_type == "ValueError"
    assert failed.completed_at is not None
    assert failed.duration_ms is not None


def test_filters_frames_and_writes_json(tmp_path: Path):
    timeline = PipelineTimeline()
    with timeline.track("vision", "process", frame_id="frame-1"):
        pass
    with timeline.track("risk", "process", frame_id="frame-2"):
        pass

    assert [record.frame_id for record in timeline.snapshot(frame_id="frame-2")] == [
        "frame-2"
    ]
    output_path = timeline.write_json(tmp_path / "timeline.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["timezone"] == "Asia/Bangkok"
    assert [record["status"] for record in payload["records"]] == [
        "COMPLETED",
        "COMPLETED",
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        timeline.write_json(output_path)


def test_clear_restarts_record_ids():
    timeline = PipelineTimeline()
    with timeline.track("vision", "process", frame_id="frame-1"):
        pass
    timeline.clear()
    with timeline.track("risk", "process", frame_id="frame-2"):
        pass

    assert timeline.snapshot()[0].record_id == 1
