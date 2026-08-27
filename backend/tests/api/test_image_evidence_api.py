from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.api.v1.jobs import get_image_evidence
from app.models import JobStatus, ProcessingJobRecord, RiskSnapshotRecord


def test_safe_no_load_image_evidence_uses_persisted_assessment() -> None:
    job_id = uuid4().hex
    job = ProcessingJobRecord(
        job_id=job_id,
        media_type="image",
        input_name="crane.png",
        input_path=Path("storage/uploads/images/crane.png"),
        output_path=Path("evidence/image/rgb.png"),
        status=JobStatus.COMPLETED,
        total_frames=None,
        processed_frames=None,
        safe_frame_count=1,
        warning_frame_count=0,
        danger_frame_count=0,
        max_risk_level="SAFE",
        processing_time_ms=100.0,
        average_processing_fps=None,
        error_message=None,
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=datetime.now(UTC),
    )
    assessment = RiskSnapshotRecord(
        snapshot_id=uuid4().hex,
        job_id=job_id,
        frame_index=None,
        timestamp_sec=None,
        risk_level="SAFE",
        assessment_status="SAFE_NO_LOAD",
        confidence=0.9,
        assessment_reliable=True,
        quality_reasons=("no_hanging_object_detected",),
        evidence_path="/uploads/images/crane.png",
        rgb_evidence_path="/evidence/image/rgb.png",
        pseudo_bev_path=None,
        created_at=datetime.now(UTC),
    )
    history = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda _: job),
        snapshots=SimpleNamespace(list=lambda **_: ((assessment,), 1)),
    )

    payload = get_image_evidence(job_id, history)  # type: ignore[arg-type]

    assert payload["risk_level"] == "SAFE"
    assert payload["assessment_status"] == "SAFE_NO_LOAD"
    assert payload["assessment_reliable"] is True
    assert payload["quality_reasons"] == ["no_hanging_object_detected"]
    assert payload["detection_url"] == "/evidence/image/rgb.png"
    assert payload["bev_url"] is None
