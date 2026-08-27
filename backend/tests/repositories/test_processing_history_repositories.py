from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.infrastructure.db import Base
from app.models import JobStatus, ProcessingJobRecord, RiskSnapshotRecord
from app.repositories import (
    SqlAlchemyProcessingJobRepository,
    SqlAlchemyRiskSnapshotRepository,
)
from app.services.processing_history_service import (
    ProcessingHistoryService,
    RiskSnapshotPolicy,
)


@pytest.fixture
def repositories(tmp_path: Path):  # type: ignore[no-untyped-def]
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'history.db'}")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    yield (
        SqlAlchemyProcessingJobRepository(sessions),
        SqlAlchemyRiskSnapshotRepository(sessions),
    )
    engine.dispose()


def _job(
    *,
    media_type: str = "video",
    status: JobStatus = JobStatus.QUEUED,
    created_at: datetime | None = None,
) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id=uuid4().hex,
        media_type=media_type,  # type: ignore[arg-type]
        input_name="crane.mp4" if media_type == "video" else "crane.png",
        input_path=(
            Path("storage/input")
            if media_type == "video"
            else Path("storage/uploads/images/crane.png")
        ),
        output_path=Path("storage/output") if media_type == "video" else None,
        status=status,
        total_frames=None,
        processed_frames=0 if media_type == "video" else None,
        safe_frame_count=0,
        warning_frame_count=0,
        danger_frame_count=0,
        max_risk_level=None,
        processing_time_ms=None,
        average_processing_fps=None,
        error_message=None,
        created_at=created_at or datetime.now(UTC),
        started_at=None,
        completed_at=None,
    )


def test_create_update_complete_and_serialize_job(repositories) -> None:  # type: ignore[no-untyped-def]
    jobs, _ = repositories
    created = jobs.create(_job())
    jobs.update(
        created.job_id,
        status=JobStatus.PROCESSING,
        total_frames=120,
        processed_frames=50,
    )
    completed_at = datetime.now(UTC)
    jobs.update(
        created.job_id,
        status=JobStatus.COMPLETED,
        processed_frames=120,
        safe_frame_count=100,
        warning_frame_count=15,
        danger_frame_count=5,
        max_risk_level="DANGER",
        processing_time_ms=4200.0,
        average_processing_fps=28.5,
        completed_at=completed_at,
    )

    restored = jobs.get(created.job_id)

    assert restored is not None
    assert restored.status is JobStatus.COMPLETED
    assert restored.input_path == Path("storage/input")
    assert restored.processed_frames == 120
    assert restored.danger_frame_count == 5
    assert restored.completed_at == completed_at


def test_failed_job_and_history_filters(repositories) -> None:  # type: ignore[no-untyped-def]
    jobs, _ = repositories
    old = jobs.create(_job(created_at=datetime.now(UTC) - timedelta(days=2)))
    image = jobs.create(_job(media_type="image"))
    jobs.update(
        image.job_id,
        status=JobStatus.FAILED,
        error_message="image processing failed",
        completed_at=datetime.now(UTC),
    )

    items, total = jobs.list(status=JobStatus.FAILED, media_type="image")
    recent, recent_total = jobs.list(from_time=datetime.now(UTC) - timedelta(days=1))

    assert total == 1
    assert items[0].job_id == image.job_id
    assert items[0].error_message == "image processing failed"
    assert recent_total == 1
    assert recent[0].job_id != old.job_id


def test_create_snapshot_filters_and_foreign_key(repositories) -> None:  # type: ignore[no-untyped-def]
    jobs, snapshots = repositories
    job = jobs.create(_job())
    snapshot = RiskSnapshotRecord(
        snapshot_id=uuid4().hex,
        job_id=job.job_id,
        frame_index=24,
        timestamp_sec=0.8,
        risk_level="DANGER",
        confidence=0.91,
        assessment_reliable=True,
        quality_reasons=("fixture",),
        evidence_path="/evidence/original.png",
        rgb_evidence_path="/evidence/rgb.png",
        pseudo_bev_path="/evidence/bev.png",
        created_at=datetime.now(UTC),
    )
    snapshots.create(snapshot)

    restored = snapshots.get(snapshot.snapshot_id)
    items, total = snapshots.list(job_id=job.job_id, risk_level="DANGER")

    assert restored == snapshot
    assert total == 1
    assert items == (snapshot,)

    invalid = replace(
        snapshot,
        snapshot_id=uuid4().hex,
        job_id=uuid4().hex,
    )
    with pytest.raises(IntegrityError):
        snapshots.create(invalid)


def test_snapshot_cooldown_and_escalation() -> None:
    policy = RiskSnapshotPolicy(min_interval_seconds=2.0)

    assert policy.should_capture("job", timestamp_sec=0.0, risk_level="WARNING")
    assert not policy.should_capture("job", timestamp_sec=1.0, risk_level="WARNING")
    assert policy.should_capture("job", timestamp_sec=1.1, risk_level="DANGER")
    assert not policy.should_capture("job", timestamp_sec=2.5, risk_level="DANGER")
    assert policy.should_capture("job", timestamp_sec=3.1, risk_level="DANGER")
    assert not policy.should_capture("other", timestamp_sec=0.0, risk_level="SAFE")


def test_safe_no_load_image_is_persisted_as_frame_assessment(repositories) -> None:  # type: ignore[no-untyped-def]
    jobs, snapshots = repositories
    job = jobs.create(_job(media_type="image"))
    service = ProcessingHistoryService(jobs, snapshots)
    response = SimpleNamespace(
        assessment_status="SAFE_NO_LOAD",
        assessment=SimpleNamespace(
            risk_level="SAFE",
            assessment_reliable=True,
            quality_reasons=["safe_no_load"],
            pairs=[],
        ),
        evidence=SimpleNamespace(
            rgb_url="/evidence/image/rgb.png",
            pseudo_bev_url=None,
        ),
    )

    persisted = service.persist_image_snapshot(
        job.job_id,
        response,  # type: ignore[arg-type]
        original_evidence_path="/uploads/images/crane.png",
    )
    records, total = snapshots.list(job_id=job.job_id)

    assert persisted is True
    assert total == 1
    assert records[0].risk_level == "SAFE"
    assert records[0].assessment_status == "SAFE_NO_LOAD"
    assert records[0].assessment_reliable is True
    assert records[0].quality_reasons == ("safe_no_load",)
    assert records[0].pseudo_bev_path is None


def test_persistence_failure_is_best_effort() -> None:
    class FailingJobs:
        def create(self, record: ProcessingJobRecord) -> ProcessingJobRecord:
            del record
            raise ConnectionError("database unavailable")

    service = ProcessingHistoryService(FailingJobs(), object())  # type: ignore[arg-type]

    assert service.create_job(_job()) is False
