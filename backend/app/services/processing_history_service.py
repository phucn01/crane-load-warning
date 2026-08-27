"""Best-effort persistence orchestration for processing history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import yaml

from app.models import (
    JobStatus,
    ProcessingJobRecord,
    RiskSnapshotRecord,
    VideoJob,
)
from app.repositories import ProcessingJobRepository, RiskSnapshotRepository
from app.schemas.detection import ImageDetectionResponse, RiskLevelValue

LOGGER = logging.getLogger(__name__)
RISK_SEVERITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2}


class RiskSnapshotPolicy:
    """Time-based sampling with immediate capture on risk escalation."""

    def __init__(self, min_interval_seconds: float = 2.0) -> None:
        if min_interval_seconds < 0:
            raise ValueError("snapshot minimum interval must be non-negative")
        self.min_interval_seconds = min_interval_seconds
        self._last: dict[str, tuple[float, RiskLevelValue]] = {}
        self._lock = RLock()

    @classmethod
    def from_yaml(cls, path: Path | None) -> RiskSnapshotPolicy:
        if path is None or not path.is_file():
            return cls()
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        persistence = payload.get("persistence", {})
        return cls(float(persistence.get("snapshot_min_interval_seconds", 2.0)))

    def should_capture(
        self,
        job_id: str,
        *,
        timestamp_sec: float,
        risk_level: RiskLevelValue,
    ) -> bool:
        if risk_level == "SAFE":
            return False
        with self._lock:
            previous = self._last.get(job_id)
            capture = (
                previous is None
                or RISK_SEVERITY[risk_level] > RISK_SEVERITY[previous[1]]
                or timestamp_sec - previous[0] >= self.min_interval_seconds
            )
            if capture:
                self._last[job_id] = (timestamp_sec, risk_level)
            return capture

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._last.pop(job_id, None)


class ProcessingHistoryService:
    """Keep persistence failures isolated from inference/runtime state."""

    def __init__(
        self,
        jobs: ProcessingJobRepository,
        snapshots: RiskSnapshotRepository,
        *,
        snapshot_policy: RiskSnapshotPolicy | None = None,
        job_update_interval_frames: int = 10,
        close_callback: Any | None = None,
    ) -> None:
        if job_update_interval_frames <= 0:
            raise ValueError("job update interval must be positive")
        self.jobs = jobs
        self.snapshots = snapshots
        self.snapshot_policy = snapshot_policy or RiskSnapshotPolicy()
        self.job_update_interval_frames = job_update_interval_frames
        self._close_callback = close_callback

    def close(self) -> None:
        if self._close_callback is not None:
            self._close_callback()

    def create_job(self, record: ProcessingJobRecord) -> bool:
        return self._write("create_processing_job", self.jobs.create, record)

    def mark_processing(self, job_id: str, *, total_frames: int | None = None) -> bool:
        changes: dict[str, object] = {
            "status": JobStatus.PROCESSING,
            "started_at": datetime.now(UTC),
        }
        if total_frames is not None:
            changes["total_frames"] = total_frames
        return self._write("mark_processing_job", self.jobs.update, job_id, **changes)

    def persist_video_progress(self, job: VideoJob) -> bool:
        if job.current_frame % self.job_update_interval_frames != 0:
            return False
        return self._write(
            "persist_video_progress",
            self.jobs.update,
            job.job_id,
            total_frames=job.total_frames,
            processed_frames=job.current_frame,
            safe_frame_count=job.safe_frame_count,
            warning_frame_count=job.warning_frame_count,
            danger_frame_count=job.danger_frame_count,
            max_risk_level=job.max_risk_level,
            processing_time_ms=job.elapsed_seconds * 1000.0,
            average_processing_fps=job.processing_fps,
        )

    def complete_video(self, job: VideoJob) -> bool:
        self.snapshot_policy.clear(job.job_id)
        return self._write(
            "complete_video_job",
            self.jobs.update,
            job.job_id,
            status=JobStatus.COMPLETED,
            output_path=job.output_path,
            total_frames=job.total_frames,
            processed_frames=job.current_frame,
            safe_frame_count=job.safe_frame_count,
            warning_frame_count=job.warning_frame_count,
            danger_frame_count=job.danger_frame_count,
            max_risk_level=job.max_risk_level,
            processing_time_ms=job.elapsed_seconds * 1000.0,
            average_processing_fps=job.processing_fps,
            completed_at=job.completed_at,
        )

    def complete_image(self, job_id: str, response: ImageDetectionResponse) -> bool:
        level = response.assessment.risk_level
        counts = {
            "SAFE": int(level == "SAFE"),
            "WARNING": int(level == "WARNING"),
            "DANGER": int(level == "DANGER"),
        }
        return self._write(
            "complete_image_job",
            self.jobs.update,
            job_id,
            status=JobStatus.COMPLETED,
            output_path=_optional_path(response.evidence.combined_url),
            safe_frame_count=counts["SAFE"],
            warning_frame_count=counts["WARNING"],
            danger_frame_count=counts["DANGER"],
            max_risk_level=level,
            processing_time_ms=response.processing_time_ms,
            completed_at=datetime.now(UTC),
        )

    def fail_job(self, job_id: str, error_message: str) -> bool:
        self.snapshot_policy.clear(job_id)
        return self._write(
            "fail_processing_job",
            self.jobs.update,
            job_id,
            status=JobStatus.FAILED,
            error_message=error_message,
            completed_at=datetime.now(UTC),
        )

    def should_capture_snapshot(
        self,
        job_id: str,
        *,
        timestamp_sec: float,
        risk_level: RiskLevelValue,
    ) -> bool:
        return self.snapshot_policy.should_capture(
            job_id,
            timestamp_sec=timestamp_sec,
            risk_level=risk_level,
        )

    def persist_snapshot(self, record: RiskSnapshotRecord) -> bool:
        return self._write("persist_risk_snapshot", self.snapshots.create, record)

    def persist_snapshots(self, records: list[RiskSnapshotRecord]) -> bool:
        if not records:
            return True
        return self._write(
            "persist_risk_snapshot_batch",
            self.snapshots.create_many,
            tuple(records),
        )

    def persist_image_snapshot(
        self,
        job_id: str,
        response: ImageDetectionResponse,
        *,
        original_evidence_path: str | None = None,
    ) -> bool:
        level = response.assessment.risk_level
        confidence = max(
            (pair.confidence for pair in response.assessment.pairs),
            default=None,
        )
        return self.persist_snapshot(
            RiskSnapshotRecord(
                snapshot_id=uuid4().hex,
                job_id=job_id,
                frame_index=None,
                timestamp_sec=None,
                risk_level=level,  # type: ignore[arg-type]
                assessment_status=response.assessment_status,
                confidence=confidence,
                assessment_reliable=response.assessment.assessment_reliable,
                quality_reasons=tuple(response.assessment.quality_reasons),
                evidence_path=original_evidence_path,
                rgb_evidence_path=response.evidence.rgb_url,
                pseudo_bev_path=response.evidence.pseudo_bev_url,
                created_at=datetime.now(UTC),
            )
        )

    def _write(self, operation: str, function: Any, *args: Any, **kwargs: Any) -> bool:
        try:
            function(*args, **kwargs)
            return True
        except Exception as error:
            LOGGER.exception(
                "=== ERROR | OPERATION=%s | ERROR_TYPE=%s ===",
                operation.upper(),
                type(error).__name__,
            )
            return False


def _optional_path(value: str | None) -> Path | None:
    return None if value is None else Path(value)


__all__ = ["ProcessingHistoryService", "RiskSnapshotPolicy"]
