"""Public contracts for persisted processing history and risk snapshots."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .detection import RiskLevelValue, StrictSchema
from .video_job import JobStatusValue


class ProcessingJobHistoryResponse(StrictSchema):
    id: str
    media_type: str
    input_name: str
    input_path: str | None
    output_path: str | None
    status: JobStatusValue
    total_frames: int | None
    processed_frames: int | None
    safe_frame_count: int = Field(ge=0)
    warning_frame_count: int = Field(ge=0)
    danger_frame_count: int = Field(ge=0)
    max_risk_level: RiskLevelValue | None
    processing_time_ms: float | None
    average_processing_fps: float | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ProcessingJobHistoryPage(StrictSchema):
    items: list[ProcessingJobHistoryResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class RiskSnapshotResponse(StrictSchema):
    id: str
    job_id: str
    frame_index: int | None
    timestamp_sec: float | None
    risk_level: RiskLevelValue
    confidence: float | None
    assessment_reliable: bool
    quality_reasons: list[str]
    evidence_path: str | None
    rgb_evidence_path: str | None
    pseudo_bev_path: str | None
    created_at: datetime


class RiskSnapshotPage(StrictSchema):
    items: list[RiskSnapshotResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


__all__ = [
    "ProcessingJobHistoryPage",
    "ProcessingJobHistoryResponse",
    "RiskSnapshotPage",
    "RiskSnapshotResponse",
]
