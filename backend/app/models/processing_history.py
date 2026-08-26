"""Domain records for durable processing history and sampled risk evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from app.schemas.detection import RiskLevelValue

from .video_job import JobStatus

MediaTypeValue = Literal["image", "video"]


@dataclass(frozen=True, slots=True)
class ProcessingJobRecord:
    job_id: str
    media_type: MediaTypeValue
    input_name: str
    input_path: Path | None
    output_path: Path | None
    status: JobStatus
    total_frames: int | None
    processed_frames: int | None
    safe_frame_count: int
    warning_frame_count: int
    danger_frame_count: int
    max_risk_level: RiskLevelValue | None
    processing_time_ms: float | None
    average_processing_fps: float | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class RiskSnapshotRecord:
    snapshot_id: str
    job_id: str
    frame_index: int | None
    timestamp_sec: float | None
    risk_level: RiskLevelValue | None
    confidence: float | None
    assessment_reliable: bool | None
    quality_reasons: tuple[str, ...]
    evidence_path: str | None
    rgb_evidence_path: str | None
    pseudo_bev_path: str | None
    created_at: datetime
    assessment_status: str = "FULL_EVALUATION"


__all__ = [
    "MediaTypeValue",
    "ProcessingJobRecord",
    "RiskSnapshotRecord",
]
