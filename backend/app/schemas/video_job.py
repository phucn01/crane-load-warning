"""Public contracts for asynchronous video processing."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.detection import RiskLevelValue, StrictSchema

JobStatusValue = Literal["queued", "processing", "completed", "failed"]


class VideoJobCreatedResponse(StrictSchema):
    job_id: str
    status: Literal["queued"] = "queued"
    status_url: str
    stream_url: str
    result_url: str


class VideoSummary(StrictSchema):
    processed_frames: int = Field(ge=0)
    safe_frames: int = Field(ge=0)
    warning_frames: int = Field(ge=0)
    danger_frames: int = Field(ge=0)
    max_risk_level: RiskLevelValue | None
    average_processing_fps: float = Field(ge=0.0)
    risk_segment_count: int = Field(ge=0)


class FrameEvidenceResponse(StrictSchema):
    frame_number: int = Field(ge=1)
    timestamp_seconds: float = Field(ge=0.0)
    risk_level: Literal["WARNING", "DANGER"]
    original_url: str
    rgb_url: str
    pseudo_bev_url: str


class RiskSegmentResponse(StrictSchema):
    segment_id: str
    start_frame: int = Field(ge=1)
    end_frame: int = Field(ge=1)
    risk_start_frame: int = Field(ge=1)
    risk_end_frame: int = Field(ge=1)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    max_risk_level: Literal["WARNING", "DANGER"]
    warning_frame_count: int = Field(ge=0)
    danger_frame_count: int = Field(ge=0)
    frame_evidence: list[FrameEvidenceResponse]
    result_url: str
    output_codec: str
    browser_playback_compatible: bool
    playback_warning: str | None


class VideoJobResponse(StrictSchema):
    job_id: str
    status: JobStatusValue
    input_path: str
    output_path: str
    current_frame: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    progress: float = Field(ge=0.0, le=100.0)
    processing_fps: float = Field(ge=0.0)
    elapsed_seconds: float = Field(ge=0.0)
    current_risk_level: RiskLevelValue | None
    max_risk_level: RiskLevelValue | None
    safe_frame_count: int = Field(ge=0)
    warning_frame_count: int = Field(ge=0)
    danger_frame_count: int = Field(ge=0)
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    stream_url: str
    result_url: str | None
    download_url: str | None
    report_url: str | None
    summary: VideoSummary | None
    risk_segments: list[RiskSegmentResponse]
    output_codec: str
    browser_playback_compatible: bool
    playback_warning: str | None


__all__ = [
    "FrameEvidenceResponse",
    "JobStatusValue",
    "RiskSegmentResponse",
    "VideoJobCreatedResponse",
    "VideoJobResponse",
    "VideoSummary",
]
