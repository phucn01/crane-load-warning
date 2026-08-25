"""In-memory video processing job state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from app.schemas.detection import RiskLevelValue


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    """Persisted views for one independently assessed risk frame."""

    frame_number: int
    timestamp_seconds: float
    risk_level: RiskLevelValue
    original_path: Path
    rgb_path: Path
    pseudo_bev_path: Path


@dataclass(frozen=True, slots=True)
class FrameRiskResult:
    """Public, lightweight risk classification for one processed video frame."""

    frame_number: int
    timestamp_seconds: float
    risk_level: RiskLevelValue


@dataclass(frozen=True, slots=True)
class RiskSegment:
    """A padded clip around contiguous non-safe frame classifications."""

    segment_id: str
    output_path: Path
    start_frame: int
    end_frame: int
    risk_start_frame: int
    risk_end_frame: int
    start_seconds: float
    end_seconds: float
    max_risk_level: RiskLevelValue
    warning_frame_count: int
    danger_frame_count: int
    frame_evidence: tuple[FrameEvidence, ...] = ()
    output_codec: str = "mp4v"
    browser_playback_compatible: bool = False
    playback_warning: str | None = None


@dataclass(frozen=True, slots=True)
class VideoJob:
    job_id: str
    status: JobStatus
    input_path: Path
    output_path: Path
    current_frame: int = 0
    total_frames: int = 0
    progress: float = 0.0
    processing_fps: float = 0.0
    elapsed_seconds: float = 0.0
    current_risk_level: RiskLevelValue | None = None
    max_risk_level: RiskLevelValue | None = None
    safe_frame_count: int = 0
    warning_frame_count: int = 0
    danger_frame_count: int = 0
    risk_segments: tuple[RiskSegment, ...] = ()
    output_codec: str = "mp4v"
    browser_playback_compatible: bool = False
    playback_warning: str | None = None
    error: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.min.replace(tzinfo=UTC)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def report_path(self) -> Path:
        return self.output_path.with_suffix(".report.json")

    def with_changes(self, **changes: object) -> VideoJob:
        return replace(self, **changes)
