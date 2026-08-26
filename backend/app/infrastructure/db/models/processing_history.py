"""SQLAlchemy mappings for processing history."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base


class ProcessingJobRow(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint("media_type in ('image', 'video')", name="ck_job_media_type"),
        CheckConstraint(
            "status in ('queued', 'processing', 'completed', 'failed')",
            name="ck_job_status",
        ),
        CheckConstraint(
            "max_risk_level is null or max_risk_level in ('SAFE', 'WARNING', 'DANGER')",
            name="ck_job_max_risk_level",
        ),
        Index("ix_processing_jobs_created_at", "created_at"),
        Index("ix_processing_jobs_status", "status"),
        Index("ix_processing_jobs_media_type", "media_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)
    input_name: Mapped[str] = mapped_column(Text, nullable=False)
    input_path: Mapped[str | None] = mapped_column(Text)
    output_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    total_frames: Mapped[int | None] = mapped_column(Integer)
    processed_frames: Mapped[int | None] = mapped_column(Integer)
    safe_frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    danger_frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_risk_level: Mapped[str | None] = mapped_column(String(16))
    processing_time_ms: Mapped[float | None] = mapped_column(Float)
    average_processing_fps: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snapshots: Mapped[list[RiskSnapshotRow]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class RiskSnapshotRow(Base):
    __tablename__ = "risk_snapshots"
    __table_args__ = (
        CheckConstraint(
            "risk_level in ('WARNING', 'DANGER')",
            name="ck_snapshot_risk_level",
        ),
        Index("ix_risk_snapshots_job_id", "job_id"),
        Index("ix_risk_snapshots_created_at", "created_at"),
        Index("ix_risk_snapshots_risk_level", "risk_level"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_index: Mapped[int | None] = mapped_column(Integer)
    timestamp_sec: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    assessment_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quality_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    evidence_path: Mapped[str | None] = mapped_column(Text)
    rgb_evidence_path: Mapped[str | None] = mapped_column(Text)
    pseudo_bev_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    job: Mapped[ProcessingJobRow] = relationship(back_populates="snapshots")


__all__ = ["ProcessingJobRow", "RiskSnapshotRow"]
