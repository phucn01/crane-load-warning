"""Persistence-only SQLAlchemy models."""

from .processing_history import FrameAssessmentRow, ProcessingJobRow, RiskSnapshotRow

__all__ = ["FrameAssessmentRow", "ProcessingJobRow", "RiskSnapshotRow"]
