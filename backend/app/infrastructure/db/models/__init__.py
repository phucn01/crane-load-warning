"""Persistence-only SQLAlchemy models."""

from .processing_history import ProcessingJobRow, RiskSnapshotRow

__all__ = ["ProcessingJobRow", "RiskSnapshotRow"]
