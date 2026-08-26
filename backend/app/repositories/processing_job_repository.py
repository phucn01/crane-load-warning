"""Repository abstractions for durable image/video processing history."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from sqlalchemy import func, select

from app.infrastructure.db.models import ProcessingJobRow
from app.infrastructure.db.session import DatabaseSessionFactory
from app.models import JobStatus, ProcessingJobRecord


class ProcessingJobRepository(Protocol):
    def create(self, record: ProcessingJobRecord) -> ProcessingJobRecord: ...

    def get(self, job_id: str) -> ProcessingJobRecord | None: ...

    def update(self, job_id: str, **changes: object) -> ProcessingJobRecord: ...

    def list(
        self,
        *,
        status: JobStatus | None = None,
        media_type: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[tuple[ProcessingJobRecord, ...], int]: ...


class InMemoryProcessingJobRepository:
    def __init__(self) -> None:
        self._records: dict[str, ProcessingJobRecord] = {}
        self._lock = RLock()

    def create(self, record: ProcessingJobRecord) -> ProcessingJobRecord:
        with self._lock:
            if record.job_id in self._records:
                raise ValueError(f"processing job already exists: {record.job_id}")
            self._records[record.job_id] = record
            return record

    def get(self, job_id: str) -> ProcessingJobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def update(self, job_id: str, **changes: object) -> ProcessingJobRecord:
        with self._lock:
            current = self._records.get(job_id)
            if current is None:
                raise KeyError(job_id)
            values = {field: getattr(current, field) for field in current.__dataclass_fields__}
            values.update(changes)
            updated = ProcessingJobRecord(**values)
            self._records[job_id] = updated
            return updated

    def list(
        self,
        *,
        status: JobStatus | None = None,
        media_type: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[tuple[ProcessingJobRecord, ...], int]:
        with self._lock:
            records = [
                item
                for item in self._records.values()
                if (status is None or item.status is status)
                and (media_type is None or item.media_type == media_type)
                and (from_time is None or item.created_at >= from_time)
                and (to_time is None or item.created_at <= to_time)
            ]
            records.sort(key=lambda item: item.created_at, reverse=True)
            return tuple(records[offset : offset + limit]), len(records)


class SqlAlchemyProcessingJobRepository:
    def __init__(self, sessions: DatabaseSessionFactory) -> None:
        self._sessions = sessions

    def create(self, record: ProcessingJobRecord) -> ProcessingJobRecord:
        with self._sessions() as session:
            try:
                session.add(_record_to_row(record))
                session.commit()
            except Exception:
                session.rollback()
                raise
        return record

    def get(self, job_id: str) -> ProcessingJobRecord | None:
        identifier = _optional_uuid(job_id)
        if identifier is None:
            return None
        with self._sessions() as session:
            row = session.get(ProcessingJobRow, identifier)
            return None if row is None else _row_to_record(row)

    def update(self, job_id: str, **changes: object) -> ProcessingJobRecord:
        with self._sessions() as session:
            try:
                row = session.get(ProcessingJobRow, _uuid(job_id))
                if row is None:
                    raise KeyError(job_id)
                _apply_changes(row, changes)
                session.commit()
                session.refresh(row)
                return _row_to_record(row)
            except Exception:
                session.rollback()
                raise

    def list(
        self,
        *,
        status: JobStatus | None = None,
        media_type: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[tuple[ProcessingJobRecord, ...], int]:
        filters = []
        if status is not None:
            filters.append(ProcessingJobRow.status == status.value)
        if media_type is not None:
            filters.append(ProcessingJobRow.media_type == media_type)
        if from_time is not None:
            filters.append(ProcessingJobRow.created_at >= from_time)
        if to_time is not None:
            filters.append(ProcessingJobRow.created_at <= to_time)
        with self._sessions() as session:
            total = session.scalar(
                select(func.count()).select_from(ProcessingJobRow).where(*filters)
            )
            rows = session.scalars(
                select(ProcessingJobRow)
                .where(*filters)
                .order_by(ProcessingJobRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return tuple(_row_to_record(row) for row in rows), int(total or 0)


def _record_to_row(record: ProcessingJobRecord) -> ProcessingJobRow:
    return ProcessingJobRow(
        id=_uuid(record.job_id),
        media_type=record.media_type,
        input_name=record.input_name,
        input_path=_path_string(record.input_path),
        output_path=_path_string(record.output_path),
        status=record.status.value,
        total_frames=record.total_frames,
        processed_frames=record.processed_frames,
        safe_frame_count=record.safe_frame_count,
        warning_frame_count=record.warning_frame_count,
        danger_frame_count=record.danger_frame_count,
        max_risk_level=record.max_risk_level,
        processing_time_ms=record.processing_time_ms,
        average_processing_fps=record.average_processing_fps,
        error_message=record.error_message,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def _apply_changes(row: ProcessingJobRow, changes: Mapping[str, object]) -> None:
    for name, value in changes.items():
        if name == "job_id" or not hasattr(row, name):
            raise ValueError(f"unsupported processing job field: {name}")
        if isinstance(value, JobStatus):
            value = value.value
        elif isinstance(value, Path):
            value = str(value)
        setattr(row, name, value)


def _row_to_record(row: ProcessingJobRow) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id=row.id.hex,
        media_type=row.media_type,  # type: ignore[arg-type]
        input_name=row.input_name,
        input_path=None if row.input_path is None else Path(row.input_path),
        output_path=None if row.output_path is None else Path(row.output_path),
        status=JobStatus(row.status),
        total_frames=row.total_frames,
        processed_frames=row.processed_frames,
        safe_frame_count=row.safe_frame_count,
        warning_frame_count=row.warning_frame_count,
        danger_frame_count=row.danger_frame_count,
        max_risk_level=row.max_risk_level,  # type: ignore[arg-type]
        processing_time_ms=row.processing_time_ms,
        average_processing_fps=row.average_processing_fps,
        error_message=row.error_message,
        created_at=_utc(row.created_at),
        started_at=_optional_utc(row.started_at),
        completed_at=_optional_utc(row.completed_at),
    )


def _uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)


def _optional_uuid(value: str) -> uuid.UUID | None:
    try:
        return _uuid(value)
    except ValueError:
        return None


def _path_string(value: Path | None) -> str | None:
    return None if value is None else str(value)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _utc(value)


__all__ = [
    "InMemoryProcessingJobRepository",
    "ProcessingJobRepository",
    "SqlAlchemyProcessingJobRepository",
]
