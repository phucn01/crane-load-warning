"""Repository abstractions for sampled WARNING/DANGER snapshots."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from threading import RLock
from typing import Literal, Protocol

from sqlalchemy import func, select

from app.infrastructure.db.models import FrameAssessmentRow
from app.infrastructure.db.session import DatabaseSessionFactory
from app.models import RiskSnapshotRecord

from .processing_job_repository import ProcessingJobRepository


class RiskSnapshotRepository(Protocol):
    def create(self, record: RiskSnapshotRecord) -> RiskSnapshotRecord: ...

    def create_many(
        self, records: tuple[RiskSnapshotRecord, ...]
    ) -> tuple[RiskSnapshotRecord, ...]: ...

    def get(self, snapshot_id: str) -> RiskSnapshotRecord | None: ...

    def list(
        self,
        *,
        job_id: str | None = None,
        risk_level: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
        order: Literal["created_desc", "frame_asc"] = "created_desc",
    ) -> tuple[tuple[RiskSnapshotRecord, ...], int]: ...


class InMemoryRiskSnapshotRepository:
    def __init__(self, jobs: ProcessingJobRepository) -> None:
        self._jobs = jobs
        self._records: dict[str, RiskSnapshotRecord] = {}
        self._lock = RLock()

    def create(self, record: RiskSnapshotRecord) -> RiskSnapshotRecord:
        with self._lock:
            if self._jobs.get(record.job_id) is None:
                raise ValueError(f"processing job does not exist: {record.job_id}")
            if record.snapshot_id in self._records:
                raise ValueError(f"risk snapshot already exists: {record.snapshot_id}")
            self._records[record.snapshot_id] = record
            return record

    def create_many(
        self, records: tuple[RiskSnapshotRecord, ...]
    ) -> tuple[RiskSnapshotRecord, ...]:
        with self._lock:
            for record in records:
                if self._jobs.get(record.job_id) is None:
                    raise ValueError(f"processing job does not exist: {record.job_id}")
                if record.snapshot_id in self._records:
                    raise ValueError(f"risk snapshot already exists: {record.snapshot_id}")
            for record in records:
                self._records[record.snapshot_id] = record
            return records

    def get(self, snapshot_id: str) -> RiskSnapshotRecord | None:
        with self._lock:
            return self._records.get(snapshot_id)

    def list(
        self,
        *,
        job_id: str | None = None,
        risk_level: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
        order: Literal["created_desc", "frame_asc"] = "created_desc",
    ) -> tuple[tuple[RiskSnapshotRecord, ...], int]:
        with self._lock:
            records = [
                item
                for item in self._records.values()
                if (job_id is None or item.job_id == job_id)
                and (risk_level is None or item.risk_level == risk_level)
                and (from_time is None or item.created_at >= from_time)
                and (to_time is None or item.created_at <= to_time)
            ]
            if order == "frame_asc":
                records.sort(key=lambda item: (item.frame_index is None, item.frame_index or 0))
            else:
                records.sort(key=lambda item: item.created_at, reverse=True)
            return tuple(records[offset : offset + limit]), len(records)


class SqlAlchemyRiskSnapshotRepository:
    def __init__(self, sessions: DatabaseSessionFactory) -> None:
        self._sessions = sessions

    def create(self, record: RiskSnapshotRecord) -> RiskSnapshotRecord:
        with self._sessions() as session:
            try:
                session.add(
                    FrameAssessmentRow(
                        id=uuid.UUID(record.snapshot_id),
                        job_id=uuid.UUID(record.job_id),
                        frame_index=record.frame_index,
                        timestamp_sec=record.timestamp_sec,
                        risk_level=record.risk_level,
                        assessment_status=record.assessment_status,
                        confidence=record.confidence,
                        assessment_reliable=record.assessment_reliable,
                        quality_reasons=list(record.quality_reasons),
                        evidence_path=record.evidence_path,
                        rgb_evidence_path=record.rgb_evidence_path,
                        pseudo_bev_path=record.pseudo_bev_path,
                        created_at=record.created_at,
                    )
                )
                session.commit()
            except Exception:
                session.rollback()
                raise
        return record

    def create_many(
        self, records: tuple[RiskSnapshotRecord, ...]
    ) -> tuple[RiskSnapshotRecord, ...]:
        if not records:
            return records
        with self._sessions() as session:
            try:
                session.add_all([_row_from_record(record) for record in records])
                session.commit()
            except Exception:
                session.rollback()
                raise
        return records

    def get(self, snapshot_id: str) -> RiskSnapshotRecord | None:
        try:
            identifier = uuid.UUID(snapshot_id)
        except ValueError:
            return None
        with self._sessions() as session:
            row = session.get(FrameAssessmentRow, identifier)
            return None if row is None else _row_to_record(row)

    def list(
        self,
        *,
        job_id: str | None = None,
        risk_level: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
        order: Literal["created_desc", "frame_asc"] = "created_desc",
    ) -> tuple[tuple[RiskSnapshotRecord, ...], int]:
        filters = []
        if job_id is not None:
            try:
                identifier = uuid.UUID(job_id)
            except ValueError:
                return (), 0
            filters.append(FrameAssessmentRow.job_id == identifier)
        if risk_level is not None:
            filters.append(FrameAssessmentRow.risk_level == risk_level)
        if from_time is not None:
            filters.append(FrameAssessmentRow.created_at >= from_time)
        if to_time is not None:
            filters.append(FrameAssessmentRow.created_at <= to_time)
        with self._sessions() as session:
            total = session.scalar(
                select(func.count()).select_from(FrameAssessmentRow).where(*filters)
            )
            ordering = (
                (FrameAssessmentRow.frame_index.asc().nulls_last(), FrameAssessmentRow.created_at.asc())
                if order == "frame_asc"
                else (FrameAssessmentRow.created_at.desc(),)
            )
            rows = session.scalars(
                select(FrameAssessmentRow)
                .where(*filters)
                .order_by(*ordering)
                .limit(limit)
                .offset(offset)
            ).all()
            return tuple(_row_to_record(row) for row in rows), int(total or 0)


def _row_to_record(row: FrameAssessmentRow) -> RiskSnapshotRecord:
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return RiskSnapshotRecord(
        snapshot_id=row.id.hex,
        job_id=row.job_id.hex,
        frame_index=row.frame_index,
        timestamp_sec=row.timestamp_sec,
        risk_level=row.risk_level,  # type: ignore[arg-type]
        assessment_status=row.assessment_status,
        confidence=row.confidence,
        assessment_reliable=row.assessment_reliable,
        quality_reasons=tuple(row.quality_reasons),
        evidence_path=row.evidence_path,
        rgb_evidence_path=row.rgb_evidence_path,
        pseudo_bev_path=row.pseudo_bev_path,
        created_at=created_at,
    )


def _row_from_record(record: RiskSnapshotRecord) -> FrameAssessmentRow:
    return FrameAssessmentRow(
        id=uuid.UUID(record.snapshot_id),
        job_id=uuid.UUID(record.job_id),
        frame_index=record.frame_index,
        timestamp_sec=record.timestamp_sec,
        risk_level=record.risk_level,
        assessment_status=record.assessment_status,
        confidence=record.confidence,
        assessment_reliable=record.assessment_reliable,
        quality_reasons=list(record.quality_reasons),
        evidence_path=record.evidence_path,
        rgb_evidence_path=record.rgb_evidence_path,
        pseudo_bev_path=record.pseudo_bev_path,
        created_at=record.created_at,
    )


__all__ = [
    "InMemoryRiskSnapshotRepository",
    "RiskSnapshotRepository",
    "SqlAlchemyRiskSnapshotRepository",
]
