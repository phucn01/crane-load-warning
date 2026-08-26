"""Read-only API for persisted frame assessments and evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import ProcessingHistoryServiceDep
from app.models import RiskSnapshotRecord
from app.schemas.processing_history import RiskSnapshotPage, RiskSnapshotResponse
from app.services.processing_history_service import ProcessingHistoryService

router = APIRouter(prefix="/risk-snapshots", tags=["risk-snapshots"])


@router.get("", response_model=RiskSnapshotPage)
def list_risk_snapshots(
    history: ProcessingHistoryServiceDep,
    job_id: str | None = None,
    risk_level: Annotated[
        str | None,
        Query(pattern="^(SAFE|WARNING|DANGER)$"),
    ] = None,
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Literal["created_desc", "frame_asc"] = "created_desc",
) -> RiskSnapshotPage:
    items, total = history.snapshots.list(
        job_id=job_id,
        risk_level=risk_level,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
        order=order,
    )
    return RiskSnapshotPage(
        items=[_snapshot_response(item, history) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{snapshot_id}", response_model=RiskSnapshotResponse)
def get_risk_snapshot(
    snapshot_id: str,
    history: ProcessingHistoryServiceDep,
) -> RiskSnapshotResponse:
    snapshot = history.snapshots.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="risk snapshot was not found",
        )
    return _snapshot_response(snapshot, history)


def _snapshot_response(
    snapshot: RiskSnapshotRecord,
    history: ProcessingHistoryService,
) -> RiskSnapshotResponse:
    evidence_path = snapshot.evidence_path
    job = history.jobs.get(snapshot.job_id)
    if (
        job is not None
        and job.media_type == "image"
        and job.input_path is not None
    ):
        evidence_path = f"/uploads/images/{job.input_path.name}"
    return RiskSnapshotResponse(
        id=snapshot.snapshot_id,
        job_id=snapshot.job_id,
        frame_index=snapshot.frame_index,
        timestamp_sec=snapshot.timestamp_sec,
        risk_level=snapshot.risk_level,
        assessment_status=snapshot.assessment_status,
        confidence=snapshot.confidence,
        assessment_reliable=snapshot.assessment_reliable,
        quality_reasons=list(snapshot.quality_reasons),
        evidence_path=evidence_path,
        rgb_evidence_path=snapshot.rgb_evidence_path,
        pseudo_bev_path=snapshot.pseudo_bev_path,
        created_at=snapshot.created_at,
    )


__all__ = ["router"]
