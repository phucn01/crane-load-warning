"""Read-only API for sampled WARNING/DANGER evidence history."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import ProcessingHistoryServiceDep
from app.models import RiskSnapshotRecord
from app.schemas.processing_history import RiskSnapshotPage, RiskSnapshotResponse

router = APIRouter(prefix="/risk-snapshots", tags=["risk-snapshots"])


@router.get("", response_model=RiskSnapshotPage)
def list_risk_snapshots(
    history: ProcessingHistoryServiceDep,
    job_id: str | None = None,
    risk_level: Annotated[
        str | None,
        Query(pattern="^(WARNING|DANGER)$"),
    ] = None,
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RiskSnapshotPage:
    items, total = history.snapshots.list(
        job_id=job_id,
        risk_level=risk_level,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )
    return RiskSnapshotPage(
        items=[_snapshot_response(item) for item in items],
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
    return _snapshot_response(snapshot)


def _snapshot_response(snapshot: RiskSnapshotRecord) -> RiskSnapshotResponse:
    return RiskSnapshotResponse(
        id=snapshot.snapshot_id,
        job_id=snapshot.job_id,
        frame_index=snapshot.frame_index,
        timestamp_sec=snapshot.timestamp_sec,
        risk_level=snapshot.risk_level,
        confidence=snapshot.confidence,
        assessment_reliable=snapshot.assessment_reliable,
        quality_reasons=list(snapshot.quality_reasons),
        evidence_path=snapshot.evidence_path,
        rgb_evidence_path=snapshot.rgb_evidence_path,
        pseudo_bev_path=snapshot.pseudo_bev_path,
        created_at=snapshot.created_at,
    )


__all__ = ["router"]
