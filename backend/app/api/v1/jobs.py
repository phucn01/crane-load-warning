"""Video job status, processing preview, and result endpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from app.api.dependencies import ProcessingHistoryServiceDep, VideoJobRepositoryDep
from app.models import JobStatus, ProcessingJobRecord, RiskSegment, VideoJob
from app.repositories import VideoJobRepository
from app.schemas.processing_history import (
    ProcessingJobHistoryPage,
    ProcessingJobHistoryResponse,
)
from app.schemas.video_job import FrameRiskResultsResponse, VideoJobResponse
from app.services.processing_history_service import ProcessingHistoryService

router = APIRouter(prefix="/jobs", tags=["video-jobs"])
BOUNDARY = "frame"


@router.get("", response_model=ProcessingJobHistoryPage)
def list_jobs(
    history: ProcessingHistoryServiceDep,
    status_filter: Annotated[
        JobStatus | None,
        Query(alias="status"),
    ] = None,
    media_type: Annotated[str | None, Query(pattern="^(image|video)$")] = None,
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProcessingJobHistoryPage:
    items, total = history.jobs.list(
        status=status_filter,
        media_type=media_type,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
    )
    return ProcessingJobHistoryPage(
        items=[_history_job_response(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}",
    response_model=VideoJobResponse | ProcessingJobHistoryResponse,
)
def get_job(
    job_id: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> VideoJobResponse | ProcessingJobHistoryResponse:
    job = repository.get(job_id)
    if job is not None:
        return _job_response(job)
    persisted = history.jobs.get(job_id)
    if persisted is not None:
        return _history_job_response(persisted)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="processing job was not found",
    )


@router.get("/{job_id}/frames", response_model=FrameRiskResultsResponse)
def get_frame_results(
    job_id: str,
    repository: VideoJobRepositoryDep,
    after_frame: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> FrameRiskResultsResponse:
    """Return lightweight per-frame risk results using a frame cursor."""
    job = _required_job(repository, job_id)
    items, available_count = repository.frame_results_page(
        job_id,
        after_frame=after_frame,
        limit=limit,
    )
    next_after_frame = items[-1].frame_number if items else after_frame
    return FrameRiskResultsResponse(
        job_id=job_id,
        job_status=job.status.value,
        items=[
            {
                "frame_number": item.frame_number,
                "timestamp_seconds": item.timestamp_seconds,
                "risk_level": item.risk_level,
            }
            for item in items
        ],
        next_after_frame=next_after_frame,
        has_more=available_count > next_after_frame,
    )


@router.get("/{job_id}/stream")
def stream_job(
    job_id: str,
    repository: VideoJobRepositoryDep,
) -> StreamingResponse:
    _required_job(repository, job_id)

    def frames() -> Iterator[bytes]:
        last_frame = 0
        while True:
            preview = repository.wait_for_preview(job_id, last_frame, timeout=1.0)
            if preview is not None:
                last_frame, jpeg = preview
                yield (
                    (
                        f"--{BOUNDARY}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
            job = repository.get(job_id)
            if job is None or job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                break
        yield f"--{BOUNDARY}--\r\n".encode("ascii")

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{job_id}/result")
def get_result(
    job_id: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> FileResponse:
    job_status, output_path, _ = _video_artifacts(job_id, repository, history)
    _require_completed_artifact(job_status, "video result")
    if not output_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video result file was not found",
        )
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"crane-safety-{job_id}.mp4",
        content_disposition_type="inline",
    )


@router.get("/{job_id}/image-evidence")
def get_image_evidence(
    job_id: str,
    history: ProcessingHistoryServiceDep,
) -> dict[str, str | None]:
    """Return all review views for a persisted image job."""
    job = history.jobs.get(job_id)
    if job is None or job.media_type != "image":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image job was not found")
    if job.status is not JobStatus.COMPLETED or job.output_path is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="image evidence is not ready")
    output = str(job.output_path).replace("\\", "/")
    if not output.lstrip("/").startswith("evidence/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image evidence was not found")
    output = "/" + output.lstrip("/")
    evidence_root = output.rsplit("/", 1)[0]
    input_name = job.input_path.name if job.input_path is not None else None
    return {
        "original_url": None if input_name is None else f"/uploads/images/{input_name}",
        "detection_url": f"{evidence_root}/rgb.png",
        "bev_url": f"{evidence_root}/pseudo_bev.png",
        "combined_url": output,
    }


@router.get("/{job_id}/report")
def get_report(
    job_id: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> FileResponse:
    job_status, _, report_path = _video_artifacts(job_id, repository, history)
    _require_completed_artifact(job_status, "video report")
    if not report_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video report file was not found",
        )
    return FileResponse(
        report_path,
        media_type="application/json",
        filename=f"crane-safety-{job_id}-report.json",
        content_disposition_type="attachment",
    )


@router.get("/{job_id}/download")
def download_result(
    job_id: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> FileResponse:
    job_status, output_path, _ = _video_artifacts(job_id, repository, history)
    _require_completed_artifact(job_status, "video download")
    if not output_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video result file was not found",
        )
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"crane-safety-{job_id}.mp4",
        content_disposition_type="attachment",
    )


@router.get("/{job_id}/segments/{segment_id}")
def get_risk_segment(
    job_id: str,
    segment_id: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> FileResponse:
    job = repository.get(job_id)
    if job is not None:
        _require_completed_artifact(job.status, "risk segments")
        segment_path = _required_segment(job, segment_id).output_path
    else:
        job_status, output_path, report_path = _video_artifacts(
            job_id,
            repository,
            history,
        )
        _require_completed_artifact(job_status, "risk segments")
        _required_report_segment(report_path, segment_id)
        segment_path = _segment_root(output_path) / f"{segment_id}.mp4"
    if not segment_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="risk segment video file was not found",
        )
    return FileResponse(
        segment_path,
        media_type="video/mp4",
        filename=f"risk-segment-{segment_id}.mp4",
        content_disposition_type="inline",
    )


@router.get(
    "/{job_id}/segments/{segment_id}/evidence/{frame_number}/{view}"
)
def get_frame_evidence(
    job_id: str,
    segment_id: str,
    frame_number: int,
    view: str,
    repository: VideoJobRepositoryDep,
    history: ProcessingHistoryServiceDep,
) -> FileResponse:
    if view not in {"original", "rgb", "bev"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frame evidence was not found",
        )
    job = repository.get(job_id)
    if job is not None:
        _require_completed_artifact(job.status, "frame evidence")
        segment = _required_segment(job, segment_id)
        evidence = next(
            (
                item
                for item in segment.frame_evidence
                if item.frame_number == frame_number
            ),
            None,
        )
        if evidence is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="frame evidence was not found",
            )
        evidence_path = {
            "original": evidence.original_path,
            "rgb": evidence.rgb_path,
            "bev": evidence.pseudo_bev_path,
        }[view]
    else:
        job_status, output_path, report_path = _video_artifacts(
            job_id,
            repository,
            history,
        )
        _require_completed_artifact(job_status, "frame evidence")
        segment_payload = _required_report_segment(report_path, segment_id)
        _require_report_evidence(segment_payload, frame_number)
        evidence_path = (
            _segment_root(output_path)
            / f"{segment_id}_evidence"
            / f"frame-{frame_number}-{view}.png"
        )
    if not evidence_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frame evidence file was not found",
        )
    return FileResponse(
        evidence_path,
        media_type="image/png",
        filename=f"frame-{frame_number}-{view}.png",
        content_disposition_type="inline",
    )


def _video_artifacts(
    job_id: str,
    repository: VideoJobRepository,
    history: ProcessingHistoryService,
) -> tuple[JobStatus, Path, Path]:
    job = repository.get(job_id)
    if job is not None:
        return job.status, job.output_path, job.report_path
    persisted = history.jobs.get(job_id)
    if (
        persisted is None
        or persisted.media_type != "video"
        or persisted.output_path is None
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video job was not found",
        )
    return (
        persisted.status,
        persisted.output_path,
        persisted.output_path.with_suffix(".report.json"),
    )


def _require_completed_artifact(job_status: JobStatus, label: str) -> None:
    if job_status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{label} is not available while job is {job_status.value}",
        )


def _segment_root(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_segments"


def _required_report_segment(report_path: Path, segment_id: str) -> dict[str, object]:
    payload = _read_report_payload(report_path)
    segments = payload.get("risk_segments")
    if not isinstance(segments, list):
        segments = []
    segment = next(
        (
            item
            for item in segments
            if isinstance(item, dict) and item.get("segment_id") == segment_id
        ),
        None,
    )
    if segment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="risk segment was not found",
        )
    return segment


def _require_report_evidence(
    segment: dict[str, object],
    frame_number: int,
) -> None:
    evidence = segment.get("evidence")
    if not isinstance(evidence, list) or not any(
        isinstance(item, dict) and item.get("frame_number") == frame_number
        for item in evidence
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frame evidence was not found",
        )


def _read_report_payload(report_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video report file was not found",
        ) from error
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video report file was not found",
        )
    return payload


def _required_job(repository: VideoJobRepository, job_id: str) -> VideoJob:
    job = repository.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video job was not found",
        )
    return job


def _required_segment(job: VideoJob, segment_id: str) -> RiskSegment:
    segment = next(
        (item for item in job.risk_segments if item.segment_id == segment_id),
        None,
    )
    if segment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="risk segment was not found",
        )
    return segment


def _job_response(job: VideoJob) -> VideoJobResponse:
    prefix = f"/api/v1/jobs/{job.job_id}"
    summary = None
    if job.status is JobStatus.COMPLETED:
        summary = {
            "processed_frames": job.current_frame,
            "safe_frames": job.safe_frame_count,
            "warning_frames": job.warning_frame_count,
            "danger_frames": job.danger_frame_count,
            "max_risk_level": job.max_risk_level,
            "average_processing_fps": job.processing_fps,
            "risk_segment_count": len(job.risk_segments),
        }
    segments = [
        {
            "segment_id": segment.segment_id,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "risk_start_frame": segment.risk_start_frame,
            "risk_end_frame": segment.risk_end_frame,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "max_risk_level": segment.max_risk_level,
            "warning_frame_count": segment.warning_frame_count,
            "danger_frame_count": segment.danger_frame_count,
            "frame_evidence": [
                {
                    "frame_number": item.frame_number,
                    "timestamp_seconds": item.timestamp_seconds,
                    "risk_level": item.risk_level,
                    "original_url": (
                        f"{prefix}/segments/{segment.segment_id}/evidence/"
                        f"{item.frame_number}/original"
                    ),
                    "rgb_url": (
                        f"{prefix}/segments/{segment.segment_id}/evidence/"
                        f"{item.frame_number}/rgb"
                    ),
                    "pseudo_bev_url": (
                        f"{prefix}/segments/{segment.segment_id}/evidence/"
                        f"{item.frame_number}/bev"
                    ),
                }
                for item in segment.frame_evidence
            ],
            "result_url": (
                f"{prefix}/segments/{segment.segment_id}"
            ),
            "output_codec": segment.output_codec,
            "browser_playback_compatible": (
                segment.browser_playback_compatible
            ),
            "playback_warning": segment.playback_warning,
        }
        for segment in job.risk_segments
    ]
    return VideoJobResponse.model_validate(
        {
            "job_id": job.job_id,
            "status": job.status.value,
            "input_path": str(job.input_path),
            "output_path": str(job.output_path),
            "current_frame": job.current_frame,
            "total_frames": job.total_frames,
            "progress": job.progress,
            "processing_fps": job.processing_fps,
            "elapsed_seconds": job.elapsed_seconds,
            "current_risk_level": job.current_risk_level,
            "max_risk_level": job.max_risk_level,
            "safe_frame_count": job.safe_frame_count,
            "warning_frame_count": job.warning_frame_count,
            "danger_frame_count": job.danger_frame_count,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "stream_url": f"{prefix}/stream",
            "frame_results_url": f"{prefix}/frames",
            "result_url": (
                f"{prefix}/result" if job.status is JobStatus.COMPLETED else None
            ),
            "download_url": (
                f"{prefix}/download" if job.status is JobStatus.COMPLETED else None
            ),
            "report_url": (
                f"{prefix}/report" if job.status is JobStatus.COMPLETED else None
            ),
            "summary": summary,
            "risk_segments": segments,
            "output_codec": job.output_codec,
            "browser_playback_compatible": job.browser_playback_compatible,
            "playback_warning": job.playback_warning,
        }
    )


def _history_job_response(
    job: ProcessingJobRecord,
) -> ProcessingJobHistoryResponse:
    output_path = None if job.output_path is None else str(job.output_path)
    # Image evidence is exposed as a URL path. Normalize legacy records that
    # may have been stored with multiple leading slashes.
    if output_path:
        public_path = output_path.replace("\\", "/")
        if public_path.lstrip("/").startswith("evidence/"):
            output_path = "/" + public_path.lstrip("/")
    return ProcessingJobHistoryResponse(
        id=job.job_id,
        media_type=job.media_type,
        input_name=job.input_name,
        input_path=None if job.input_path is None else str(job.input_path),
        output_path=output_path,
        status=job.status.value,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        safe_frame_count=job.safe_frame_count,
        warning_frame_count=job.warning_frame_count,
        danger_frame_count=job.danger_frame_count,
        max_risk_level=job.max_risk_level,
        processing_time_ms=job.processing_time_ms,
        average_processing_fps=job.average_processing_fps,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


__all__ = ["router"]
