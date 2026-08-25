"""Video job status, processing preview, and result endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from app.api.dependencies import VideoJobRepositoryDep
from app.models import JobStatus, RiskSegment, VideoJob
from app.repositories import VideoJobRepository
from app.schemas.video_job import FrameRiskResultsResponse, VideoJobResponse

router = APIRouter(prefix="/jobs", tags=["video-jobs"])
BOUNDARY = "frame"


@router.get("/{job_id}", response_model=VideoJobResponse)
def get_job(
    job_id: str,
    repository: VideoJobRepositoryDep,
) -> VideoJobResponse:
    job = _required_job(repository, job_id)
    return _job_response(job)


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
) -> FileResponse:
    job = _required_job(repository, job_id)
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"video result is not available while job is {job.status.value}",
        )
    if not job.output_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video result file was not found",
        )
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"crane-safety-{job.job_id}.mp4",
        content_disposition_type="inline",
    )


@router.get("/{job_id}/report")
def get_report(
    job_id: str,
    repository: VideoJobRepositoryDep,
) -> FileResponse:
    job = _required_job(repository, job_id)
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"video report is not available while job is {job.status.value}",
        )
    if not job.report_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video report file was not found",
        )
    return FileResponse(
        job.report_path,
        media_type="application/json",
        filename=f"crane-safety-{job.job_id}-report.json",
        content_disposition_type="attachment",
    )


@router.get("/{job_id}/download")
def download_result(
    job_id: str,
    repository: VideoJobRepositoryDep,
) -> FileResponse:
    job = _required_job(repository, job_id)
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"video download is not available while job is {job.status.value}",
        )
    if not job.output_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="video result file was not found",
        )
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=f"crane-safety-{job.job_id}.mp4",
        content_disposition_type="attachment",
    )


@router.get("/{job_id}/segments/{segment_id}")
def get_risk_segment(
    job_id: str,
    segment_id: str,
    repository: VideoJobRepositoryDep,
) -> FileResponse:
    job = _required_job(repository, job_id)
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"risk segments are not available while job is {job.status.value}",
        )
    segment = _required_segment(job, segment_id)
    if not segment.output_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="risk segment video file was not found",
        )
    return FileResponse(
        segment.output_path,
        media_type="video/mp4",
        filename=f"risk-segment-{segment.segment_id}.mp4",
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
) -> FileResponse:
    job = _required_job(repository, job_id)
    if job.status is not JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "frame evidence is not available while job is "
                f"{job.status.value}"
            ),
        )
    segment = _required_segment(job, segment_id)
    evidence = next(
        (
            item
            for item in segment.frame_evidence
            if item.frame_number == frame_number
        ),
        None,
    )
    if evidence is None or view not in {"original", "rgb", "bev"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frame evidence was not found",
        )
    evidence_paths = {
        "original": evidence.original_path,
        "rgb": evidence.rgb_path,
        "bev": evidence.pseudo_bev_path,
    }
    evidence_path = evidence_paths[view]
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


__all__ = ["router"]
