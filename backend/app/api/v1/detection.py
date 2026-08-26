"""Image upload endpoint and transport-level validation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    ProcessingHistoryServiceDep,
    VideoJobRepositoryDep,
    VideoWorkerDep,
)
from app.core.logging import log_operation
from app.models import JobStatus, ProcessingJobRecord
from app.schemas.detection import ImageDetectionResponse
from app.schemas.video_job import VideoJobCreatedResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/detection", tags=["detection"])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    history: ProcessingHistoryServiceDep,
) -> ImageDetectionResponse:
    analysis_id: str | None = None
    try:
        with log_operation(LOGGER, "validate_image_upload"):
            suffix = _validate_upload_type(file)
        maximum = request.app.state.settings.max_upload_bytes
        with log_operation(LOGGER, "read_image_upload"):
            payload = await file.read(maximum + 1)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="uploaded image is empty",
            )
        if len(payload) > maximum:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"uploaded image exceeds the {maximum}-byte limit",
            )

        with log_operation(
            LOGGER, "decode_image_upload", upload_bytes=len(payload)
        ):
            image_bgr = _decode_supported_image(payload)
        service = request.app.state.image_processing_service
        analysis_id = uuid4().hex
        input_path: Path = request.app.state.image_upload_root / (
            f"{analysis_id}{suffix}"
        )
        with log_operation(
            LOGGER, "persist_image_upload", upload_bytes=len(payload)
        ):
            try:
                await run_in_threadpool(_write_image_upload, payload, input_path)
            except OSError as error:
                input_path.unlink(missing_ok=True)
                LOGGER.exception(
                    "=== ERROR | OPERATION=PERSIST_IMAGE_UPLOAD | "
                    "ERROR_TYPE=%s ===",
                    type(error).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="image upload could not be stored",
                ) from error
        created_at = datetime.now(UTC)
        history.create_job(
            ProcessingJobRecord(
                job_id=analysis_id,
                media_type="image",
                input_name=Path(file.filename or "image").name,
                input_path=input_path,
                output_path=None,
                status=JobStatus.QUEUED,
                total_frames=None,
                processed_frames=None,
                safe_frame_count=0,
                warning_frame_count=0,
                danger_frame_count=0,
                max_risk_level=None,
                processing_time_ms=None,
                average_processing_fps=None,
                error_message=None,
                created_at=created_at,
                started_at=None,
                completed_at=None,
            )
        )
        history.mark_processing(analysis_id)
        try:
            with log_operation(LOGGER, "process_image_request"):
                response = await run_in_threadpool(
                    service.process,
                    image_bgr,
                    run_id=analysis_id,
                )
            history.complete_image(analysis_id, response)
            history.persist_image_snapshot(
                analysis_id,
                response,
                original_evidence_path=f"/uploads/images/{input_path.name}",
            )
            return response
        except Exception as error:
            history.fail_job(analysis_id, "image processing failed")
            LOGGER.exception(
                "=== ERROR | OPERATION=IMAGE_PIPELINE | ERROR_TYPE=%s ===",
                type(error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="image processing failed",
            ) from error
    finally:
        await file.close()


@router.post(
    "/video",
    response_model=VideoJobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def detect_video(
    request: Request,
    file: Annotated[UploadFile, File()],
    repository: VideoJobRepositoryDep,
    worker: VideoWorkerDep,
    history: ProcessingHistoryServiceDep,
) -> VideoJobCreatedResponse:
    destination: Path | None = None
    job = None
    try:
        with log_operation(LOGGER, "validate_video_upload"):
            suffix = _validate_video_upload_type(file)
        settings = request.app.state.settings
        upload_root: Path = request.app.state.video_upload_root
        output_root: Path = request.app.state.video_output_root
        upload_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = upload_root / f"{uuid4().hex}{suffix}"
        with log_operation(LOGGER, "persist_video_upload"):
            await run_in_threadpool(
                _copy_upload,
                file,
                destination,
                settings.max_video_upload_bytes,
            )
        output_path = output_root / f"{uuid4().hex}.mp4"
        with log_operation(LOGGER, "create_and_submit_video_job"):
            job = repository.create(input_path=destination, output_path=output_path)
            history.create_job(
                ProcessingJobRecord(
                    job_id=job.job_id,
                    media_type="video",
                    input_name=Path(file.filename or "video").name,
                    input_path=destination,
                    output_path=output_path,
                    status=JobStatus.QUEUED,
                    total_frames=None,
                    processed_frames=0,
                    safe_frame_count=0,
                    warning_frame_count=0,
                    danger_frame_count=0,
                    max_risk_level=None,
                    processing_time_ms=None,
                    average_processing_fps=None,
                    error_message=None,
                    created_at=job.created_at,
                    started_at=None,
                    completed_at=None,
                )
            )
            worker.submit(job.job_id)
        LOGGER.info("=== EVENT | VIDEO_JOB_SUBMITTED | JOB_ID=%s ===", job.job_id)
        prefix = f"/api/v1/jobs/{job.job_id}"
        return VideoJobCreatedResponse(
            job_id=job.job_id,
            status_url=prefix,
            stream_url=f"{prefix}/stream",
            frame_results_url=f"{prefix}/frames",
            result_url=f"{prefix}/result",
        )
    except HTTPException:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    except Exception as error:
        if destination is not None:
            destination.unlink(missing_ok=True)
        if job is not None:
            repository.remove(job.job_id)
            history.fail_job(job.job_id, "video upload failed")
        LOGGER.exception(
            "=== ERROR | OPERATION=VIDEO_UPLOAD | ERROR_TYPE=%s ===",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="video upload failed",
        ) from error
    finally:
        await file.close()


def _validate_upload_type(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if (
        suffix not in SUPPORTED_EXTENSIONS
        or content_type not in SUPPORTED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only JPG, JPEG, and PNG images are supported",
        )
    return suffix


def _write_image_upload(payload: bytes, destination: Path) -> None:
    with destination.open("xb") as output:
        output.write(payload)


def _decode_supported_image(payload: bytes) -> np.ndarray:
    is_jpeg = payload.startswith(b"\xff\xd8\xff")
    is_png = payload.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is not a valid JPG or PNG image",
        )
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded image could not be decoded",
        )
    return image


def _validate_video_upload_type(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if (
        suffix not in SUPPORTED_VIDEO_EXTENSIONS
        or content_type not in SUPPORTED_VIDEO_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only MP4, MOV, AVI, MKV, and WebM videos are supported",
        )
    return suffix


def _copy_upload(file: UploadFile, destination: Path, maximum: int) -> None:
    total = 0
    with destination.open("xb") as output:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > maximum:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"uploaded video exceeds the {maximum}-byte limit",
                )
            output.write(chunk)
    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded video is empty",
        )


__all__ = ["router"]
