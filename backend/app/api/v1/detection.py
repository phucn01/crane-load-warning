"""Image upload endpoint and transport-level validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import VideoJobRepositoryDep, VideoWorkerDep
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
) -> ImageDetectionResponse:
    try:
        _validate_upload_type(file)
        maximum = request.app.state.settings.max_upload_bytes
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

        image_bgr = _decode_supported_image(payload)
        service = request.app.state.image_processing_service
        try:
            return await run_in_threadpool(service.process, image_bgr)
        except Exception as error:
            LOGGER.exception("image pipeline failed: %s", type(error).__name__)
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
) -> VideoJobCreatedResponse:
    destination: Path | None = None
    job = None
    try:
        suffix = _validate_video_upload_type(file)
        settings = request.app.state.settings
        upload_root: Path = request.app.state.video_upload_root
        output_root: Path = request.app.state.video_output_root
        upload_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = upload_root / f"{uuid4().hex}{suffix}"
        await run_in_threadpool(
            _copy_upload,
            file,
            destination,
            settings.max_video_upload_bytes,
        )
        output_path = output_root / f"{uuid4().hex}.mp4"
        job = repository.create(input_path=destination, output_path=output_path)
        worker.submit(job.job_id)
        prefix = f"/api/v1/jobs/{job.job_id}"
        return VideoJobCreatedResponse(
            job_id=job.job_id,
            status_url=prefix,
            stream_url=f"{prefix}/stream",
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
        LOGGER.exception("video upload failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="video upload failed",
        ) from error
    finally:
        await file.close()


def _validate_upload_type(file: UploadFile) -> None:
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
