"""Image upload endpoint and transport-level validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.schemas.detection import ImageDetectionResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/detection", tags=["detection"])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}


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


__all__ = ["router"]
