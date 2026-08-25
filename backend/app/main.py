"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from app.api.v1 import detection, health, jobs
from app.core.config import Settings
from app.core.logging import configure_logging
from app.repositories import VideoJobRepository
from app.services.image_processing_service import ImageProcessingService
from app.services.video_processing_service import VideoProcessingService
from app.services.video_transcoding_service import BrowserVideoConverter
from app.workers import VideoWorker


def create_app(
    *,
    settings: Settings | None = None,
    image_processing_service: Any | None = None,
    video_job_repository: VideoJobRepository | None = None,
    video_worker: Any | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()
    repository = video_job_repository or VideoJobRepository()
    upload_root = resolved_settings.video_upload_root or (
        resolved_settings.evidence_root.parent / "uploads" / "videos"
    )
    output_root = resolved_settings.video_output_root or (
        resolved_settings.evidence_root.parent / "outputs" / "videos"
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        service = image_processing_service
        if service is None:
            service = ImageProcessingService.from_settings(resolved_settings)
        application.state.image_processing_service = service
        application.state.video_job_repository = repository
        application.state.video_upload_root = upload_root
        application.state.video_output_root = output_root
        upload_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        worker = video_worker or VideoWorker(
            VideoProcessingService(
                frame_processor=service,
                repository=repository,
                segment_pre_roll_seconds=(
                    resolved_settings.risk_segment_pre_roll_seconds
                ),
                segment_post_roll_seconds=(
                    resolved_settings.risk_segment_post_roll_seconds
                ),
                video_converter=BrowserVideoConverter.discover(
                    resolved_settings.ffmpeg_path
                ),
            )
        )
        application.state.video_worker = worker
        if resolved_settings.preload_models:
            await run_in_threadpool(service.preload_models)
        try:
            yield
        finally:
            await run_in_threadpool(worker.close)

    configure_logging()
    application = FastAPI(
        title="Crane Load Warning API",
        version="1.0.0",
        description="Local image and asynchronous video safety assessment.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )
    application.include_router(health.router, prefix="/api/v1")
    application.include_router(detection.router, prefix="/api/v1")
    application.include_router(jobs.router, prefix="/api/v1")
    application.mount(
        "/evidence",
        StaticFiles(directory=resolved_settings.evidence_root, check_dir=False),
        name="evidence",
    )
    return application


app = create_app()


__all__ = ["app", "create_app"]
