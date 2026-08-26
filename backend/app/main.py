"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from app.api.v1 import detection, health, jobs, risk_snapshots
from app.core.config import PROJECT_ROOT, Settings
from app.core.logging import (
    bind_request_id,
    configure_logging,
    log_operation,
    reset_request_id,
)
from app.infrastructure.db import create_database_session_factory
from app.repositories import (
    InMemoryProcessingJobRepository,
    InMemoryRiskSnapshotRepository,
    SqlAlchemyProcessingJobRepository,
    SqlAlchemyRiskSnapshotRepository,
    VideoJobRepository,
)
from app.services.image_processing_service import ImageProcessingService
from app.services.processing_history_service import (
    ProcessingHistoryService,
    RiskSnapshotPolicy,
)
from app.services.video_processing_service import VideoProcessingService
from app.services.video_transcoding_service import BrowserVideoConverter
from app.workers import VideoWorker

LOGGER = logging.getLogger(__name__)
API_LOG_SEPARATOR = "=" * 100


def create_app(
    *,
    settings: Settings | None = None,
    image_processing_service: Any | None = None,
    video_job_repository: VideoJobRepository | None = None,
    video_worker: Any | None = None,
) -> FastAPI:
    if settings is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    resolved_settings = settings or Settings.from_environment()
    repository = video_job_repository or VideoJobRepository()
    history_service = _processing_history_service(resolved_settings)
    upload_root = resolved_settings.video_upload_root or (
        resolved_settings.evidence_root.parent / "uploads" / "videos"
    )
    output_root = resolved_settings.video_output_root or (
        resolved_settings.evidence_root.parent / "outputs" / "videos"
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        LOGGER.info("=== APPLICATION STARTING ===")
        service = image_processing_service
        with log_operation(LOGGER, "initialize_image_processing_service"):
            if service is None:
                service = ImageProcessingService.from_settings(resolved_settings)
        application.state.image_processing_service = service
        application.state.video_job_repository = repository
        application.state.processing_history_service = history_service
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
                history_service=history_service,
                snapshot_evidence_root=resolved_settings.evidence_root,
            )
        )
        application.state.video_worker = worker
        if resolved_settings.preload_models:
            with log_operation(LOGGER, "preload_models"):
                await run_in_threadpool(service.preload_models)
        LOGGER.info("=== APPLICATION STARTED ===")
        try:
            yield
        finally:
            with log_operation(LOGGER, "shutdown_video_worker"):
                await run_in_threadpool(worker.close)
            with log_operation(LOGGER, "close_video_job_repository"):
                await run_in_threadpool(repository.close)
            with log_operation(LOGGER, "close_processing_history"):
                await run_in_threadpool(history_service.close)
            LOGGER.info("=== APPLICATION STOPPED ===")

    configure_logging()
    application = FastAPI(
        title="Crane Load Warning API",
        version="1.0.0",
        description="Local image and asynchronous video safety assessment.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.persistence_backend = (
        "postgresql" if resolved_settings.database_url else "memory"
    )

    @application.middleware("http")
    async def log_http_request(request: Any, call_next: Any) -> Any:
        request_id = uuid4().hex
        token = bind_request_id(request_id)
        started = perf_counter()
        LOGGER.info(API_LOG_SEPARATOR)
        LOGGER.info(
            "=== HTTP START ===\n    METHOD       : %s\n    PATH         : %s",
            request.method,
            request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception as error:
            LOGGER.exception(
                "=== HTTP ERROR ===\n    METHOD       : %s\n    PATH         : %s\n"
                "    DURATION_MS  : %.3f\n    ERROR_TYPE   : %s",
                request.method,
                request.url.path,
                (perf_counter() - started) * 1000.0,
                type(error).__name__,
            )
            LOGGER.info(API_LOG_SEPARATOR)
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            LOGGER.info(
                "=== HTTP END ===\n    METHOD       : %s\n    PATH         : %s\n"
                "    STATUS_CODE  : %s\n    DURATION_MS  : %.3f",
                request.method,
                request.url.path,
                response.status_code,
                (perf_counter() - started) * 1000.0,
            )
            LOGGER.info(API_LOG_SEPARATOR)
            return response
        finally:
            reset_request_id(token)

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
    application.include_router(risk_snapshots.router, prefix="/api/v1")
    application.mount(
        "/evidence",
        StaticFiles(directory=resolved_settings.evidence_root, check_dir=False),
        name="evidence",
    )
    return application


def _processing_history_service(settings: Settings) -> ProcessingHistoryService:
    snapshot_policy = RiskSnapshotPolicy.from_yaml(settings.persistence_config)
    close_callback = None
    if settings.database_url is None:
        jobs = InMemoryProcessingJobRepository()
        snapshots = InMemoryRiskSnapshotRepository(jobs)
    else:
        sessions = create_database_session_factory(settings.database_url)
        jobs = SqlAlchemyProcessingJobRepository(sessions)
        snapshots = SqlAlchemyRiskSnapshotRepository(sessions)
        engine = sessions.kw["bind"]
        close_callback = engine.dispose
    return ProcessingHistoryService(
        jobs,
        snapshots,
        snapshot_policy=snapshot_policy,
        job_update_interval_frames=settings.database_job_update_interval,
        close_callback=close_callback,
    )


app = create_app()


__all__ = ["app", "create_app"]
