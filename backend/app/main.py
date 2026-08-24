"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from app.api.v1 import detection, health
from app.core.config import Settings
from app.core.logging import configure_logging
from app.services.image_processing_service import ImageProcessingService


def create_app(
    *,
    settings: Settings | None = None,
    image_processing_service: Any | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        service = image_processing_service
        if service is None:
            service = ImageProcessingService.from_settings(resolved_settings)
        application.state.image_processing_service = service
        if resolved_settings.preload_models:
            await run_in_threadpool(service.preload_models)
        yield

    configure_logging()
    application = FastAPI(
        title="Crane Load Warning API",
        version="1.0.0",
        description="Local image safety assessment using the offline pipeline.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.include_router(health.router, prefix="/api/v1")
    application.include_router(detection.router, prefix="/api/v1")
    application.mount(
        "/evidence",
        StaticFiles(directory=resolved_settings.evidence_root, check_dir=False),
        name="evidence",
    )
    return application


app = create_app()


__all__ = ["app", "create_app"]
