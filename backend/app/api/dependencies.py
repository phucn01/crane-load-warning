"""Typed dependencies shared by API endpoints."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.repositories import VideoJobRepository
from app.services.processing_history_service import ProcessingHistoryService
from app.workers import VideoWorker


def get_video_job_repository(request: Request) -> VideoJobRepository:
    """Return the application's video-job repository with its concrete type."""
    return cast(VideoJobRepository, request.app.state.video_job_repository)


def get_video_worker(request: Request) -> VideoWorker:
    """Return the application's video worker with its concrete type."""
    return cast(VideoWorker, request.app.state.video_worker)


def get_processing_history_service(request: Request) -> ProcessingHistoryService:
    return cast(
        ProcessingHistoryService,
        request.app.state.processing_history_service,
    )


VideoJobRepositoryDep = Annotated[
    VideoJobRepository,
    Depends(get_video_job_repository),
]

VideoWorkerDep = Annotated[
    VideoWorker,
    Depends(get_video_worker),
]

ProcessingHistoryServiceDep = Annotated[
    ProcessingHistoryService,
    Depends(get_processing_history_service),
]


__all__ = [
    "ProcessingHistoryServiceDep",
    "VideoJobRepositoryDep",
    "VideoWorkerDep",
    "get_processing_history_service",
    "get_video_job_repository",
    "get_video_worker",
]
