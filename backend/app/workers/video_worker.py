"""Small in-process executor for blocking video inference workloads."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from app.services.video_processing_service import VideoProcessingService


class VideoWorker:
    def __init__(
        self,
        service: VideoProcessingService,
        *,
        max_workers: int = 1,
    ) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="video-worker",
        )
        self._futures: set[Future[None]] = set()
        self._lock = Lock()

    def submit(self, job_id: str) -> None:
        future = self._executor.submit(self.service.process, job_id)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard)

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _discard(self, future: Future[None]) -> None:
        with self._lock:
            self._futures.discard(future)


__all__ = ["VideoWorker"]
