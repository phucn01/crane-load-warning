"""Thread-safe in-memory repository for video jobs and latest previews."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, RLock
from uuid import uuid4

from app.models import JobStatus, VideoJob


class VideoJobRepository:
    def __init__(self) -> None:
        self._jobs: dict[str, VideoJob] = {}
        self._previews: dict[str, tuple[int, bytes]] = {}
        self._condition = Condition(RLock())

    def create(self, *, input_path: Path, output_path: Path) -> VideoJob:
        job = VideoJob(
            job_id=uuid4().hex,
            status=JobStatus.QUEUED,
            input_path=input_path,
            output_path=output_path,
            created_at=datetime.now(UTC),
        )
        with self._condition:
            self._jobs[job.job_id] = job
            self._condition.notify_all()
        return job

    def get(self, job_id: str) -> VideoJob | None:
        with self._condition:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> VideoJob:
        with self._condition:
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(job_id)
            updated = current.with_changes(**changes)
            self._jobs[job_id] = updated
            self._condition.notify_all()
            return updated

    def set_preview(self, job_id: str, frame_number: int, jpeg: bytes) -> None:
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._previews[job_id] = (frame_number, jpeg)
            self._condition.notify_all()

    def wait_for_preview(
        self,
        job_id: str,
        after_frame: int,
        *,
        timeout: float = 1.0,
    ) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    job_id not in self._jobs
                    or self._previews.get(job_id, (-1, b""))[0] > after_frame
                    or self._jobs[job_id].status
                    in {JobStatus.COMPLETED, JobStatus.FAILED}
                ),
                timeout=timeout,
            )
            preview = self._previews.get(job_id)
            if preview is not None and preview[0] > after_frame:
                return preview
            return None

    def remove(self, job_id: str) -> VideoJob | None:
        with self._condition:
            self._previews.pop(job_id, None)
            job = self._jobs.pop(job_id, None)
            self._condition.notify_all()
            return job
