"""Thread-safe execution timeline shared by frame pipelines."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Any

TIMEZONE_NAME = "Asia/Bangkok"
LOCAL_TIMEZONE = timezone(timedelta(hours=7), name=TIMEZONE_NAME)
LOGGER = logging.getLogger(__name__)


@contextmanager
def log_pipeline_operation(
    component: str,
    operation: str,
    *,
    frame_id: str,
    entity_id: str | None = None,
) -> Iterator[None]:
    """Emit duration logs without retaining records in memory."""

    entity = "" if entity_id is None else f"\n    ENTITY_ID    : {entity_id}"
    LOGGER.info(
        "=== START ===\n    COMPONENT    : %s\n    OPERATION    : %s\n"
        "    FRAME_ID     : %s%s",
        component.upper(),
        operation.upper(),
        frame_id,
        entity,
    )
    started_ns = perf_counter_ns()
    try:
        yield
    except BaseException as error:
        LOGGER.exception(
            "=== ERROR ===\n    COMPONENT    : %s\n    OPERATION    : %s\n"
            "    FRAME_ID     : %s%s\n    DURATION_MS  : %.3f\n"
            "    ERROR_TYPE   : %s",
            component.upper(),
            operation.upper(),
            frame_id,
            entity,
            (perf_counter_ns() - started_ns) / 1_000_000.0,
            type(error).__name__,
        )
        raise
    else:
        LOGGER.info(
            "=== END ===\n    COMPONENT    : %s\n    OPERATION    : %s\n"
            "    FRAME_ID     : %s%s\n    DURATION_MS  : %.3f",
            component.upper(),
            operation.upper(),
            frame_id,
            entity,
            (perf_counter_ns() - started_ns) / 1_000_000.0,
        )


class TimelineStatus(StrEnum):
    """Current or terminal state of one timed operation."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TimelineRecord:
    """One observable pipeline operation for a single frame."""

    record_id: int
    component: str
    operation: str
    frame_id: str
    started_at: str
    status: TimelineStatus
    completed_at: str | None = None
    duration_ms: float | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class PipelineTimeline:
    """Collect start, completion, duration, and failure telemetry."""

    def __init__(self) -> None:
        self._records: list[TimelineRecord] = []
        self._next_record_id = 1
        self._lock = Lock()

    @contextmanager
    def track(
        self,
        component: str,
        operation: str,
        *,
        frame_id: str,
    ) -> Iterator[None]:
        """Track one operation and preserve its failure for later inspection."""

        if not component:
            raise ValueError("component must not be empty")
        if not operation:
            raise ValueError("operation must not be empty")
        started_at = _local_now()
        started_ns = perf_counter_ns()
        with self._lock:
            record_id = self._next_record_id
            self._next_record_id += 1
            record_index = len(self._records)
            self._records.append(
                TimelineRecord(
                    record_id=record_id,
                    component=component,
                    operation=operation,
                    frame_id=frame_id,
                    started_at=started_at,
                    status=TimelineStatus.RUNNING,
                )
            )

        try:
            yield
        except BaseException as error:
            self._finish(
                record_index,
                started_ns=started_ns,
                status=TimelineStatus.FAILED,
                error_type=type(error).__name__,
            )
            raise
        else:
            self._finish(
                record_index,
                started_ns=started_ns,
                status=TimelineStatus.COMPLETED,
            )

    def snapshot(self, *, frame_id: str | None = None) -> tuple[TimelineRecord, ...]:
        """Return an immutable snapshot, optionally filtered by frame."""

        with self._lock:
            records = tuple(self._records)
        if frame_id is None:
            return records
        return tuple(record for record in records if record.frame_id == frame_id)

    def to_dict(self) -> dict[str, Any]:
        records = self.snapshot()
        return {
            "schema_version": "1.0",
            "timezone": TIMEZONE_NAME,
            "records": [record.to_dict() for record in records],
        }

    def write_json(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist the current timeline snapshot as JSON."""

        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"timeline artifact already exists: {output_path}")
        payload = json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        return output_path

    def clear(self) -> None:
        """Remove all records and restart record numbering."""

        with self._lock:
            if any(
                record.status is TimelineStatus.RUNNING for record in self._records
            ):
                raise RuntimeError("cannot clear timeline while an operation is running")
            self._records.clear()
            self._next_record_id = 1

    def _finish(
        self,
        record_index: int,
        *,
        started_ns: int,
        status: TimelineStatus,
        error_type: str | None = None,
    ) -> None:
        duration_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        with self._lock:
            self._records[record_index] = replace(
                self._records[record_index],
                completed_at=_local_now(),
                duration_ms=round(duration_ms, 3),
                status=status,
                error_type=error_type,
            )


def _local_now() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")


__all__ = [
    "PipelineTimeline",
    "TimelineRecord",
    "TimelineStatus",
    "log_pipeline_operation",
]
