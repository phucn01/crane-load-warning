"""Blocking frame-by-frame video processing over the existing safety pipeline."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.logging import log_operation
from app.models import (
    FrameEvidence,
    FrameRiskResult,
    JobStatus,
    RiskSegment,
    RiskSnapshotRecord,
)
from app.repositories import VideoJobRepository
from app.schemas.detection import RiskLevelValue
from app.services.image_processing_service import ProcessedFrame
from app.services.processing_history_service import ProcessingHistoryService
from app.services.video_report_service import write_video_report
from app.services.video_transcoding_service import (
    BrowserVideoConverter,
    VideoCompatibility,
)

LOGGER = logging.getLogger(__name__)
RISK_SEVERITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2}


@dataclass(slots=True)
class _EvidenceCandidate:
    frame_number: int
    timestamp_seconds: float
    risk_level: RiskLevelValue
    original: NDArray[np.uint8]
    annotated: NDArray[np.uint8]
    pseudo_bev: NDArray[np.uint8]


@dataclass(slots=True)
class _ActiveRiskSegment:
    segment_id: str
    output_path: Path
    writer: cv2.VideoWriter
    start_frame: int
    risk_start_frame: int
    risk_end_frame: int
    last_written_frame: int
    max_risk_level: RiskLevelValue
    warning_frame_count: int = 0
    danger_frame_count: int = 0
    safe_tail_frames: int = 0
    evidence: list[_EvidenceCandidate] = field(default_factory=list)


class _RiskSegmentRecorder:
    """Write padded risk clips from already-annotated frames without inference."""

    def __init__(
        self,
        *,
        output_root: Path,
        fps: float,
        frame_size: tuple[int, int],
        fourcc: str,
        pre_roll_seconds: float,
        post_roll_seconds: float,
    ) -> None:
        self.output_root = output_root
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
        self.pre_roll_frames = max(0, round(pre_roll_seconds * fps))
        self.post_roll_frames = max(0, round(post_roll_seconds * fps))
        self._pre_roll: deque[tuple[int, bytes]] = deque(
            maxlen=self.pre_roll_frames
        )
        self._active: _ActiveRiskSegment | None = None

    def accept(
        self,
        *,
        frame_number: int,
        original: NDArray[np.uint8],
        annotated: NDArray[np.uint8],
        pseudo_bev: NDArray[np.uint8] | None,
        preview_jpeg: bytes,
        risk_level: RiskLevelValue | None,
    ) -> RiskSegment | None:
        finalized = None
        if self._active is None and risk_level not in (None, "SAFE"):
            self._start(
                frame_number,
                original,
                annotated,
                _required_pseudo_bev(pseudo_bev),
                risk_level,
            )
        elif self._active is not None:
            if risk_level == "SAFE" and self.post_roll_frames == 0:
                finalized = self._finalize()
            else:
                self._active.writer.write(annotated)
                self._active.last_written_frame = frame_number
                if risk_level == "SAFE":
                    self._active.safe_tail_frames += 1
                    if self._active.safe_tail_frames >= self.post_roll_frames:
                        finalized = self._finalize()
                elif risk_level is not None:
                    self._record_risk_frame(frame_number, risk_level)
                    self._record_evidence(
                        frame_number,
                        original,
                        annotated,
                        _required_pseudo_bev(pseudo_bev),
                        risk_level,
                    )

        self._pre_roll.append((frame_number, preview_jpeg))
        return finalized

    def finish(self) -> RiskSegment | None:
        return self._finalize() if self._active is not None else None

    def abort(self) -> None:
        if self._active is None:
            return
        active = self._active
        self._active = None
        active.writer.release()
        active.output_path.unlink(missing_ok=True)

    def _start(
        self,
        frame_number: int,
        original: NDArray[np.uint8],
        annotated: NDArray[np.uint8],
        pseudo_bev: NDArray[np.uint8],
        risk_level: RiskLevelValue,
    ) -> None:
        segment_id = uuid4().hex
        LOGGER.info(
            "=== START | OPERATION=RISK_SEGMENT | SEGMENT_ID=%s | "
            "START_FRAME=%s | RISK_LEVEL=%s ===",
            segment_id,
            frame_number,
            risk_level,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        output_path = self.output_root / f"{segment_id}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            self.frame_size,
        )
        if not writer.isOpened():
            writer.release()
            raise OSError("risk segment video writer could not be opened")

        start_frame = frame_number
        try:
            for buffered_frame_number, jpeg in self._pre_roll:
                buffered = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if buffered is None:
                    raise OSError("could not decode an in-memory pre-roll frame")
                writer.write(buffered)
                start_frame = min(start_frame, buffered_frame_number)
            writer.write(annotated)
        except Exception:
            writer.release()
            output_path.unlink(missing_ok=True)
            raise

        self._active = _ActiveRiskSegment(
            segment_id=segment_id,
            output_path=output_path,
            writer=writer,
            start_frame=start_frame,
            risk_start_frame=frame_number,
            risk_end_frame=frame_number,
            last_written_frame=frame_number,
            max_risk_level=risk_level,
        )
        self._record_risk_frame(frame_number, risk_level)
        self._record_evidence(
            frame_number,
            original,
            annotated,
            pseudo_bev,
            risk_level,
        )

    def _record_risk_frame(
        self,
        frame_number: int,
        risk_level: RiskLevelValue,
    ) -> None:
        assert self._active is not None
        self._active.risk_end_frame = frame_number
        self._active.safe_tail_frames = 0
        if risk_level == "WARNING":
            self._active.warning_frame_count += 1
        elif risk_level == "DANGER":
            self._active.danger_frame_count += 1
        if RISK_SEVERITY[risk_level] > RISK_SEVERITY[
            self._active.max_risk_level
        ]:
            self._active.max_risk_level = risk_level

    def _record_evidence(
        self,
        frame_number: int,
        original: NDArray[np.uint8],
        annotated: NDArray[np.uint8],
        pseudo_bev: NDArray[np.uint8],
        risk_level: RiskLevelValue,
    ) -> None:
        assert self._active is not None
        candidate = _EvidenceCandidate(
            frame_number=frame_number,
            timestamp_seconds=(frame_number - 1) / self.fps,
            risk_level=risk_level,
            original=original.copy(),
            annotated=annotated.copy(),
            pseudo_bev=pseudo_bev.copy(),
        )
        self._active.evidence.append(candidate)

    def _finalize(self) -> RiskSegment:
        assert self._active is not None
        active = self._active
        active.writer.release()
        try:
            frame_evidence = self._persist_evidence(active)
            segment = RiskSegment(
                segment_id=active.segment_id,
                output_path=active.output_path,
                start_frame=active.start_frame,
                end_frame=active.last_written_frame,
                risk_start_frame=active.risk_start_frame,
                risk_end_frame=active.risk_end_frame,
                start_seconds=(active.start_frame - 1) / self.fps,
                end_seconds=active.last_written_frame / self.fps,
                max_risk_level=active.max_risk_level,
                warning_frame_count=active.warning_frame_count,
                danger_frame_count=active.danger_frame_count,
                frame_evidence=frame_evidence,
            )
        except Exception:
            active.output_path.unlink(missing_ok=True)
            raise
        self._active = None
        LOGGER.info(
            "=== END | OPERATION=RISK_SEGMENT | SEGMENT_ID=%s | "
            "START_FRAME=%s | END_FRAME=%s | MAX_RISK_LEVEL=%s ===",
            segment.segment_id,
            segment.start_frame,
            segment.end_frame,
            segment.max_risk_level,
        )
        return segment

    def _persist_evidence(
        self,
        active: _ActiveRiskSegment,
    ) -> tuple[FrameEvidence, ...]:
        # Keep a compact, deterministic representative set without storing
        # first/peak/last state on the active segment itself.
        candidates = active.evidence
        selected: list[_EvidenceCandidate] = []
        if candidates:
            selected.append(candidates[0])
            peak = max(candidates, key=lambda item: RISK_SEVERITY[item.risk_level])
            if peak.frame_number != selected[0].frame_number:
                selected.append(peak)
            if candidates[-1].frame_number not in {
                item.frame_number for item in selected
            }:
                selected.append(candidates[-1])
        candidates_by_frame = {candidate.frame_number: candidate for candidate in selected}
        evidence_dir = self.output_root / f"{active.segment_id}_evidence"
        evidence_dir.mkdir(parents=True, exist_ok=False)
        written_paths: list[Path] = []
        evidence: list[FrameEvidence] = []
        try:
            for frame_number in sorted(candidates_by_frame):
                candidate = candidates_by_frame[frame_number]
                original_path = evidence_dir / f"frame-{frame_number}-original.png"
                rgb_path = evidence_dir / f"frame-{frame_number}-rgb.png"
                pseudo_bev_path = evidence_dir / f"frame-{frame_number}-bev.png"
                _write_png(original_path, candidate.original)
                written_paths.append(original_path)
                _write_png(rgb_path, candidate.annotated)
                written_paths.append(rgb_path)
                _write_png(pseudo_bev_path, candidate.pseudo_bev)
                written_paths.append(pseudo_bev_path)
                evidence.append(
                    FrameEvidence(
                        frame_number=frame_number,
                        timestamp_seconds=candidate.timestamp_seconds,
                        risk_level=candidate.risk_level,
                        original_path=original_path,
                        rgb_path=rgb_path,
                        pseudo_bev_path=pseudo_bev_path,
                    )
                )
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
            evidence_dir.rmdir()
            raise
        return tuple(evidence)


class FrameProcessor(Protocol):
    def process_video_frame(
        self,
        image_bgr: NDArray[np.uint8],
        *,
        upload_id: str,
        frame_index: int,
        timestamp: float,
    ) -> ProcessedFrame: ...


class VideoProcessingService:
    """Read, assess, annotate, preview, and write every source frame once."""

    def __init__(
        self,
        *,
        frame_processor: FrameProcessor,
        repository: VideoJobRepository,
        output_fourcc: str = "mp4v",
        segment_pre_roll_seconds: float = 2.0,
        segment_post_roll_seconds: float = 2.0,
        video_converter: BrowserVideoConverter | None = None,
        history_service: ProcessingHistoryService | None = None,
        snapshot_evidence_root: Path | None = None,
    ) -> None:
        if len(output_fourcc) != 4:
            raise ValueError("output_fourcc must contain exactly four characters")
        self.frame_processor = frame_processor
        self.repository = repository
        self.output_fourcc = output_fourcc
        if (
            not math.isfinite(segment_pre_roll_seconds)
            or not math.isfinite(segment_post_roll_seconds)
            or segment_pre_roll_seconds < 0.0
            or segment_post_roll_seconds < 0.0
        ):
            raise ValueError("risk segment roll durations must be non-negative")
        self.segment_pre_roll_seconds = segment_pre_roll_seconds
        self.segment_post_roll_seconds = segment_post_roll_seconds
        self.video_converter = video_converter or BrowserVideoConverter.discover()
        self.history_service = history_service
        self.snapshot_evidence_root = snapshot_evidence_root

    def process(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None:
            LOGGER.warning("=== WARNING | VIDEO_JOB_NOT_FOUND | JOB_ID=%s ===", job_id)
            return

        capture: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | None = None
        segment_recorder: _RiskSegmentRecorder | None = None
        started_clock = perf_counter()
        LOGGER.info("=== START | OPERATION=VIDEO_JOB_PROCESSING | JOB_ID=%s ===", job_id)
        try:
            with log_operation(LOGGER, "open_source_video", job_id=job_id):
                capture = cv2.VideoCapture(str(job.input_path))
                if not capture.isOpened():
                    raise ValueError("uploaded video could not be opened")

                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                source_fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(source_fps) or source_fps <= 0.0:
                    source_fps = 25.0
                if width <= 0 or height <= 0:
                    raise ValueError("uploaded video has invalid frame dimensions")

            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            with log_operation(LOGGER, "open_output_video", job_id=job_id):
                writer = cv2.VideoWriter(
                    str(job.output_path),
                    cv2.VideoWriter_fourcc(*self.output_fourcc),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(
                        f"output video writer could not open codec {self.output_fourcc}"
                    )
            LOGGER.info(
                "=== EVENT | VIDEO_METADATA | JOB_ID=%s | WIDTH=%s | HEIGHT=%s | "
                "SOURCE_FPS=%.3f | TOTAL_FRAMES=%s ===",
                job_id,
                width,
                height,
                source_fps,
                total_frames,
            )

            segment_recorder = _RiskSegmentRecorder(
                output_root=(
                    job.output_path.parent / f"{job.output_path.stem}_segments"
                ),
                fps=source_fps,
                frame_size=(width, height),
                fourcc=self.output_fourcc,
                pre_roll_seconds=self.segment_pre_roll_seconds,
                post_roll_seconds=self.segment_post_roll_seconds,
            )

            self.repository.update(
                job_id,
                status=JobStatus.PROCESSING,
                total_frames=total_frames,
                started_at=datetime.now(UTC),
            )
            if self.history_service is not None:
                self.history_service.mark_processing(
                    job_id,
                    total_frames=total_frames,
                )
            counts = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
            max_risk: str | None = None
            processed = 0
            pending_snapshots: list[RiskSnapshotRecord] = []

            while True:
                frame_started = perf_counter()
                readable, frame = capture.read()
                if not readable:
                    break
                processed += 1
                timestamp_seconds = (processed - 1) / source_fps
                LOGGER.info(
                    "=== EVENT | VIDEO_FRAME_POSITION | JOB_ID=%s | FRAME=%s | "
                    "TIMESTAMP_SECONDS=%.3f | SOURCE_FPS=%.3f ===",
                    job_id,
                    processed,
                    timestamp_seconds,
                    source_fps,
                )
                original = frame.copy()
                assessed = self.frame_processor.process_video_frame(
                    frame,
                    upload_id=job_id,
                    frame_index=processed - 1,
                    timestamp=timestamp_seconds,
                )
                annotated = _source_sized_frame(assessed.annotated_bgr, width, height)
                writer.write(annotated)
                preview = _encode_preview(annotated)
                self.repository.set_preview(job_id, processed, preview)

                risk = (
                    None
                    if assessed.risk_level is None
                    else assessed.risk_level.value
                )
                self._persist_risk_snapshot(
                    job_id=job_id,
                    frame_index=processed - 1,
                    timestamp_seconds=timestamp_seconds,
                    risk_level=risk,
                    original=original,
                    assessed=assessed,
                    pending=pending_snapshots,
                )
                self.repository.append_frame_result(
                    job_id,
                FrameRiskResult(
                        frame_number=processed,
                        timestamp_seconds=timestamp_seconds,
                        risk_level=risk,
                        assessment_status=assessed.assessment_status,
                    ),
                )
                segment = segment_recorder.accept(
                    frame_number=processed,
                    original=original,
                    annotated=annotated,
                    pseudo_bev=assessed.pseudo_bev_bgr,
                    preview_jpeg=preview,
                    risk_level=risk,
                )
                if segment is not None:
                    self._append_segment(job_id, segment)
                if risk is not None:
                    counts[risk] += 1
                if risk is not None and (
                    max_risk is None or RISK_SEVERITY[risk] > RISK_SEVERITY[max_risk]
                ):
                    max_risk = risk
                elapsed = max(perf_counter() - started_clock, 1e-9)
                progress = (
                    min(99.9, processed * 100.0 / total_frames)
                    if total_frames > 0
                    else 0.0
                )
                updated_job = self.repository.update(
                    job_id,
                    current_frame=processed,
                    progress=round(progress, 2),
                    processing_fps=processed / elapsed,
                    elapsed_seconds=elapsed,
                    current_risk_level=risk,
                    max_risk_level=max_risk,
                    safe_frame_count=counts["SAFE"],
                    warning_frame_count=counts["WARNING"],
                    danger_frame_count=counts["DANGER"],
                )
                if self.history_service is not None:
                    self.history_service.persist_video_progress(updated_job)
                LOGGER.info(
                    "=== END | OPERATION=VIDEO_FRAME | JOB_ID=%s | FRAME=%s | "
                    "TOTAL_FRAMES=%s | DURATION_MS=%.3f | RISK_LEVEL=%s | "
                    "PROGRESS=%.2f ===",
                    job_id,
                    processed,
                    total_frames,
                    (perf_counter() - frame_started) * 1000.0,
                    risk,
                    progress,
                )

                if len(pending_snapshots) >= 32:
                    self._flush_snapshot_batch(pending_snapshots)
                    pending_snapshots.clear()

            if processed == 0:
                raise ValueError("uploaded video contains no readable frames")
            self._flush_snapshot_batch(pending_snapshots)
            pending_snapshots.clear()
            final_segment = segment_recorder.finish()
            if final_segment is not None:
                self._append_segment(job_id, final_segment)
            # Finalize the MP4 container before the completed result becomes visible.
            writer.release()
            writer = None
            with log_operation(LOGGER, "transcode_full_video", job_id=job_id):
                full_compatibility = self.video_converter.convert(job.output_path)
            with log_operation(LOGGER, "transcode_risk_segments", job_id=job_id):
                self._finalize_segment_codecs(job_id)
            elapsed = max(perf_counter() - started_clock, 1e-9)
            completion_changes = {
                "status": JobStatus.COMPLETED,
                "current_frame": processed,
                "total_frames": total_frames or processed,
                "progress": 100.0,
                "processing_fps": processed / elapsed,
                "elapsed_seconds": elapsed,
                "output_codec": full_compatibility.codec,
                "browser_playback_compatible": (
                    full_compatibility.browser_playback_compatible
                ),
                "playback_warning": full_compatibility.warning,
                "completed_at": datetime.now(UTC),
            }
            current = self.repository.get(job_id)
            if current is None:
                raise KeyError(job_id)
            completed_job = current.with_changes(**completion_changes)
            with log_operation(LOGGER, "write_video_report", job_id=job_id):
                write_video_report(completed_job)
            completed_job = self.repository.update(job_id, **completion_changes)
            if self.history_service is not None:
                self.history_service.complete_video(completed_job)
            LOGGER.info(
                "=== END | OPERATION=VIDEO_JOB_PROCESSING | JOB_ID=%s | "
                "DURATION_MS=%.3f | PROCESSED_FRAMES=%s | AVERAGE_FPS=%.3f | "
                "MAX_RISK_LEVEL=%s ===",
                job_id,
                elapsed * 1000.0,
                processed,
                processed / elapsed,
                max_risk,
            )
        except Exception as error:
            LOGGER.exception(
                "=== ERROR | OPERATION=VIDEO_JOB_PROCESSING | JOB_ID=%s | "
                "ERROR_TYPE=%s ===",
                job_id,
                type(error).__name__,
            )
            elapsed = max(perf_counter() - started_clock, 0.0)
            self.repository.update(
                job_id,
                status=JobStatus.FAILED,
                elapsed_seconds=elapsed,
                error=_public_error(error),
                completed_at=datetime.now(UTC),
            )
            if self.history_service is not None:
                self.history_service.fail_job(job_id, _public_error(error))
        finally:
            if segment_recorder is not None:
                segment_recorder.abort()
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()

    def _append_segment(self, job_id: str, segment: RiskSegment) -> None:
        job = self.repository.get(job_id)
        if job is None:
            raise KeyError(job_id)
        self.repository.update(
            job_id,
            risk_segments=(*job.risk_segments, segment),
        )

    def _persist_risk_snapshot(
        self,
        *,
        job_id: str,
        frame_index: int,
        timestamp_seconds: float,
        risk_level: RiskLevelValue | None,
        original: NDArray[np.uint8],
        assessed: ProcessedFrame,
        pending: list[RiskSnapshotRecord] | None = None,
    ) -> None:
        history = self.history_service
        root = self.snapshot_evidence_root
        if history is None:
            return
        snapshot_id = uuid4().hex
        # Every processed video frame gets image evidence. The database record
        # and its files therefore stay in sync; cooldown sampling is no longer
        # used to decide whether a frame image is written.
        capture_evidence = root is not None
        relative_root = Path(job_id) / "snapshots" / snapshot_id
        output_root = None if root is None else root / relative_root
        evidence_paths: dict[str, str | None] = {
            "evidence_path": None,
            "rgb_evidence_path": None,
            "pseudo_bev_path": None,
        }
        try:
            if capture_evidence and output_root is not None:
                output_root.mkdir(parents=True, exist_ok=False)
                original_path = output_root / "original.png"
                rgb_path = output_root / "rgb.png"
                _write_png(original_path, original)
                _write_png(rgb_path, assessed.annotated_bgr)
                url_root = "/evidence/" + relative_root.as_posix()
                evidence_paths["evidence_path"] = f"{url_root}/{original_path.name}"
                evidence_paths["rgb_evidence_path"] = f"{url_root}/{rgb_path.name}"
                if assessed.pseudo_bev_bgr is not None:
                    pseudo_bev_path = output_root / "pseudo_bev.png"
                    _write_png(pseudo_bev_path, assessed.pseudo_bev_bgr)
                    evidence_paths["pseudo_bev_path"] = f"{url_root}/{pseudo_bev_path.name}"
            record = RiskSnapshotRecord(
                    snapshot_id=snapshot_id,
                    job_id=job_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_seconds,
                    risk_level=risk_level,  # type: ignore[arg-type]
                    confidence=assessed.confidence,
                    assessment_reliable=assessed.assessment_reliable,
                    quality_reasons=assessed.quality_reasons,
                    assessment_status=assessed.assessment_status,
                    evidence_path=evidence_paths["evidence_path"],
                    rgb_evidence_path=evidence_paths["rgb_evidence_path"],
                    pseudo_bev_path=evidence_paths["pseudo_bev_path"],
                    created_at=datetime.now(UTC),
                )
            if pending is not None:
                pending.append(record)
            else:
                history.persist_snapshot(record)
        except Exception as error:
            LOGGER.exception(
                "=== ERROR | OPERATION=PERSIST_SNAPSHOT_EVIDENCE | "
                "JOB_ID=%s | FRAME=%s | ERROR_TYPE=%s ===",
                job_id,
                frame_index,
                type(error).__name__,
            )

    def _flush_snapshot_batch(self, records: list[RiskSnapshotRecord]) -> None:
        if self.history_service is None or not records:
            return
        self.history_service.persist_snapshots(records)

    def _persist_safe_no_load_frame(
        self,
        *,
        job_id: str,
        frame_index: int,
        timestamp_seconds: float,
        original: NDArray[np.uint8],
        assessed: ProcessedFrame,
    ) -> None:
        """Save person-only SAFE evidence as a normal history record."""
        history = self.history_service
        root = self.snapshot_evidence_root
        if history is None or root is None:
            return
        snapshot_id = uuid4().hex
        relative_root = Path(job_id) / "safe_no_load" / snapshot_id
        output_root = root / relative_root
        try:
            output_root.mkdir(parents=True, exist_ok=False)
            original_path = output_root / "original.png"
            rgb_path = output_root / "rgb.png"
            _write_png(original_path, original)
            _write_png(rgb_path, assessed.annotated_bgr)
            url_root = "/evidence/" + relative_root.as_posix()
            history.persist_snapshot(
                RiskSnapshotRecord(
                    snapshot_id=snapshot_id,
                    job_id=job_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_seconds,
                    risk_level="SAFE",
                    assessment_status="SAFE_NO_LOAD",
                    confidence=assessed.confidence,
                    assessment_reliable=assessed.assessment_reliable,
                    quality_reasons=assessed.quality_reasons,
                    evidence_path=f"{url_root}/{original_path.name}",
                    rgb_evidence_path=f"{url_root}/{rgb_path.name}",
                    pseudo_bev_path=None,
                    created_at=datetime.now(UTC),
                )
            )
        except Exception as error:
            LOGGER.exception(
                "=== ERROR | OPERATION=PERSIST_SAFE_NO_LOAD | JOB_ID=%s | "
                "FRAME=%s | ERROR_TYPE=%s ===",
                job_id,
                frame_index,
                type(error).__name__,
            )

    def _finalize_segment_codecs(self, job_id: str) -> None:
        job = self.repository.get(job_id)
        if job is None:
            raise KeyError(job_id)
        finalized: list[RiskSegment] = []
        for segment in job.risk_segments:
            compatibility = self.video_converter.convert(segment.output_path)
            finalized.append(_segment_with_compatibility(segment, compatibility))
        self.repository.update(job_id, risk_segments=tuple(finalized))


def _source_sized_frame(
    frame: NDArray[np.uint8], width: int, height: int
) -> NDArray[np.uint8]:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _encode_preview(frame: NDArray[np.uint8]) -> bytes:
    encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not encoded:
        raise OSError("could not encode processing preview")
    return payload.tobytes()


def _write_png(path: Path, frame: NDArray[np.uint8]) -> None:
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write frame evidence {path.name}")


def _required_pseudo_bev(
    frame: NDArray[np.uint8] | None,
) -> NDArray[np.uint8]:
    if frame is None:
        raise OSError("non-safe frame did not produce Pseudo-BEV evidence")
    return frame


def _public_error(error: Exception) -> str:
    if isinstance(error, (ValueError, OSError)):
        return str(error)
    return "video processing failed"


def _segment_with_compatibility(
    segment: RiskSegment,
    compatibility: VideoCompatibility,
) -> RiskSegment:
    return replace(
        segment,
        output_codec=compatibility.codec,
        browser_playback_compatible=compatibility.browser_playback_compatible,
        playback_warning=compatibility.warning,
    )


__all__ = ["FrameProcessor", "VideoProcessingService"]
