from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from fastapi.testclient import TestClient
from risk_engine import RiskLevel

from app.core.config import Settings
from app.main import create_app
from app.models import JobStatus
from app.repositories import VideoJobRepository
from app.services.image_processing_service import ProcessedFrame
from app.services.video_processing_service import VideoProcessingService


class FakeFrameProcessor:
    def __init__(self, levels: tuple[RiskLevel, ...]) -> None:
        self.levels = levels
        self.calls = 0
        self.preload_calls = 0

    def preload_models(self) -> None:
        self.preload_calls += 1

    def readiness(self) -> dict[str, object]:
        return {
            "pipeline_ready": True,
            "pipeline_version": "fake-video-pipeline",
            "models_loaded": {"fake": True},
        }

    def process_video_frame(
        self,
        image_bgr: np.ndarray,
        *,
        upload_id: str,
        frame_index: int,
        timestamp: float,
    ) -> ProcessedFrame:
        del upload_id, timestamp
        self.calls += 1
        return ProcessedFrame(
            annotated_bgr=image_bgr.copy(),
            pseudo_bev_bgr=np.full((20, 20, 3), frame_index * 20, dtype=np.uint8),
            risk_level=self.levels[frame_index % len(self.levels)],
        )


class FailingFrameProcessor(FakeFrameProcessor):
    def process_video_frame(self, *args: object, **kwargs: object) -> ProcessedFrame:
        del args, kwargs
        raise RuntimeError("private model failure")


class BlockingFrameProcessor(FakeFrameProcessor):
    def __init__(self) -> None:
        super().__init__((RiskLevel.SAFE,))
        self.started = Event()
        self.release = Event()

    def process_video_frame(self, *args: object, **kwargs: object) -> ProcessedFrame:
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release processor")
        return super().process_video_frame(*args, **kwargs)  # type: ignore[arg-type]


def _client(tmp_path: Path, service: FakeFrameProcessor) -> TestClient:
    placeholder = tmp_path / "config.yaml"
    placeholder.write_text("fixture: true\n", encoding="utf-8")
    settings = Settings(
        models_config=placeholder,
        geometry_config=placeholder,
        risk_config=placeholder,
        evidence_root=tmp_path / "evidence",
        video_upload_root=tmp_path / "storage" / "uploads" / "videos",
        video_output_root=tmp_path / "storage" / "outputs" / "videos",
    )
    return TestClient(create_app(settings=settings, image_processing_service=service))


def _video_bytes(tmp_path: Path, *, frame_count: int = 3) -> bytes:
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            writer.write(np.full((24, 32, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


def _wait_for_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        payload = client.get(f"/api/v1/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("video job did not finish")


def test_video_job_processes_frames_and_exposes_result(tmp_path: Path) -> None:
    service = FakeFrameProcessor(
        (RiskLevel.SAFE, RiskLevel.WARNING, RiskLevel.DANGER)
    )
    with _client(tmp_path, service) as client:
        created = client.post(
            "/api/v1/detection/video",
            files={"file": ("clip.avi", _video_bytes(tmp_path), "video/x-msvideo")},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]
        assert created.json()["status"] == "queued"

        job = _wait_for_terminal(client, job_id)
        assert job["status"] == "completed"
        assert job["progress"] == 100.0
        assert job["current_frame"] == 3
        assert job["total_frames"] == 3
        assert job["safe_frame_count"] == 1
        assert job["warning_frame_count"] == 1
        assert job["danger_frame_count"] == 1
        assert job["max_risk_level"] == "DANGER"
        assert job["output_codec"] == "h264"
        assert job["browser_playback_compatible"] is True
        assert job["playback_warning"] is None
        assert job["summary"]["processed_frames"] == 3
        assert job["summary"]["risk_segment_count"] == 1
        assert len(job["risk_segments"]) == 1
        segment = job["risk_segments"][0]
        assert segment["start_frame"] == 1
        assert segment["end_frame"] == 3
        assert segment["max_risk_level"] == "DANGER"
        assert segment["warning_frame_count"] == 1
        assert segment["danger_frame_count"] == 1
        assert [item["frame_number"] for item in segment["frame_evidence"]] == [2, 3]
        assert [item["risk_level"] for item in segment["frame_evidence"]] == [
            "WARNING",
            "DANGER",
        ]
        assert segment["output_codec"] == "h264"
        assert segment["browser_playback_compatible"] is True
        assert service.calls == 3
        evidence = segment["frame_evidence"][1]

        preview = client.get(f"/api/v1/jobs/{job_id}/stream")
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith(
            "multipart/x-mixed-replace"
        )
        assert b"Content-Type: image/jpeg" in preview.content

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["content-type"] == "video/mp4"
        assert result.content

        download = client.get(job["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "video/mp4"
        assert "attachment" in download.headers["content-disposition"]
        assert download.content == result.content

        report = client.get(job["report_url"])
        assert report.status_code == 200
        assert report.headers["content-type"] == "application/json"
        assert "attachment" in report.headers["content-disposition"]
        report_payload = json.loads(report.content)
        assert report_payload["schema_version"] == "1.0"
        assert report_payload["job_id"] == job_id
        assert report_payload["summary"]["processed_frames"] == 3
        assert report_payload["summary"]["danger_frames"] == 1
        assert report_payload["video"]["codec"] == "h264"
        assert report_payload["video"]["url"] == job["result_url"]
        assert report_payload["video"]["download_url"] == job["download_url"]
        assert len(report_payload["risk_segments"]) == 1
        report_evidence = report_payload["risk_segments"][0]["evidence"]
        assert [item["frame_number"] for item in report_evidence] == [2, 3]
        assert report_evidence[1]["original_url"] == evidence["original_url"]

        segment_result = client.get(segment["result_url"])
        assert segment_result.status_code == 200
        assert segment_result.headers["content-type"] == "video/mp4"
        assert segment_result.content

        original = client.get(evidence["original_url"])
        rgb = client.get(evidence["rgb_url"])
        bev = client.get(evidence["pseudo_bev_url"])
        assert original.status_code == 200
        assert original.headers["content-type"] == "image/png"
        original_image = cv2.imdecode(
            np.frombuffer(original.content, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert original_image is not None
        assert np.array_equal(
            original_image,
            np.full((24, 32, 3), 60, dtype=np.uint8),
        )
        assert rgb.status_code == 200
        assert rgb.headers["content-type"] == "image/png"
        assert cv2.imdecode(np.frombuffer(rgb.content, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
        assert bev.status_code == 200
        assert bev.headers["content-type"] == "image/png"
        assert cv2.imdecode(np.frombuffer(bev.content, dtype=np.uint8), cv2.IMREAD_COLOR) is not None


def test_blocking_inference_does_not_block_status_or_health_api(tmp_path: Path) -> None:
    service = BlockingFrameProcessor()
    with _client(tmp_path, service) as client:
        created = client.post(
            "/api/v1/detection/video",
            files={"file": ("clip.avi", _video_bytes(tmp_path), "video/x-msvideo")},
        )
        assert service.started.wait(timeout=1.0)
        job_id = created.json()["job_id"]

        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "processing"
        assert client.get("/api/v1/health").status_code == 200

        service.release.set()
        assert _wait_for_terminal(client, job_id)["status"] == "completed"


def test_invalid_video_becomes_failed_job(tmp_path: Path) -> None:
    with _client(tmp_path, FakeFrameProcessor((RiskLevel.SAFE,))) as client:
        response = client.post(
            "/api/v1/detection/video",
            files={"file": ("broken.mp4", b"not-video", "video/mp4")},
        )
        assert response.status_code == 202
        job = _wait_for_terminal(client, response.json()["job_id"])

        assert job["status"] == "failed"
        assert job["error"] == "uploaded video could not be opened"
        assert client.get(response.json()["result_url"]).status_code == 409
        assert client.get(f"/api/v1/jobs/{job['job_id']}/report").status_code == 409


def test_ai_pipeline_failure_is_sanitized_and_marks_job_failed(tmp_path: Path) -> None:
    with _client(tmp_path, FailingFrameProcessor((RiskLevel.SAFE,))) as client:
        response = client.post(
            "/api/v1/detection/video",
            files={"file": ("clip.avi", _video_bytes(tmp_path), "video/x-msvideo")},
        )
        job = _wait_for_terminal(client, response.json()["job_id"])

        assert job["status"] == "failed"
        assert job["error"] == "video processing failed"
        assert "private model failure" not in str(job)


def test_unknown_video_job_endpoints_return_404(tmp_path: Path) -> None:
    with _client(tmp_path, FakeFrameProcessor((RiskLevel.SAFE,))) as client:
        assert client.get("/api/v1/jobs/missing").status_code == 404
        assert client.get("/api/v1/jobs/missing/stream").status_code == 404
        assert client.get("/api/v1/jobs/missing/result").status_code == 404
        assert client.get("/api/v1/jobs/missing/download").status_code == 404
        assert client.get("/api/v1/jobs/missing/report").status_code == 404


def test_repository_cleanup_removes_job_and_preview(tmp_path: Path) -> None:
    repository = VideoJobRepository()
    job = repository.create(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
    )
    repository.set_preview(job.job_id, 1, b"jpeg")

    removed = repository.remove(job.job_id)

    assert removed == job
    assert repository.get(job.job_id) is None
    assert repository.wait_for_preview(job.job_id, 0, timeout=0.0) is None


def test_processing_transitions_and_releases_video_files(tmp_path: Path) -> None:
    class RecordingRepository(VideoJobRepository):
        def __init__(self) -> None:
            super().__init__()
            self.statuses: list[JobStatus] = []

        def update(self, job_id: str, **changes: object):  # type: ignore[no-untyped-def]
            if isinstance(changes.get("status"), JobStatus):
                self.statuses.append(changes["status"])  # type: ignore[arg-type]
            return super().update(job_id, **changes)

    input_path = tmp_path / "fixture.avi"
    input_path.write_bytes(_video_bytes(tmp_path, frame_count=2))
    output_path = tmp_path / "result.mp4"
    repository = RecordingRepository()
    job = repository.create(input_path=input_path, output_path=output_path)
    service = VideoProcessingService(
        frame_processor=FakeFrameProcessor((RiskLevel.SAFE, RiskLevel.DANGER)),
        repository=repository,
    )

    service.process(job.job_id)

    assert repository.statuses == [JobStatus.PROCESSING, JobStatus.COMPLETED]
    completed = repository.get(job.job_id)
    assert completed is not None
    assert completed.progress == 100.0
    assert len(completed.risk_segments) == 1
    # On Windows these deletes also prove capture and writer no longer hold handles.
    input_path.unlink()
    output_path.unlink()
    segment_path = completed.risk_segments[0].output_path
    evidence_paths = [
        path
        for item in completed.risk_segments[0].frame_evidence
        for path in (item.original_path, item.rgb_path, item.pseudo_bev_path)
    ]
    segment_path.unlink()
    for evidence_path in evidence_paths:
        evidence_path.unlink()
    evidence_paths[0].parent.rmdir()
    segment_path.parent.rmdir()
    assert not input_path.exists()
    assert not output_path.exists()
    assert not segment_path.exists()
    assert all(not path.exists() for path in evidence_paths)


def test_post_roll_closes_and_separates_frame_risk_segments(tmp_path: Path) -> None:
    input_path = tmp_path / "five-frames.avi"
    input_path.write_bytes(_video_bytes(tmp_path, frame_count=5))
    repository = VideoJobRepository()
    job = repository.create(
        input_path=input_path,
        output_path=tmp_path / "five-frames-result.mp4",
    )
    service = VideoProcessingService(
        frame_processor=FakeFrameProcessor(
            (
                RiskLevel.SAFE,
                RiskLevel.WARNING,
                RiskLevel.SAFE,
                RiskLevel.SAFE,
                RiskLevel.DANGER,
            )
        ),
        repository=repository,
        segment_pre_roll_seconds=0.1,
        segment_post_roll_seconds=0.1,
    )

    service.process(job.job_id)

    completed = repository.get(job.job_id)
    assert completed is not None
    assert [segment.max_risk_level for segment in completed.risk_segments] == [
        "WARNING",
        "DANGER",
    ]
    assert [
        (segment.start_frame, segment.end_frame)
        for segment in completed.risk_segments
    ] == [(1, 3), (4, 5)]
    assert [
        [item.frame_number for item in segment.frame_evidence]
        for segment in completed.risk_segments
    ] == [[2], [5]]


def test_segment_evidence_selects_first_peak_and_last_risk_frames(tmp_path: Path) -> None:
    input_path = tmp_path / "evidence-selection.avi"
    input_path.write_bytes(_video_bytes(tmp_path, frame_count=5))
    repository = VideoJobRepository()
    job = repository.create(
        input_path=input_path,
        output_path=tmp_path / "evidence-selection.mp4",
    )
    service = VideoProcessingService(
        frame_processor=FakeFrameProcessor(
            (
                RiskLevel.WARNING,
                RiskLevel.WARNING,
                RiskLevel.DANGER,
                RiskLevel.DANGER,
                RiskLevel.WARNING,
            )
        ),
        repository=repository,
    )

    service.process(job.job_id)

    completed = repository.get(job.job_id)
    assert completed is not None
    assert len(completed.risk_segments) == 1
    evidence = completed.risk_segments[0].frame_evidence
    assert [item.frame_number for item in evidence] == [1, 3, 5]
    assert [item.risk_level for item in evidence] == [
        "WARNING",
        "DANGER",
        "WARNING",
    ]
