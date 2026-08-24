from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.detection import ImageDetectionResponse


class FakeImageProcessingService:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def readiness(self) -> dict[str, object]:
        return {
            "pipeline_ready": True,
            "pipeline_version": "test-pipeline",
            "models_loaded": {"fake": True},
        }

    def process(self, image_bgr: np.ndarray) -> ImageDetectionResponse:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return ImageDetectionResponse.model_validate(
            {
                "status": "completed",
                "processing_time_ms": 12.5,
                "assessment": {
                    "risk_level": "WARNING",
                    "assessment_reliable": False,
                    "quality_reasons": ["fixture"],
                    "contributing_person_ids": [],
                    "contributing_load_ids": [],
                    "pairs": [],
                },
                "summary": {
                    "person_count": 0,
                    "load_count": 0,
                    "rope_count": 0,
                },
                "detections": [],
                "geometry": {
                    "coordinate_system": "relative_pseudo_bev_not_metric",
                    "depth_low": 0.0,
                    "depth_high": 1.0,
                    "quality_reasons": [],
                    "persons": [],
                    "loads": [],
                },
                "evidence": {
                    "rgb_url": "/evidence/test/rgb.png",
                    "pseudo_bev_url": "/evidence/test/pseudo_bev.png",
                    "combined_url": None,
                },
                "metadata": {
                    "pipeline_version": "test-pipeline",
                    "frame_id": "test-frame",
                    "image_width": int(image_bgr.shape[1]),
                    "image_height": int(image_bgr.shape[0]),
                    "depth": {
                        "height": int(image_bgr.shape[0]),
                        "width": int(image_bgr.shape[1]),
                        "dtype": "float32",
                        "finite_min": 0.0,
                        "finite_max": 1.0,
                        "finite_fraction": 1.0,
                        "convention": "relative_depth_not_metric",
                    },
                    "models_loaded": {"fake": True},
                    "config_versions": {"test": "fixture"},
                },
            }
        )


def _client(tmp_path: Path, service: FakeImageProcessingService) -> TestClient:
    placeholder = tmp_path / "config.yaml"
    placeholder.write_text("fixture: true\n", encoding="utf-8")
    settings = Settings(
        models_config=placeholder,
        geometry_config=placeholder,
        risk_config=placeholder,
        evidence_root=tmp_path / "evidence",
    )
    return TestClient(create_app(settings=settings, image_processing_service=service))


def _png_bytes() -> bytes:
    image = np.full((16, 24, 3), 127, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    return payload.tobytes()


def test_health_reports_process_and_pipeline_readiness(tmp_path: Path) -> None:
    with _client(tmp_path, FakeImageProcessingService()) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_ready": True,
        "pipeline_version": "test-pipeline",
        "models_loaded": {"fake": True},
    }


def test_empty_image_is_rejected_before_pipeline(tmp_path: Path) -> None:
    service = FakeImageProcessingService()
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("empty.png", b"", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "uploaded image is empty"
    assert service.calls == 0


def test_corrupted_image_is_rejected_before_pipeline(tmp_path: Path) -> None:
    service = FakeImageProcessingService()
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
        )

    assert response.status_code == 400
    assert service.calls == 0


def test_image_decode_failure_is_rejected_before_pipeline(tmp_path: Path) -> None:
    service = FakeImageProcessingService()
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("broken.jpg", b"\xff\xd8\xffgarbage", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "uploaded image could not be decoded"
    assert service.calls == 0


def test_unsupported_image_type_is_rejected(tmp_path: Path) -> None:
    service = FakeImageProcessingService()
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("image.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 400
    assert service.calls == 0


def test_mocked_image_analysis_returns_public_contract(tmp_path: Path) -> None:
    service = FakeImageProcessingService()
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("crane.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["assessment"]["risk_level"] == "WARNING"
    assert response.json()["metadata"]["image_width"] == 24
    assert service.calls == 1


def test_pipeline_exception_returns_sanitized_500(tmp_path: Path) -> None:
    service = FakeImageProcessingService(failure=RuntimeError("secret traceback"))
    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/detection/image",
            files={"file": ("crane.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "image processing failed"}
    assert "secret traceback" not in response.text


def test_unknown_resource_returns_404(tmp_path: Path) -> None:
    with _client(tmp_path, FakeImageProcessingService()) as client:
        response = client.get("/api/v1/unknown")

    assert response.status_code == 404


def test_cors_allows_only_configured_frontend_origin(tmp_path: Path) -> None:
    with _client(tmp_path, FakeImageProcessingService()) as client:
        allowed = client.options(
            "/api/v1/detection/image",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        denied = client.options(
            "/api/v1/detection/image",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ("http://localhost:5173")
    assert "access-control-allow-origin" not in denied.headers
