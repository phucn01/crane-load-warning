"""Thin orchestration adapter over the existing offline image pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import yaml
from annotation_engine import (
    EvidenceTraceability,
    OfflineEvidenceComposer,
    render_image_overlay,
)
from geometry_engine import (
    GeometryFramePipeline,
    GeometryFrameResult,
    load_geometry_config,
)
from geometry_engine import (
    PseudoBEVRectangle as GeometryRectangle,
)
from numpy.typing import NDArray
from risk_engine import (
    EventStateMachine,
    MediaFrameContext,
    MediaType,
    PersonObservation,
    Point2D,
    Rectangle2D,
    RiskEvaluator,
    RiskFramePipeline,
    RiskFrameResult,
    RiskPairInput,
    ZoneGeometry,
    load_risk_policy,
)
from vision_engine import Detection, VisionFramePipeline, build_model_manager
from vision_engine.model_manager import ModelManager

from app.core.config import Settings
from app.schemas.detection import ImageDetectionResponse

PIPELINE_VERSION = "vision-geometry-risk-annotation-v1"


class ImageProcessingService:
    """Reuse one model manager while processing uploaded images end to end."""

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        geometry_pipeline: GeometryFramePipeline,
        risk_pipeline: RiskFramePipeline,
        annotation_composer: OfflineEvidenceComposer,
        evidence_root: Path,
        config_versions: Mapping[str, str],
    ) -> None:
        self.model_manager = model_manager
        self.vision_pipeline = VisionFramePipeline(model_manager)
        self.geometry_pipeline = geometry_pipeline
        self.risk_pipeline = risk_pipeline
        self.annotation_composer = annotation_composer
        self.evidence_root = evidence_root.resolve()
        self.config_versions = dict(config_versions)
        self._processing_lock = Lock()
        self.evidence_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> ImageProcessingService:
        model_config = _load_yaml(settings.models_config, "models config")
        geometry_config = load_geometry_config(
            _required_file(settings.geometry_config, "geometry config")
        )
        risk_policy = load_risk_policy(
            _required_file(settings.risk_config, "risk config")
        )
        model_manager = build_model_manager(
            model_config,
            config_dir=settings.models_config.parent,
        )
        return cls(
            model_manager=model_manager,
            geometry_pipeline=GeometryFramePipeline(geometry_config),
            risk_pipeline=RiskFramePipeline(
                evaluator=RiskEvaluator(risk_policy.evaluation),
                state_machine=EventStateMachine(risk_policy.events),
            ),
            annotation_composer=OfflineEvidenceComposer(),
            evidence_root=settings.evidence_root,
            config_versions={
                "models": _config_fingerprint(settings.models_config),
                "geometry": _config_fingerprint(settings.geometry_config),
                "risk": _config_fingerprint(settings.risk_config),
            },
        )

    def preload_models(self) -> None:
        self.model_manager.load_all()

    def readiness(self) -> dict[str, Any]:
        metadata = self.model_manager.metadata()
        models = metadata.get("models", {})
        return {
            "pipeline_ready": True,
            "pipeline_version": PIPELINE_VERSION,
            "models_loaded": {
                str(name): bool(values.get("loaded", False))
                for name, values in models.items()
                if isinstance(values, Mapping)
            },
        }

    def process(self, image_bgr: NDArray[np.uint8]) -> ImageDetectionResponse:
        """Run one decoded BGR image and persist only public annotation images."""

        started = perf_counter()
        run_id = uuid4().hex
        run_dir = self.evidence_root / run_id
        context = MediaFrameContext(
            upload_id=run_id,
            frame_id=run_id,
            frame_index=0,
            timestamp=0.0,
            media_type=MediaType.IMAGE,
        )

        # The adapters contain their own safeguards, while this lock also prevents
        # concurrent access to model backends that do not guarantee thread safety.
        with self._processing_lock:
            vision = self.vision_pipeline.process(image_bgr, frame_id=run_id)
            geometry = self.geometry_pipeline.process(
                vision.detections,
                vision.relative_depth.depth_map,
                frame_id=run_id,
            )
            risk = self.risk_pipeline.process(
                _geometry_to_risk_inputs(geometry),
                context=context,
            )
            traceability = EvidenceTraceability(
                pipeline_version=PIPELINE_VERSION,
                model_versions=_model_versions(self.model_manager.metadata()),
                config_versions=self.config_versions,
            )
            evidence = self._write_annotations(
                run_dir=run_dir,
                image_bgr=image_bgr,
                detections=vision.detections,
                geometry=geometry,
                risk_result=risk,
                context=context,
                traceability=traceability,
            )

        counts = Counter(item["class_name"] for item in vision.detections)
        models = self.model_manager.metadata().get("models", {})
        payload = {
            "status": "completed",
            "processing_time_ms": round((perf_counter() - started) * 1000.0, 3),
            "assessment": _assessment_payload(risk.assessment),
            "summary": {
                "person_count": counts.get("person", 0),
                "load_count": counts.get("hanging_object", 0),
                "rope_count": counts.get("hanging_rope", 0) + counts.get("rope", 0),
            },
            "detections": _detection_payloads(vision.detections),
            "geometry": _geometry_payload(geometry),
            "evidence": evidence,
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "frame_id": run_id,
                "image_width": int(image_bgr.shape[1]),
                "image_height": int(image_bgr.shape[0]),
                "depth": vision.relative_depth.metadata.to_dict(),
                "models_loaded": {
                    str(name): bool(values.get("loaded", False))
                    for name, values in models.items()
                    if isinstance(values, Mapping)
                },
                "config_versions": self.config_versions,
            },
        }
        return ImageDetectionResponse.model_validate(payload)

    def _write_annotations(
        self,
        *,
        run_dir: Path,
        image_bgr: NDArray[np.uint8],
        detections: tuple[Detection, ...],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
    ) -> dict[str, str | None]:
        run_dir.mkdir(parents=True, exist_ok=False)
        rgb_path = run_dir / "rgb.png"
        pseudo_bev_path = run_dir / "pseudo_bev.png"
        _write_png(
            rgb_path,
            render_image_overlay(image_bgr, detections, risk_result.assessment),
        )
        _write_png(
            pseudo_bev_path,
            self.annotation_composer.render_pseudo_bev(
                geometry,
                risk_result.assessment,
            ),
        )
        combined = self.annotation_composer.write(
            image_bgr=image_bgr,
            detections=detections,
            geometry=geometry,
            risk_result=risk_result,
            context=context,
            traceability=traceability,
            output_dir=run_dir,
        )
        prefix = f"/evidence/{run_dir.name}"
        return {
            "rgb_url": f"{prefix}/{rgb_path.name}",
            "pseudo_bev_url": f"{prefix}/{pseudo_bev_path.name}",
            "combined_url": (
                None
                if combined is None
                else f"{prefix}/{combined.evidence_image_path.name}"
            ),
        }


def _geometry_to_risk_inputs(
    geometry: GeometryFrameResult,
) -> tuple[RiskPairInput, ...]:
    pairs: list[RiskPairInput] = []
    for person in geometry.persons:
        point = person.pseudo_bev_point
        for load in geometry.loads:
            zones = load.safety_zones
            reasons = tuple(
                dict.fromkeys((*person.quality_reasons, *load.quality_reasons))
            )
            pairs.append(
                RiskPairInput(
                    person=PersonObservation(
                        person_id=person.person_id,
                        anchor=(
                            None
                            if point is None
                            else Point2D(point.lateral, point.longitudinal)
                        ),
                        confidence=person.confidence,
                        anchor_reliable=point is not None,
                        mask_reliable=person.mask_reliable,
                        quality_reasons=reasons,
                    ),
                    load_id=load.load_id,
                    zones=(
                        None
                        if zones is None
                        else ZoneGeometry(
                            danger=_risk_rectangle(zones.danger),
                            warning=_risk_rectangle(zones.warning),
                        )
                    ),
                )
            )
    return tuple(pairs)


def _risk_rectangle(rectangle: GeometryRectangle) -> Rectangle2D:
    return Rectangle2D(
        minimum_x=rectangle.minimum_lateral,
        maximum_x=rectangle.maximum_lateral,
        minimum_y=rectangle.minimum_longitudinal,
        maximum_y=rectangle.maximum_longitudinal,
    )


def _assessment_payload(assessment: Any) -> dict[str, Any]:
    return {
        "risk_level": assessment.level.value,
        "assessment_reliable": assessment.assessment_reliable,
        "quality_reasons": list(assessment.quality_reasons),
        "contributing_person_ids": list(assessment.contributing_person_ids),
        "contributing_load_ids": list(assessment.contributing_load_ids),
        "pairs": [
            {
                "person_id": pair.person_id,
                "load_id": pair.load_id,
                "risk_level": pair.level.value,
                "matched_zone": (
                    None if pair.matched_zone is None else pair.matched_zone.value
                ),
                "confidence": pair.confidence,
                "assessment_reliable": pair.assessment_reliable,
                "quality_reasons": list(pair.quality_reasons),
            }
            for pair in assessment.pair_assessments
        ],
    }


def _detection_payloads(detections: tuple[Detection, ...]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    payloads: list[dict[str, Any]] = []
    for detection in detections:
        class_name = detection["class_name"]
        counts[class_name] += 1
        x1, y1, x2, y2 = detection["bbox"]
        payloads.append(
            {
                "detection_id": f"{class_name}_{counts[class_name]:02d}",
                "source_model": detection["source_model"],
                "class_id": detection["class_id"],
                "class_name": class_name,
                "confidence": detection["confidence"],
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "has_mask": isinstance(detection["mask"], np.ndarray),
            }
        )
    return payloads


def _geometry_payload(geometry: GeometryFrameResult) -> dict[str, Any]:
    return {
        "coordinate_system": "relative_pseudo_bev_not_metric",
        "depth_low": geometry.depth_low,
        "depth_high": geometry.depth_high,
        "quality_reasons": list(geometry.quality_reasons),
        "persons": [
            {
                "person_id": person.person_id,
                "confidence": person.confidence,
                "bbox": _bbox_payload(person.bbox),
                "pseudo_bev_point": _point_payload(person.pseudo_bev_point),
                "mask_reliable": person.mask_reliable,
                "quality_reasons": list(person.quality_reasons),
            }
            for person in geometry.persons
        ],
        "loads": [
            {
                "load_id": load.load_id,
                "confidence": load.confidence,
                "bbox": _bbox_payload(load.bbox),
                "pseudo_bev_points": [
                    _point_payload(point) for point in load.pseudo_bev_points
                ],
                "safety_zones": (
                    None
                    if load.safety_zones is None
                    else {
                        "footprint": _rectangle_payload(load.safety_zones.footprint),
                        "danger": _rectangle_payload(load.safety_zones.danger),
                        "warning": _rectangle_payload(load.safety_zones.warning),
                    }
                ),
                "quality_reasons": list(load.quality_reasons),
            }
            for load in geometry.loads
        ],
    }


def _bbox_payload(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(("x1", "y1", "x2", "y2"), bbox, strict=True))


def _point_payload(point: Any) -> dict[str, float] | None:
    if point is None:
        return None
    return {"lateral": point.lateral, "longitudinal": point.longitudinal}


def _rectangle_payload(rectangle: GeometryRectangle) -> dict[str, float]:
    return {
        "minimum_lateral": rectangle.minimum_lateral,
        "maximum_lateral": rectangle.maximum_lateral,
        "minimum_longitudinal": rectangle.minimum_longitudinal,
        "maximum_longitudinal": rectangle.maximum_longitudinal,
    }


def _write_png(path: Path, image: NDArray[np.generic]) -> None:
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"could not encode annotation image: {path.name}")
    path.write_bytes(payload.tobytes())


def _required_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    resolved = _required_file(path, label)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} root must be a YAML mapping")
    return payload


def _config_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(_required_file(path, "config").read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _model_versions(metadata: Mapping[str, Any]) -> dict[str, str]:
    models = metadata.get("models", {})
    if not isinstance(models, Mapping):
        return {"models": "unknown"}
    return {
        str(name): str(values.get("identifier", values.get("name", "unknown")))
        if isinstance(values, Mapping)
        else str(values)
        for name, values in models.items()
    }


__all__ = ["PIPELINE_VERSION", "ImageProcessingService"]
