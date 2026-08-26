"""Thin orchestration adapter over the existing offline image pipeline."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
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
    render_safe_no_load_overlay,
    render_skipped_overlay,
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
    RiskLevel,
    RiskPairInput,
    ZoneGeometry,
    load_risk_policy,
)
from vision_engine import (
    Detection,
    VisionFramePipeline,
    VisionFrameResult,
    build_model_manager,
)
from vision_engine.model_manager import ModelManager

from app.core.config import Settings
from app.core.logging import log_operation
from app.schemas.detection import ImageDetectionResponse

PIPELINE_VERSION = "vision-geometry-risk-annotation-v1"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """One independently assessed frame and its public evidence views."""

    annotated_bgr: NDArray[np.uint8]
    pseudo_bev_bgr: NDArray[np.uint8] | None
    risk_level: RiskLevel | None
    assessment_status: str = "FULL_EVALUATION"
    confidence: float | None = None
    assessment_reliable: bool = False
    quality_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _FramePipelineResult:
    vision: VisionFrameResult
    geometry: GeometryFrameResult
    risk: RiskFrameResult
    annotated_bgr: NDArray[np.uint8]


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
        with log_operation(LOGGER, "load_runtime_configuration"):
            model_config = _load_yaml(settings.models_config, "models config")
            geometry_config = load_geometry_config(
                _required_file(settings.geometry_config, "geometry config")
            )
            risk_policy = load_risk_policy(
                _required_file(settings.risk_config, "risk config")
            )
        with log_operation(LOGGER, "build_model_manager"):
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
        with log_operation(LOGGER, "model_manager_load_all"):
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

    def process(
        self,
        image_bgr: NDArray[np.uint8],
        *,
        run_id: str | None = None,
    ) -> ImageDetectionResponse:
        """Run one decoded BGR image and persist only public annotation images."""

        started = perf_counter()
        run_id = run_id or uuid4().hex
        LOGGER.info(
            "=== START | OPERATION=IMAGE_PROCESSING | RUN_ID=%s | WIDTH=%s | "
            "HEIGHT=%s ===",
            run_id,
            image_bgr.shape[1],
            image_bgr.shape[0],
        )
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
            detections = self.vision_pipeline.detect(
                image_bgr, frame_id=context.frame_id
            )
            has_person = any(d["class_name"] == "person" for d in detections)
            has_load = any(d["class_name"] == "hanging_object" for d in detections)
            if not (has_person and has_load):
                status = "SAFE_NO_LOAD" if has_person else (
                    "SKIPPED_NO_PERSON" if has_load else "SKIPPED_NO_REQUIRED_OBJECTS"
                )
                annotated = (
                    render_safe_no_load_overlay(
                        image_bgr, detections, frame_id=context.frame_id,
                        frame_local_labels=False,
                    )
                    if status == "SAFE_NO_LOAD"
                    else render_skipped_overlay(
                        image_bgr, detections, status=status,
                        frame_id=context.frame_id,
                    )
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                rgb_path = run_dir / "rgb.png"
                _write_png(rgb_path, annotated)
                counts = Counter(item["class_name"] for item in detections)
                models = self.model_manager.metadata().get("models", {})
                payload = {
                    "status": "completed",
                    "assessment_status": status,
                    "processing_time_ms": round((perf_counter() - started) * 1000.0, 3),
                    "assessment": {
                        "risk_level": "SAFE",
                        "assessment_reliable": status == "SAFE_NO_LOAD",
                        "quality_reasons": [status.lower()],
                        "contributing_person_ids": [],
                        "contributing_load_ids": [],
                        "pairs": [],
                    },
                    "summary": {
                        "person_count": counts.get("person", 0),
                        "load_count": counts.get("hanging_object", 0),
                        "rope_count": counts.get("hanging_rope", 0) + counts.get("rope", 0),
                    },
                    "detections": _detection_payloads(tuple(detections)),
                    "geometry": {
                        "coordinate_system": "relative_pseudo_bev_not_metric",
                        "depth_low": None,
                        "depth_high": None,
                        "quality_reasons": [status.lower()],
                        "persons": [],
                        "loads": [],
                    },
                    "evidence": {
                        "rgb_url": f"/evidence/{run_id}/rgb.png",
                        "pseudo_bev_url": None,
                        "combined_url": f"/evidence/{run_id}/rgb.png",
                    },
                    "metadata": {
                        "pipeline_version": PIPELINE_VERSION,
                        "frame_id": run_id,
                        "image_width": int(image_bgr.shape[1]),
                        "image_height": int(image_bgr.shape[0]),
                        "depth": None,
                        "models_loaded": {
                            str(name): bool(values.get("loaded", False))
                            for name, values in models.items()
                            if isinstance(values, Mapping)
                        },
                        "config_versions": self.config_versions,
                    },
                }
                return ImageDetectionResponse.model_validate(payload)
            frame = self._run_frame_pipeline(
                image_bgr,
                context=context,
                update_temporal_event=True,
                frame_local_labels=False,
            )
            traceability = EvidenceTraceability(
                pipeline_version=PIPELINE_VERSION,
                model_versions=_model_versions(self.model_manager.metadata()),
                config_versions=self.config_versions,
            )
            with log_operation(LOGGER, "write_image_evidence", run_id=run_id):
                evidence = self._write_annotations(
                    run_dir=run_dir,
                    image_bgr=image_bgr,
                    annotated_bgr=frame.annotated_bgr,
                    detections=frame.vision.detections,
                    geometry=frame.geometry,
                    risk_result=frame.risk,
                    context=context,
                    traceability=traceability,
                )

        counts = Counter(item["class_name"] for item in frame.vision.detections)
        models = self.model_manager.metadata().get("models", {})
        payload = {
            "status": "completed",
            "processing_time_ms": round((perf_counter() - started) * 1000.0, 3),
            "assessment": _assessment_payload(frame.risk.assessment),
            "summary": {
                "person_count": counts.get("person", 0),
                "load_count": counts.get("hanging_object", 0),
                "rope_count": counts.get("hanging_rope", 0) + counts.get("rope", 0),
            },
            "detections": _detection_payloads(frame.vision.detections),
            "geometry": _geometry_payload(frame.geometry),
            "evidence": evidence,
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "frame_id": run_id,
                "image_width": int(image_bgr.shape[1]),
                "image_height": int(image_bgr.shape[0]),
                "depth": frame.vision.relative_depth.metadata.to_dict(),
                "models_loaded": {
                    str(name): bool(values.get("loaded", False))
                    for name, values in models.items()
                    if isinstance(values, Mapping)
                },
                "config_versions": self.config_versions,
            },
        }
        response = ImageDetectionResponse.model_validate(payload)
        LOGGER.info(
            "=== END | OPERATION=IMAGE_PROCESSING | RUN_ID=%s | "
            "DURATION_MS=%.3f | RISK_LEVEL=%s | DETECTIONS=%s ===",
            run_id,
            (perf_counter() - started) * 1000.0,
            frame.risk.assessment.level.value,
            len(frame.vision.detections),
        )
        return response

    def process_video_frame(
        self,
        image_bgr: NDArray[np.uint8],
        *,
        upload_id: str,
        frame_index: int,
        timestamp: float,
    ) -> ProcessedFrame:
        """Run the existing frame pipeline once without cross-frame association."""

        frame_id = f"{upload_id}:frame:{frame_index}"
        context = MediaFrameContext(
            upload_id=upload_id,
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp=timestamp,
            media_type=MediaType.VIDEO,
        )
        started = perf_counter()
        with self._processing_lock:
            # Every video frame is detected, but person-only frames do not need
            # depth/geometry/risk inference.  They are explicit SAFE evidence.
            detections = self.vision_pipeline.detect(
                image_bgr, frame_id=context.frame_id
            )
            has_person = any(d["class_name"] == "person" for d in detections)
            has_load = any(d["class_name"] == "hanging_object" for d in detections)
            if has_person and not has_load:
                annotated = render_safe_no_load_overlay(
                    image_bgr,
                    detections,
                    frame_id=context.frame_id,
                    frame_local_labels=True,
                )
                result = ProcessedFrame(
                    annotated_bgr=annotated,
                    pseudo_bev_bgr=None,
                    risk_level=RiskLevel.SAFE,
                    assessment_status="SAFE_NO_LOAD",
                    confidence=max(
                        (float(d["confidence"]) for d in detections if d["class_name"] == "person"),
                        default=None,
                    ),
                    assessment_reliable=True,
                    quality_reasons=("no_hanging_object_detected",),
                )
                LOGGER.info(
                    "=== END | OPERATION=VIDEO_FRAME_PROCESSING | JOB_ID=%s | "
                    "FRAME_INDEX=%s | DURATION_MS=%.3f | STATUS=SAFE_NO_LOAD ===",
                    upload_id,
                    frame_index,
                    (perf_counter() - started) * 1000.0,
                )
                return result

            if not has_person:
                status = (
                    "SKIPPED_NO_PERSON"
                    if has_load
                    else "SKIPPED_NO_REQUIRED_OBJECTS"
                )
                return ProcessedFrame(
                    annotated_bgr=render_skipped_overlay(
                        image_bgr,
                        detections,
                        status=status,
                        frame_id=context.frame_id,
                    ),
                    pseudo_bev_bgr=None,
                    risk_level=None,
                    assessment_status=status,
                    assessment_reliable=None,
                    quality_reasons=("person_not_detected",),
                )

            vision = VisionFrameResult(
                frame_id=context.frame_id,
                detections=detections,
                relative_depth=self.vision_pipeline.estimate_depth(
                    image_bgr, frame_id=context.frame_id
                ),
            )
            frame = self._run_frame_pipeline(
                image_bgr,
                context=context,
                update_temporal_event=False,
                frame_local_labels=True,
                vision=vision,
            )
            pseudo_bev = (
                None
                if frame.risk.assessment.level is RiskLevel.SAFE
                else self.annotation_composer.render_pseudo_bev(
                    frame.geometry,
                    frame.risk.assessment,
                )
            )
        result = ProcessedFrame(
            annotated_bgr=frame.annotated_bgr,
            pseudo_bev_bgr=pseudo_bev,
            risk_level=frame.risk.assessment.level,
            assessment_status="FULL_EVALUATION",
            confidence=max(
                (
                    pair.confidence
                    for pair in frame.risk.assessment.pair_assessments
                ),
                default=None,
            ),
            assessment_reliable=frame.risk.assessment.assessment_reliable,
            quality_reasons=tuple(frame.risk.assessment.quality_reasons),
        )
        LOGGER.info(
            "=== END | OPERATION=VIDEO_FRAME_PROCESSING | JOB_ID=%s | "
            "FRAME_INDEX=%s | DURATION_MS=%.3f | RISK_LEVEL=%s ===",
            upload_id,
            frame_index,
            (perf_counter() - started) * 1000.0,
            result.risk_level.value,
        )
        return result

    def _run_frame_pipeline(
        self,
        image_bgr: NDArray[np.uint8],
        *,
        context: MediaFrameContext,
        update_temporal_event: bool,
        frame_local_labels: bool,
        vision: VisionFrameResult | None = None,
    ) -> _FramePipelineResult:
        """Single shared Vision -> Geometry -> Risk -> Annotation path."""

        fields = {"frame_id": context.frame_id}
        with log_operation(LOGGER, "vision_pipeline", **fields):
            vision = vision or self.vision_pipeline.process(
                image_bgr, frame_id=context.frame_id
            )
        with log_operation(LOGGER, "geometry_pipeline", **fields):
            geometry = self.geometry_pipeline.process(
                vision.detections,
                vision.relative_depth.depth_map,
                frame_id=context.frame_id,
            )
        with log_operation(LOGGER, "risk_pipeline", **fields):
            risk = self.risk_pipeline.process(
                _geometry_to_risk_inputs(geometry),
                context=context,
                update_temporal_event=update_temporal_event,
            )
        with log_operation(LOGGER, "render_image_overlay", **fields):
            annotated = render_image_overlay(
                image_bgr,
                vision.detections,
                risk.assessment,
                frame_local_labels=frame_local_labels,
            )
        return _FramePipelineResult(
            vision=vision,
            geometry=geometry,
            risk=risk,
            annotated_bgr=annotated,
        )

    def _write_annotations(
        self,
        *,
        run_dir: Path,
        image_bgr: NDArray[np.uint8],
        annotated_bgr: NDArray[np.uint8],
        detections: tuple[Detection, ...],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
    ) -> dict[str, str | None]:
        run_dir.mkdir(parents=True, exist_ok=False)
        rgb_path = run_dir / "rgb.png"
        pseudo_bev_path = run_dir / "pseudo_bev.png"
        _write_png(rgb_path, annotated_bgr)
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
            "combined_url": f"{prefix}/{combined.evidence_image_path.name}",
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


__all__ = ["PIPELINE_VERSION", "ImageProcessingService", "ProcessedFrame"]
