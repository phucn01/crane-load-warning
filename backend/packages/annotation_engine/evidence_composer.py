"""Compose and persist offline evidence for non-safe frame assessments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from geometry_engine import GeometryFrameResult, PseudoBEVPoint, PseudoBEVRectangle
from numpy.typing import NDArray
from pipeline_timeline import PipelineTimeline
from risk_engine import (
    EventDecision,
    FrameRiskAssessment,
    MediaFrameContext,
    RiskFrameResult,
    RiskLevel,
    SafetyAssessment,
)
from vision_engine.contracts import Detection

from .contracts import EvidenceArtifacts, EvidenceTraceability
from .image_overlay import RISK_COLORS_BGR, render_image_overlay
from .pseudo_bev_overlay import render_pseudo_bev_chart


class OfflineEvidenceComposer:
    """Create paired image and JSON evidence for WARNING/DANGER frames."""

    def __init__(
        self,
        *,
        pseudo_bev_size: tuple[int, int] = (960, 720),
        timeline: PipelineTimeline | None = None,
    ) -> None:
        width, height = pseudo_bev_size
        if width < 320 or height < 240:
            raise ValueError("pseudo_bev_size dimensions must be at least 320x240")
        self.pseudo_bev_size = width, height
        self.timeline = timeline

    def render_pseudo_bev(
        self,
        geometry: GeometryFrameResult,
        assessment: FrameRiskAssessment,
    ) -> NDArray[np.uint8]:
        """Render the one BEV representation shared by all evidence outputs."""

        width, height = self.pseudo_bev_size
        return render_pseudo_bev_chart(
            geometry,
            assessment,
            width=width,
            height=height,
        )

    def compose(
        self,
        *,
        image_bgr: NDArray[np.generic],
        detections: Iterable[Detection],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
    ) -> NDArray[np.uint8]:
        """Compose camera, Pseudo-BEV, and concise assessment metadata."""

        if self.timeline is not None:
            with self.timeline.track(
                "annotation",
                "compose",
                frame_id=context.frame_id,
            ):
                return self._compose(
                    image_bgr=image_bgr,
                    detections=detections,
                    geometry=geometry,
                    risk_result=risk_result,
                    context=context,
                    traceability=traceability,
                )
        return self._compose(
            image_bgr=image_bgr,
            detections=detections,
            geometry=geometry,
            risk_result=risk_result,
            context=context,
            traceability=traceability,
        )

    def _compose(
        self,
        *,
        image_bgr: NDArray[np.generic],
        detections: Iterable[Detection],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
    ) -> NDArray[np.uint8]:

        _validate_frame_identity(geometry, risk_result.assessment, context)
        image_overlay = render_image_overlay(
            image_bgr,
            tuple(detections),
            risk_result.assessment,
        )
        pseudo_bev_overlay = self.render_pseudo_bev(
            geometry,
            risk_result.assessment,
        )
        return compose_evidence_image(
            image_overlay,
            pseudo_bev_overlay,
            assessment=risk_result.assessment,
            context=context,
            traceability=traceability,
            panel_size=self.pseudo_bev_size,
        )

    def write(
        self,
        *,
        image_bgr: NDArray[np.generic],
        detections: Iterable[Detection],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
        output_dir: str | Path,
        overwrite: bool = False,
    ) -> EvidenceArtifacts | None:
        """Write image and JSON only when the immediate frame is non-safe."""

        if self.timeline is not None:
            with self.timeline.track(
                "annotation",
                "write",
                frame_id=context.frame_id,
            ):
                return self._write(
                    image_bgr=image_bgr,
                    detections=detections,
                    geometry=geometry,
                    risk_result=risk_result,
                    context=context,
                    traceability=traceability,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
        return self._write(
            image_bgr=image_bgr,
            detections=detections,
            geometry=geometry,
            risk_result=risk_result,
            context=context,
            traceability=traceability,
            output_dir=output_dir,
            overwrite=overwrite,
        )

    def _write(
        self,
        *,
        image_bgr: NDArray[np.generic],
        detections: Iterable[Detection],
        geometry: GeometryFrameResult,
        risk_result: RiskFrameResult,
        context: MediaFrameContext,
        traceability: EvidenceTraceability,
        output_dir: str | Path,
        overwrite: bool,
    ) -> EvidenceArtifacts | None:

        if risk_result.assessment.level is RiskLevel.SAFE:
            return None
        _validate_frame_identity(geometry, risk_result.assessment, context)
        composed = self.compose(
            image_bgr=image_bgr,
            detections=detections,
            geometry=geometry,
            risk_result=risk_result,
            context=context,
            traceability=traceability,
        )
        stem = _artifact_stem(context)
        artifact_dir = Path(output_dir)
        image_path = artifact_dir / f"{stem}_evidence.png"
        json_path = artifact_dir / f"{stem}_assessment.json"
        if not overwrite:
            existing = [path for path in (image_path, json_path) if path.exists()]
            if existing:
                raise FileExistsError(f"evidence artifact already exists: {existing[0]}")

        encoded, png = cv2.imencode(".png", composed)
        if not encoded:
            raise OSError("could not encode evidence image as PNG")
        payload = build_assessment_payload(
            geometry=geometry,
            risk_result=risk_result,
            context=context,
            traceability=traceability,
            evidence_image_ref=image_path.name,
        )
        json_bytes = (
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(png.tobytes())
        json_path.write_bytes(json_bytes)
        return EvidenceArtifacts(image_path, json_path)


def compose_evidence_image(
    image_overlay: NDArray[np.generic],
    pseudo_bev_overlay: NDArray[np.generic],
    *,
    assessment: FrameRiskAssessment,
    context: MediaFrameContext,
    traceability: EvidenceTraceability,
    panel_size: tuple[int, int] = (640, 640),
) -> NDArray[np.uint8]:
    """Place the two explanatory views beside a deterministic metadata footer."""

    panel_width, panel_height = panel_size
    left = _fit_panel(
        image_overlay,
        target_height=panel_height,
        target_width=panel_width,
    )
    right = _fit_panel(
        pseudo_bev_overlay,
        target_height=panel_height,
        target_width=panel_width,
    )
    panels = np.hstack((left, right))
    footer = np.full((96, panels.shape[1], 3), 30, dtype=np.uint8)
    color = RISK_COLORS_BGR[assessment.level]
    cv2.rectangle(footer, (0, 0), (10, footer.shape[0]), color, -1)
    primary = (
        f"{context.frame_id} | {assessment.level.value} | "
        f"reliable={assessment.assessment_reliable} | "
        f"pipeline={traceability.pipeline_version}"
    )
    secondary = (
        f"media={context.media_type.value} frame={context.frame_index} "
        f"t={context.timestamp:.3f}s | pairs={len(assessment.pair_assessments)}"
    )
    reasons = ", ".join(assessment.quality_reasons) or "none"
    _put_footer_text(footer, primary, y=27, color=color)
    _put_footer_text(footer, secondary, y=54, color=(220, 220, 220))
    _put_footer_text(footer, f"quality reasons: {reasons}", y=81, color=(200, 200, 200))
    return np.vstack((panels, footer))


def build_assessment_payload(
    *,
    geometry: GeometryFrameResult,
    risk_result: RiskFrameResult,
    context: MediaFrameContext,
    traceability: EvidenceTraceability,
    evidence_image_ref: str,
) -> dict[str, Any]:
    """Build a JSON-safe assessment with model/config traceability."""

    assessment = risk_result.assessment
    _validate_frame_identity(geometry, assessment, context)
    return {
        "schema_version": traceability.schema_version,
        "media": {
            "upload_id": context.upload_id,
            "frame_id": context.frame_id,
            "frame_index": context.frame_index,
            "timestamp": context.timestamp,
            "media_type": context.media_type.value,
        },
        "assessment": {
            "level": assessment.level.value,
            "assessment_reliable": assessment.assessment_reliable,
            "quality_reasons": list(assessment.quality_reasons),
            "contributing_person_ids": list(assessment.contributing_person_ids),
            "contributing_load_ids": list(assessment.contributing_load_ids),
            "pairs": [_pair_payload(pair) for pair in assessment.pair_assessments],
        },
        "event": _event_payload(risk_result.event),
        "geometry": {
            "coordinate_system": "relative_pseudo_bev_not_metric",
            "persons": [
                {
                    "person_id": person.person_id,
                    "track_id": person.track_id,
                    "bbox_xyxy": list(person.bbox),
                    "pseudo_bev_point": _point_payload(person.pseudo_bev_point),
                    "quality_reasons": list(person.quality_reasons),
                }
                for person in geometry.persons
            ],
            "loads": [
                {
                    "load_id": load.load_id,
                    "track_id": load.track_id,
                    "bbox_xyxy": list(load.bbox),
                    "anchors": [_point_payload(point) for point in load.pseudo_bev_points],
                    "zones": (
                        None
                        if load.safety_zones is None
                        else {
                            "footprint": _rectangle_payload(
                                load.safety_zones.footprint
                            ),
                            "danger": _rectangle_payload(load.safety_zones.danger),
                            "warning": _rectangle_payload(load.safety_zones.warning),
                        }
                    ),
                    "quality_reasons": list(load.quality_reasons),
                }
                for load in geometry.loads
            ],
        },
        "traceability": {
            "pipeline_version": traceability.pipeline_version,
            "model_versions": _sorted_mapping(traceability.model_versions),
            "config_versions": _sorted_mapping(traceability.config_versions),
        },
        "artifacts": {"evidence_image": evidence_image_ref},
    }


def _pair_payload(pair: SafetyAssessment) -> dict[str, Any]:
    return {
        "person_id": pair.person_id,
        "person_track_id": pair.person_track_id,
        "load_id": pair.load_id,
        "load_track_id": pair.load_track_id,
        "level": pair.level.value,
        "matched_zone": None if pair.matched_zone is None else pair.matched_zone.value,
        "confidence": pair.confidence,
        "assessment_reliable": pair.assessment_reliable,
        "quality_reasons": list(pair.quality_reasons),
        "zone_geometry": (
            None
            if pair.zone_geometry is None
            else {
                "danger": {
                    "minimum_x": pair.zone_geometry.danger.minimum_x,
                    "maximum_x": pair.zone_geometry.danger.maximum_x,
                    "minimum_y": pair.zone_geometry.danger.minimum_y,
                    "maximum_y": pair.zone_geometry.danger.maximum_y,
                },
                "warning": {
                    "minimum_x": pair.zone_geometry.warning.minimum_x,
                    "maximum_x": pair.zone_geometry.warning.maximum_x,
                    "minimum_y": pair.zone_geometry.warning.minimum_y,
                    "maximum_y": pair.zone_geometry.warning.maximum_y,
                },
            }
        ),
    }


def _event_payload(event: EventDecision | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "event_key": event.event_key,
        "event_scope": event.event_scope.value,
        "level": event.level.value,
        "changed": event.changed,
        "alert_triggered": event.alert_triggered,
        "held_for_quality": event.held_for_quality,
        "technical_reasons": list(event.technical_reasons),
    }


def _point_payload(point: PseudoBEVPoint | None) -> dict[str, float] | None:
    if point is None:
        return None
    return {"lateral": point.lateral, "longitudinal": point.longitudinal}


def _rectangle_payload(rectangle: PseudoBEVRectangle) -> dict[str, float]:
    return {
        "minimum_lateral": rectangle.minimum_lateral,
        "maximum_lateral": rectangle.maximum_lateral,
        "minimum_longitudinal": rectangle.minimum_longitudinal,
        "maximum_longitudinal": rectangle.maximum_longitudinal,
    }


def _sorted_mapping(values: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(values[key]) for key in sorted(values, key=str)}


def _validate_frame_identity(
    geometry: GeometryFrameResult,
    assessment: FrameRiskAssessment,
    context: MediaFrameContext,
) -> None:
    frame_ids = {geometry.frame_id, assessment.frame_id, context.frame_id}
    if len(frame_ids) != 1:
        raise ValueError("geometry, assessment, and media context frame_id must match")


def _artifact_stem(context: MediaFrameContext) -> str:
    safe_frame_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", context.frame_id).strip("._")
    safe_frame_id = safe_frame_id or "frame"
    return f"{context.frame_index:06d}_{safe_frame_id}"


def _fit_panel(
    image: NDArray[np.generic], *, target_height: int, target_width: int
) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("evidence panels must have shape (height, width, 3)")
    source = np.clip(image, 0, 255).astype(np.uint8)
    scale = min(target_width / source.shape[1], target_height / source.shape[0])
    resized_width = max(1, round(source.shape[1] * scale))
    resized_height = max(1, round(source.shape[0] * scale))
    resized = cv2.resize(
        source,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.full((target_height, target_width, 3), 24, dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def _put_footer_text(
    footer: NDArray[np.uint8],
    text: str,
    *,
    y: int,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        footer,
        text[:180],
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


__all__ = [
    "OfflineEvidenceComposer",
    "build_assessment_payload",
    "compose_evidence_image",
]
