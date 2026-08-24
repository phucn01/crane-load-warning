from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from geometry_engine import (
    GeometryFrameResult,
    ImagePoint,
    LoadAnchorCandidate,
    LoadAnchorCandidates,
    LoadGeometryResult,
    LoadSafetyZones,
    PersonGeometryResult,
    PersonRepresentative,
    PseudoBEVPoint,
    PseudoBEVRectangle,
    RepresentativeDepth,
)
from risk_engine import (
    FrameRiskAssessment,
    MediaFrameContext,
    MediaType,
    Rectangle2D,
    RiskFrameResult,
    RiskLevel,
    SafetyAssessment,
    ZoneGeometry,
)
from vision_engine.contracts import Detection


@dataclass(frozen=True)
class AnnotationBundle:
    image_bgr: np.ndarray
    detections: tuple[Detection, ...]
    geometry: GeometryFrameResult
    risk_result: RiskFrameResult
    context: MediaFrameContext


@pytest.fixture
def annotation_bundle() -> AnnotationBundle:
    image = np.full((200, 240, 3), 40, dtype=np.uint8)
    person_mask = np.zeros((200, 240), dtype=bool)
    person_mask[70:171, 30:61] = True
    detections = (
        _detection("person", (20.0, 60.0, 70.0, 180.0), mask=person_mask),
        _detection("hanging_object", (90.0, 70.0, 160.0, 145.0)),
        _detection("hanging_rope", (120.0, 31.0, 130.0, 70.0)),
    )

    representative_depth = RepresentativeDepth(
        value=0.5,
        source="fixture",
        quality="high",
    )
    person = PersonGeometryResult(
        person_id="person_01",
        confidence=0.91,
        bbox=(20.0, 60.0, 70.0, 180.0),
        representative=PersonRepresentative(
            point=ImagePoint(45.0, 165.0),
            point_source="mask_bottom",
            depth=representative_depth,
        ),
        pseudo_bev_point=PseudoBEVPoint(0.05, 0.30),
        mask_reliable=True,
    )
    anchors = (
        _anchor("anchor-1", 105.0, 120.0),
        _anchor("anchor-2", 145.0, 120.0),
    )
    pseudo_bev_anchors = (
        PseudoBEVPoint(-0.12, 0.50),
        PseudoBEVPoint(0.12, 0.50),
    )
    zones = LoadSafetyZones(
        footprint=PseudoBEVRectangle(-0.15, 0.15, 0.44, 0.56),
        danger=PseudoBEVRectangle(-0.30, 0.30, 0.30, 0.70),
        warning=PseudoBEVRectangle(-0.50, 0.50, 0.10, 0.90),
    )
    load = LoadGeometryResult(
        load_id="hanging_object_01",
        confidence=0.88,
        bbox=(90.0, 70.0, 160.0, 145.0),
        representative_depth=representative_depth,
        candidate_selection=LoadAnchorCandidates(
            seed_depth=0.5,
            seed_source="fixture",
            generated_patch_count=2,
            rejected_invalid_depth_count=0,
            candidates=anchors,
        ),
        inner_bbox=(105.0, 88.0, 145.0, 127.0),
        connected_candidates=anchors,
        final_anchors=anchors,
        pseudo_bev_points=pseudo_bev_anchors,
        safety_zones=zones,
    )
    geometry = GeometryFrameResult(
        frame_id="frame-000000",
        depth_low=0.1,
        depth_high=0.9,
        persons=(person,),
        loads=(load,),
    )
    risk_zones = ZoneGeometry(
        danger=Rectangle2D(-0.30, 0.30, 0.30, 0.70),
        warning=Rectangle2D(-0.50, 0.50, 0.10, 0.90),
    )
    pair = SafetyAssessment(
        level=RiskLevel.DANGER,
        person_id=person.person_id,
        load_id=load.load_id,
        zone_geometry=risk_zones,
        confidence=person.confidence,
        assessment_reliable=True,
        matched_zone=RiskLevel.DANGER,
    )
    frame_assessment = FrameRiskAssessment(
        frame_id=geometry.frame_id,
        level=RiskLevel.DANGER,
        assessment_reliable=True,
        quality_reasons=(),
        pair_assessments=(pair,),
        contributing_person_ids=(person.person_id,),
        contributing_load_ids=(load.load_id,),
    )
    context = MediaFrameContext(
        upload_id="image-001",
        frame_id=geometry.frame_id,
        frame_index=0,
        timestamp=0.0,
        media_type=MediaType.IMAGE,
    )
    return AnnotationBundle(
        image_bgr=image,
        detections=detections,
        geometry=geometry,
        risk_result=RiskFrameResult(frame_assessment),
        context=context,
    )


def _detection(
    class_name: str,
    bbox: tuple[float, float, float, float],
    *,
    mask: np.ndarray | None = None,
) -> Detection:
    x1, y1, x2, y2 = bbox
    return {
        "source_model": "fixture-model",
        "class_id": 0,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox": bbox,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "mask": mask,
    }


def _anchor(candidate_id: str, x: float, y: float) -> LoadAnchorCandidate:
    return LoadAnchorCandidate(
        candidate_id=candidate_id,
        grid_x=0,
        grid_y=0,
        point=ImagePoint(x, y),
        patch_bbox=(round(x) - 1, round(y) - 1, round(x) + 1, round(y) + 1),
        depth=0.5,
        valid_depth_count=9,
        valid_depth_fraction=1.0,
        seed_depth_difference=0.0,
        is_seed_consistent=True,
    )
