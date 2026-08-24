"""Pydantic response contracts for image safety assessment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevelValue = Literal["SAFE", "WARNING", "DANGER"]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictSchema):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(StrictSchema):
    detection_id: str
    source_model: str
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    has_mask: bool


class PseudoBEVPoint(StrictSchema):
    lateral: float
    longitudinal: float


class PseudoBEVRectangle(StrictSchema):
    minimum_lateral: float
    maximum_lateral: float
    minimum_longitudinal: float
    maximum_longitudinal: float


class SafetyZones(StrictSchema):
    footprint: PseudoBEVRectangle
    danger: PseudoBEVRectangle
    warning: PseudoBEVRectangle


class PersonGeometry(StrictSchema):
    person_id: str
    confidence: float
    bbox: BoundingBox
    pseudo_bev_point: PseudoBEVPoint | None
    mask_reliable: bool
    quality_reasons: list[str]


class LoadGeometry(StrictSchema):
    load_id: str
    confidence: float
    bbox: BoundingBox
    pseudo_bev_points: list[PseudoBEVPoint]
    safety_zones: SafetyZones | None
    quality_reasons: list[str]


class GeometryResponse(StrictSchema):
    coordinate_system: Literal["relative_pseudo_bev_not_metric"] = (
        "relative_pseudo_bev_not_metric"
    )
    depth_low: float
    depth_high: float
    quality_reasons: list[str]
    persons: list[PersonGeometry]
    loads: list[LoadGeometry]


class PairAssessment(StrictSchema):
    person_id: str
    load_id: str
    risk_level: RiskLevelValue
    matched_zone: RiskLevelValue | None
    confidence: float = Field(ge=0.0, le=1.0)
    assessment_reliable: bool
    quality_reasons: list[str]


class AssessmentResponse(StrictSchema):
    risk_level: RiskLevelValue
    assessment_reliable: bool
    quality_reasons: list[str]
    contributing_person_ids: list[str]
    contributing_load_ids: list[str]
    pairs: list[PairAssessment]


class DetectionSummary(StrictSchema):
    person_count: int = Field(ge=0)
    load_count: int = Field(ge=0)
    rope_count: int = Field(ge=0)


class EvidenceResponse(StrictSchema):
    rgb_url: str | None = None
    pseudo_bev_url: str | None = None
    combined_url: str | None = None


class DepthMetadata(StrictSchema):
    height: int
    width: int
    dtype: str
    finite_min: float
    finite_max: float
    finite_fraction: float
    convention: str


class ProcessingMetadata(StrictSchema):
    pipeline_version: str
    frame_id: str
    image_width: int
    image_height: int
    depth: DepthMetadata
    models_loaded: dict[str, bool]
    config_versions: dict[str, str]


class ImageDetectionResponse(StrictSchema):
    status: Literal["completed"] = "completed"
    processing_time_ms: float = Field(ge=0.0)
    assessment: AssessmentResponse
    summary: DetectionSummary
    detections: list[DetectionItem]
    geometry: GeometryResponse
    evidence: EvidenceResponse
    metadata: ProcessingMetadata


__all__ = ["ImageDetectionResponse", "RiskLevelValue"]
