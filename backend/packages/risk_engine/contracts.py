"""Public, depth-independent contracts for safety risk assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    """The only risk levels exposed to consumers."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"

    @property
    def severity(self) -> int:
        return {RiskLevel.SAFE: 0, RiskLevel.WARNING: 1, RiskLevel.DANGER: 2}[self]


class EventScope(str, Enum):
    """What one temporal event key represents."""

    SCENE = "scene"
    PAIR = "pair"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class QualityReason(str, Enum):
    MISSING_PERSON_ANCHOR = "missing_person_anchor"
    INVALID_PERSON_ANCHOR = "invalid_person_anchor"
    UNRELIABLE_PERSON_ANCHOR = "unreliable_person_anchor"
    UNRELIABLE_PERSON_MASK = "unreliable_person_mask"
    LOW_PERSON_CONFIDENCE = "low_person_confidence"
    INVALID_PERSON_CONFIDENCE = "invalid_person_confidence"
    MISSING_ZONE_GEOMETRY = "missing_zone_geometry"


@dataclass(frozen=True, slots=True)
class Point2D:
    """A point in any consistent 2D coordinate system, normally image pixels."""

    x: float
    y: float

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y)


@dataclass(frozen=True, slots=True)
class Rectangle2D:
    """Axis-aligned bounds in a consistent 2D coordinate system."""

    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float

    def __post_init__(self) -> None:
        values = (self.minimum_x, self.maximum_x, self.minimum_y, self.maximum_y)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectangle bounds must be finite")
        if self.maximum_x < self.minimum_x or self.maximum_y < self.minimum_y:
            raise ValueError("rectangle maximums must not be less than minimums")


@dataclass(frozen=True, slots=True)
class ZoneGeometry:
    """Nested danger and warning rectangles belonging to one load."""

    danger: Rectangle2D
    warning: Rectangle2D


@dataclass(frozen=True, slots=True)
class PersonObservation:
    """Person data required by the 2D risk engine."""

    person_id: str
    anchor: Point2D | None
    confidence: float
    anchor_reliable: bool = True
    mask_reliable: bool | None = None
    quality_reasons: tuple[str, ...] = ()
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class SafetyAssessment:
    """Deterministic assessment for one person/load pair in one frame."""

    level: RiskLevel
    person_id: str
    load_id: str
    zone_geometry: ZoneGeometry | None
    confidence: float
    assessment_reliable: bool
    quality_reasons: tuple[str, ...] = ()
    matched_zone: RiskLevel | None = None
    person_track_id: str | None = None
    load_track_id: str | None = None


@dataclass(frozen=True, slots=True)
class RiskPairInput:
    """Inputs needed to evaluate one person/load pair in one frame."""

    person: PersonObservation
    load_id: str
    zones: ZoneGeometry | None
    load_track_id: str | None = None


@dataclass(frozen=True, slots=True)
class MediaFrameContext:
    """Identity and video timing shared by every uploaded media frame."""

    upload_id: str
    frame_id: str
    frame_index: int
    timestamp: float
    media_type: MediaType

    def __post_init__(self) -> None:
        if not self.upload_id:
            raise ValueError("upload_id must not be empty")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative value")
        if not isinstance(self.media_type, MediaType):
            raise TypeError("media_type must be a MediaType")

    @property
    def has_temporal_event(self) -> bool:
        return self.media_type is MediaType.VIDEO

    @property
    def scene_event_key(self) -> str:
        return f"upload:{self.upload_id}"


@dataclass(frozen=True, slots=True)
class FrameRiskAssessment:
    """Risk aggregated across every evaluated pair in one uploaded frame."""

    frame_id: str
    level: RiskLevel
    assessment_reliable: bool
    quality_reasons: tuple[str, ...]
    pair_assessments: tuple[SafetyAssessment, ...]
    contributing_person_ids: tuple[str, ...] = ()
    contributing_load_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskSignal:
    """Identity-free temporal input consumed by the event state machine."""

    level: RiskLevel
    assessment_reliable: bool
    quality_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventDecision:
    """Stable event state after applying temporal policies."""

    event_key: str
    event_scope: EventScope
    level: RiskLevel
    changed: bool = False
    alert_triggered: bool = False
    held_for_quality: bool = False
    technical_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskFrameResult:
    """Immediate frame assessment plus an optional temporal video decision."""

    assessment: FrameRiskAssessment
    event: EventDecision | None = None


__all__ = [
    "EventDecision",
    "EventScope",
    "FrameRiskAssessment",
    "MediaFrameContext",
    "MediaType",
    "PersonObservation",
    "Point2D",
    "QualityReason",
    "Rectangle2D",
    "RiskFrameResult",
    "RiskLevel",
    "RiskPairInput",
    "RiskSignal",
    "SafetyAssessment",
    "ZoneGeometry",
]
