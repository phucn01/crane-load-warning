"""Deterministic rectangle-based risk evaluation without depth input."""

from __future__ import annotations

import math

from .contracts import (
    PersonObservation,
    Point2D,
    QualityReason,
    Rectangle2D,
    RiskLevel,
    SafetyAssessment,
    ZoneGeometry,
)
from .policies import EvaluationPolicy


def point_in_zone(
    point: Point2D,
    zone: Rectangle2D,
    *,
    epsilon: float = 1e-9,
) -> bool:
    """Check whether a finite point is inside or on a rectangular zone boundary."""

    if not point.is_finite:
        return False

    within_x_bounds = (
        zone.minimum_x - epsilon <= point.x <= zone.maximum_x + epsilon
    )
    within_y_bounds = (
        zone.minimum_y - epsilon <= point.y <= zone.maximum_y + epsilon
    )
    return within_x_bounds and within_y_bounds


class RiskEvaluator:
    def __init__(self, policy: EvaluationPolicy | None = None) -> None:
        self.policy = policy or EvaluationPolicy()

    def evaluate(
        self,
        person: PersonObservation,
        *,
        load_id: str,
        zones: ZoneGeometry | None,
        load_track_id: str | None = None,
    ) -> SafetyAssessment:
        quality_reasons = _quality_reasons(
            person,
            zones,
            minimum_person_confidence=self.policy.minimum_person_confidence,
        )
        assessment_reliable = not quality_reasons
        matched_zone: RiskLevel | None = None
        if person.anchor is not None and person.anchor.is_finite and zones is not None:
            if point_in_zone(
                person.anchor,
                zones.danger,
                epsilon=self.policy.boundary_epsilon,
            ):
                matched_zone = RiskLevel.DANGER
            elif point_in_zone(
                person.anchor,
                zones.warning,
                epsilon=self.policy.boundary_epsilon,
            ):
                matched_zone = RiskLevel.WARNING

        geometry_level = matched_zone or RiskLevel.SAFE
        level = geometry_level
        if not assessment_reliable:
            level = max(
                (geometry_level, self.policy.unreliable_fallback_level),
                key=lambda candidate: candidate.severity,
            )

        confidence = person.confidence if math.isfinite(person.confidence) else 0.0
        confidence = min(1.0, max(0.0, confidence))
        return SafetyAssessment(
            level=level,
            person_id=person.person_id,
            load_id=load_id,
            zone_geometry=zones,
            confidence=confidence,
            assessment_reliable=assessment_reliable,
            quality_reasons=quality_reasons,
            matched_zone=matched_zone,
            person_track_id=person.track_id,
            load_track_id=load_track_id,
        )


def _quality_reasons(
    person: PersonObservation,
    zones: ZoneGeometry | None,
    *,
    minimum_person_confidence: float,
) -> tuple[str, ...]:
    reasons = list(person.quality_reasons)
    if zones is None:
        reasons.append(QualityReason.MISSING_ZONE_GEOMETRY.value)
    if person.anchor is None:
        reasons.append(QualityReason.MISSING_PERSON_ANCHOR.value)
    elif not person.anchor.is_finite:
        reasons.append(QualityReason.INVALID_PERSON_ANCHOR.value)
    if not person.anchor_reliable:
        reasons.append(QualityReason.UNRELIABLE_PERSON_ANCHOR.value)
    if person.mask_reliable is False:
        reasons.append(QualityReason.UNRELIABLE_PERSON_MASK.value)
    if not math.isfinite(person.confidence):
        reasons.append(QualityReason.INVALID_PERSON_CONFIDENCE.value)
    elif person.confidence < minimum_person_confidence:
        reasons.append(QualityReason.LOW_PERSON_CONFIDENCE.value)
    return tuple(dict.fromkeys(reasons))


__all__ = ["RiskEvaluator", "point_in_zone"]
