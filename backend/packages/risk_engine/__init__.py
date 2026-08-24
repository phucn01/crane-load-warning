"""Depth-independent, deterministic crane safety risk engine."""

from .contracts import (
    EventDecision,
    EventScope,
    FrameRiskAssessment,
    MediaFrameContext,
    MediaType,
    PersonObservation,
    Point2D,
    QualityReason,
    Rectangle2D,
    RiskFrameResult,
    RiskLevel,
    RiskPairInput,
    RiskSignal,
    SafetyAssessment,
    ZoneGeometry,
)
from .evaluator import RiskEvaluator, point_in_zone
from .event_state_machine import EventStateMachine
from .frame_pipeline import RiskFramePipeline, aggregate_frame_risk
from .policies import EvaluationPolicy, EventPolicy, RiskPolicy, load_risk_policy

__all__ = [
    "EvaluationPolicy",
    "EventDecision",
    "EventPolicy",
    "EventScope",
    "EventStateMachine",
    "FrameRiskAssessment",
    "MediaFrameContext",
    "MediaType",
    "PersonObservation",
    "Point2D",
    "QualityReason",
    "Rectangle2D",
    "RiskEvaluator",
    "RiskFramePipeline",
    "RiskFrameResult",
    "RiskLevel",
    "RiskPairInput",
    "RiskPolicy",
    "RiskSignal",
    "SafetyAssessment",
    "ZoneGeometry",
    "aggregate_frame_risk",
    "load_risk_policy",
    "point_in_zone",
]
