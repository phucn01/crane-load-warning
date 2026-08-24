"""Validated and YAML-loadable policies for the risk engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import RiskLevel


def _fraction(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    minimum_person_confidence: float = 0.35
    boundary_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        _fraction("evaluation.minimum_person_confidence", self.minimum_person_confidence)
        _non_negative("evaluation.boundary_epsilon", self.boundary_epsilon)


@dataclass(frozen=True, slots=True)
class EventPolicy:
    warning_enter_frames: int = 2
    danger_enter_frames: int = 3
    exit_confirmation_seconds: float = 1.5
    grace_period_seconds: float = 0.5
    cooldown_seconds: float = 5.0

    def __post_init__(self) -> None:
        _positive_int("events.warning_enter_frames", self.warning_enter_frames)
        _positive_int("events.danger_enter_frames", self.danger_enter_frames)
        _non_negative("events.exit_confirmation_seconds", self.exit_confirmation_seconds)
        _non_negative("events.grace_period_seconds", self.grace_period_seconds)
        _non_negative("events.cooldown_seconds", self.cooldown_seconds)

    def confirmation_frames(self, level: RiskLevel) -> int:
        if level is RiskLevel.DANGER:
            return self.danger_enter_frames
        if level is RiskLevel.WARNING:
            return self.warning_enter_frames
        return 1


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    evaluation: EvaluationPolicy = EvaluationPolicy()
    events: EventPolicy = EventPolicy()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RiskPolicy:
        _reject_unknown("risk policy", payload, {"evaluation", "events"})
        evaluation = _section(payload, "evaluation")
        events = _section(payload, "events")
        _reject_unknown(
            "evaluation",
            evaluation,
            {"minimum_person_confidence", "boundary_epsilon"},
        )
        _reject_unknown(
            "events",
            events,
            {
                "warning_enter_frames",
                "danger_enter_frames",
                "exit_confirmation_seconds",
                "grace_period_seconds",
                "cooldown_seconds",
            },
        )
        default = cls()
        return cls(
            evaluation=EvaluationPolicy(
                minimum_person_confidence=float(
                    evaluation.get(
                        "minimum_person_confidence",
                        default.evaluation.minimum_person_confidence,
                    )
                ),
                boundary_epsilon=float(
                    evaluation.get("boundary_epsilon", default.evaluation.boundary_epsilon)
                ),
            ),
            events=EventPolicy(
                warning_enter_frames=int(
                    events.get("warning_enter_frames", default.events.warning_enter_frames)
                ),
                danger_enter_frames=int(
                    events.get("danger_enter_frames", default.events.danger_enter_frames)
                ),
                exit_confirmation_seconds=float(
                    events.get(
                        "exit_confirmation_seconds",
                        default.events.exit_confirmation_seconds,
                    )
                ),
                grace_period_seconds=float(
                    events.get("grace_period_seconds", default.events.grace_period_seconds)
                ),
                cooldown_seconds=float(
                    events.get("cooldown_seconds", default.events.cooldown_seconds)
                ),
            ),
        )


def load_risk_policy(path: str | Path) -> RiskPolicy:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise TypeError("risk policy root must be a YAML mapping")
    return RiskPolicy.from_mapping(payload)


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _reject_unknown(name: str, payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {', '.join(unknown)}")


__all__ = [
    "EvaluationPolicy",
    "EventPolicy",
    "RiskPolicy",
    "load_risk_policy",
]
