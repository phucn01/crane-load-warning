"""Temporal filtering for stable scene or tracked-pair safety events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import EventDecision, EventScope, RiskLevel, RiskSignal
from .policies import EventPolicy


@dataclass(slots=True)
class _EventState:
    event_scope: EventScope = EventScope.SCENE
    level: RiskLevel = RiskLevel.SAFE
    pending_level: RiskLevel | None = None
    pending_frames: int = 0
    pending_since: float | None = None
    unreliable_since: float | None = None
    last_alert_at: float | None = None
    last_timestamp: float | None = None


class EventStateMachine:
    """Keep independent deterministic state for generic scene or pair keys."""

    def __init__(self, policy: EventPolicy | None = None) -> None:
        self.policy = policy or EventPolicy()
        self._states: dict[str, _EventState] = {}

    def update(
        self,
        signal: RiskSignal,
        *,
        event_key: str,
        event_scope: EventScope,
        timestamp: float,
    ) -> EventDecision:
        if not event_key:
            raise ValueError("event_key must not be empty")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        state = self._states.get(event_key)
        if state is None:
            state = _EventState(event_scope=event_scope)
            self._states[event_key] = state
        elif state.event_scope is not event_scope:
            raise ValueError("event_scope cannot change for an existing event_key")
        if state.last_timestamp is not None and timestamp < state.last_timestamp:
            raise ValueError("timestamps must be non-decreasing for each event_key")
        state.last_timestamp = timestamp

        reasons = signal.quality_reasons
        if not signal.assessment_reliable:
            if state.unreliable_since is None:
                state.unreliable_since = timestamp
            if timestamp - state.unreliable_since <= self.policy.grace_period_seconds:
                self._clear_pending(state)
                return self._decision(
                    event_key,
                    state,
                    held_for_quality=True,
                    technical_reasons=reasons,
                )
        else:
            state.unreliable_since = None

        target = signal.level
        if target is state.level:
            self._clear_pending(state)
            return self._decision(
                event_key,
                state,
                technical_reasons=reasons,
            )

        if target.severity > state.level.severity:
            changed = self._confirm_entry(state, target)
        else:
            changed = self._confirm_exit(state, target, timestamp)

        alert = False
        if (
            changed
            and state.level is RiskLevel.DANGER
            and (
                state.last_alert_at is None
                or timestamp - state.last_alert_at >= self.policy.cooldown_seconds
            )
        ):
            state.last_alert_at = timestamp
            alert = True
        return self._decision(
            event_key,
            state,
            changed=changed,
            alert_triggered=alert,
            technical_reasons=reasons,
        )

    def current_level(self, event_key: str) -> RiskLevel:
        state = self._states.get(event_key)
        return state.level if state is not None else RiskLevel.SAFE

    def reset(self, event_key: str | None = None) -> None:
        """Clear all state or one exact event key."""

        if event_key is None:
            self._states.clear()
            return
        self._states.pop(event_key, None)

    def _confirm_entry(self, state: _EventState, target: RiskLevel) -> bool:
        if state.pending_level is target:
            state.pending_frames += 1
        else:
            state.pending_level = target
            state.pending_frames = 1
            state.pending_since = None
        if state.pending_frames < self.policy.confirmation_frames(target):
            return False
        state.level = target
        self._clear_pending(state)
        return True

    def _confirm_exit(
        self,
        state: _EventState,
        target: RiskLevel,
        timestamp: float,
    ) -> bool:
        if state.pending_level is not target:
            state.pending_level = target
            state.pending_frames = 0
            state.pending_since = timestamp
        assert state.pending_since is not None
        if timestamp - state.pending_since < self.policy.exit_confirmation_seconds:
            return False
        state.level = target
        self._clear_pending(state)
        return True

    @staticmethod
    def _clear_pending(state: _EventState) -> None:
        state.pending_level = None
        state.pending_frames = 0
        state.pending_since = None

    @staticmethod
    def _decision(
        event_key: str,
        state: _EventState,
        *,
        changed: bool = False,
        alert_triggered: bool = False,
        held_for_quality: bool = False,
        technical_reasons: tuple[str, ...] = (),
    ) -> EventDecision:
        return EventDecision(
            event_key=event_key,
            event_scope=state.event_scope,
            level=state.level,
            changed=changed,
            alert_triggered=alert_triggered,
            held_for_quality=held_for_quality,
            technical_reasons=technical_reasons,
        )


__all__ = ["EventStateMachine"]
