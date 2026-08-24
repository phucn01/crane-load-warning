import pytest
from risk_engine import (
    EventPolicy,
    EventScope,
    EventStateMachine,
    RiskLevel,
    RiskSignal,
)


def signal(
    level: RiskLevel,
    *,
    reliable: bool = True,
) -> RiskSignal:
    return RiskSignal(
        level=level,
        assessment_reliable=reliable,
        quality_reasons=() if reliable else ("missing_person_anchor",),
    )


def update(machine, level, timestamp, *, key="camera:01", reliable=True):
    return machine.update(
        signal(level, reliable=reliable),
        event_key=key,
        event_scope=EventScope.SCENE,
        timestamp=timestamp,
    )


def policy(**overrides) -> EventPolicy:
    values = {
        "warning_enter_frames": 2,
        "danger_enter_frames": 3,
        "exit_confirmation_seconds": 1.0,
        "grace_period_seconds": 0.5,
        "cooldown_seconds": 5.0,
    }
    values.update(overrides)
    return EventPolicy(**values)


def test_single_danger_frame_does_not_create_an_alert():
    machine = EventStateMachine(policy())

    decision = update(machine, RiskLevel.DANGER, 0.0)

    assert decision.level is RiskLevel.SAFE
    assert decision.alert_triggered is False


def test_danger_requires_confirmation_frames():
    machine = EventStateMachine(policy())

    assert not update(machine, RiskLevel.DANGER, 0.0).alert_triggered
    assert not update(machine, RiskLevel.DANGER, 0.1).alert_triggered
    decision = update(machine, RiskLevel.DANGER, 0.2)

    assert decision.level is RiskLevel.DANGER
    assert decision.changed is True
    assert decision.alert_triggered is True


def test_short_detection_loss_holds_current_event_during_grace_period():
    machine = EventStateMachine(policy(danger_enter_frames=1))
    update(machine, RiskLevel.DANGER, 0.0)

    decision = update(machine, RiskLevel.WARNING, 0.2, reliable=False)

    assert decision.level is RiskLevel.DANGER
    assert decision.held_for_quality is True
    assert decision.technical_reasons == ("missing_person_anchor",)


def test_exit_requires_continuous_confirmation_time():
    machine = EventStateMachine(policy(danger_enter_frames=1))
    update(machine, RiskLevel.DANGER, 0.0)

    assert update(machine, RiskLevel.SAFE, 1.0).level is RiskLevel.DANGER
    assert update(machine, RiskLevel.SAFE, 1.9).level is RiskLevel.DANGER
    decision = update(machine, RiskLevel.SAFE, 2.0)

    assert decision.level is RiskLevel.SAFE
    assert decision.changed is True


def test_confidence_oscillation_cannot_accumulate_confirmation_frames():
    machine = EventStateMachine(policy(warning_enter_frames=2))

    update(machine, RiskLevel.WARNING, 0.0)
    update(machine, RiskLevel.WARNING, 0.1, reliable=False)
    decision = update(machine, RiskLevel.WARNING, 0.2)

    assert decision.level is RiskLevel.SAFE


def test_event_keys_have_independent_state():
    machine = EventStateMachine(policy(danger_enter_frames=1))

    first = update(machine, RiskLevel.DANGER, 0.0, key="camera:01")
    second = update(machine, RiskLevel.SAFE, 0.0, key="camera:02")

    assert first.level is RiskLevel.DANGER
    assert second.level is RiskLevel.SAFE


def test_rejects_scope_change_for_existing_event_key():
    machine = EventStateMachine(policy())
    update(machine, RiskLevel.SAFE, 0.0)

    with pytest.raises(ValueError, match="event_scope cannot change"):
        machine.update(
            signal(RiskLevel.SAFE),
            event_key="camera:01",
            event_scope=EventScope.PAIR,
            timestamp=0.1,
        )
