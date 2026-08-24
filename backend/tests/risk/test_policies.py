from pathlib import Path

import pytest
from risk_engine import RiskLevel, RiskPolicy, load_risk_policy

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_loads_example_policy():
    policy = load_risk_policy(PROJECT_ROOT / "configs" / "risk-policy.example.yaml")

    assert policy.evaluation.unreliable_fallback_level is RiskLevel.WARNING
    assert policy.events.danger_enter_frames == 3


def test_rejects_safe_as_unreliable_fallback():
    with pytest.raises(ValueError, match="cannot be SAFE"):
        RiskPolicy.from_mapping(
            {"evaluation": {"unreliable_fallback_level": "SAFE"}}
        )


def test_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        RiskPolicy.from_mapping({"depth_tolerance": 0.2})
