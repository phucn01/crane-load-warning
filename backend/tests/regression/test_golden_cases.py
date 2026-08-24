import json
from pathlib import Path

import pytest
from risk_engine import (
    PersonObservation,
    Point2D,
    Rectangle2D,
    RiskEvaluator,
    RiskLevel,
    ZoneGeometry,
)

CASES_DIR = Path(__file__).with_name("golden_cases")


def rectangle(values: dict[str, float]) -> Rectangle2D:
    return Rectangle2D(**values)


@pytest.mark.parametrize("case_path", sorted(CASES_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_saved_intermediate_golden_case(case_path: Path):
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    zones = ZoneGeometry(
        danger=rectangle(payload["zones"]["danger"]),
        warning=rectangle(payload["zones"]["warning"]),
    )
    person_data = payload["person"]
    anchor_data = person_data["anchor"]
    anchor = None if anchor_data is None else Point2D(*anchor_data)
    assessment = RiskEvaluator().evaluate(
        PersonObservation(
            person_id=person_data["id"],
            anchor=anchor,
            confidence=person_data["confidence"],
            anchor_reliable=person_data.get("anchor_reliable", True),
            mask_reliable=person_data.get("mask_reliable"),
        ),
        load_id=payload["load_id"],
        zones=zones,
    )

    expected = payload["expected"]
    assert assessment.level is RiskLevel(expected["level"])
    assert assessment.assessment_reliable is expected["assessment_reliable"]
