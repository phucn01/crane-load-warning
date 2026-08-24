import pytest
from risk_engine import (
    PersonObservation,
    Point2D,
    Rectangle2D,
    RiskEvaluator,
    RiskLevel,
    ZoneGeometry,
    point_in_zone,
)


def rectangle(x1: float, y1: float, x2: float, y2: float) -> Rectangle2D:
    return Rectangle2D(x1, x2, y1, y2)


@pytest.fixture
def zones() -> ZoneGeometry:
    return ZoneGeometry(
        danger=rectangle(25.0, 25.0, 75.0, 75.0),
        warning=rectangle(0.0, 0.0, 100.0, 100.0),
    )


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (Point2D(50.0, 50.0), True),
        (Point2D(0.0, 50.0), True),
        (Point2D(0.0, 0.0), True),
        (Point2D(-0.01, 50.0), False),
    ],
)
def test_point_in_zone_includes_edges_and_corners(point: Point2D, expected: bool):
    assert point_in_zone(point, rectangle(0.0, 0.0, 100.0, 100.0)) is expected


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Point2D(150.0, 50.0), RiskLevel.SAFE),
        (Point2D(10.0, 50.0), RiskLevel.WARNING),
        (Point2D(50.0, 50.0), RiskLevel.DANGER),
        (Point2D(25.0, 50.0), RiskLevel.DANGER),
    ],
)
def test_evaluates_nested_zones(anchor: Point2D, expected: RiskLevel, zones):
    assessment = RiskEvaluator().evaluate(
        PersonObservation("person-1", anchor, confidence=0.9),
        load_id="load-1",
        zones=zones,
    )

    assert assessment.level is expected
    assert assessment.assessment_reliable is True


def test_unreliable_input_never_creates_a_safe_assessment(zones):
    assessment = RiskEvaluator().evaluate(
        PersonObservation("person-1", None, confidence=0.9),
        load_id="load-1",
        zones=zones,
    )

    assert assessment.level is RiskLevel.WARNING
    assert assessment.assessment_reliable is False
    assert assessment.quality_reasons == ("missing_person_anchor",)


def test_low_confidence_preserves_a_geometric_danger(zones):
    assessment = RiskEvaluator().evaluate(
        PersonObservation("person-1", Point2D(50.0, 50.0), confidence=0.1),
        load_id="load-1",
        zones=zones,
    )

    assert assessment.level is RiskLevel.DANGER
    assert assessment.matched_zone is RiskLevel.DANGER
    assert assessment.quality_reasons == ("low_person_confidence",)


def test_rejects_reversed_rectangle_bounds():
    with pytest.raises(ValueError, match="maximums"):
        Rectangle2D(10.0, 0.0, 0.0, 10.0)


def test_same_input_always_returns_same_assessment(zones):
    evaluator = RiskEvaluator()
    person = PersonObservation("person-1", Point2D(10.0, 20.0), confidence=0.8)

    first = evaluator.evaluate(person, load_id="load-1", zones=zones)
    second = evaluator.evaluate(person, load_id="load-1", zones=zones)

    assert first == second
