import math

import pytest
from geometry_engine.config import PseudoBEVConfig, ZoneBufferConfig, ZonesConfig
from geometry_engine.contracts import PseudoBEVPoint, PseudoBEVRectangle
from geometry_engine.zones import (
    build_load_footprint,
    build_load_zones,
    expand_footprint,
    rectangle_from_center_and_half_size,
)


def test_builds_load_footprint_from_finite_anchor_minimums_and_maximums():
    footprint = build_load_footprint(
        (
            PseudoBEVPoint(lateral=0.20, longitudinal=0.60),
            PseudoBEVPoint(lateral=-0.15, longitudinal=0.40),
            PseudoBEVPoint(lateral=0.05, longitudinal=0.90),
            PseudoBEVPoint(lateral=math.nan, longitudinal=0.50),
        ),
        load_bbox=(10.0, 20.0, 80.0, 90.0),
        image_width=100,
    )

    assert footprint is not None
    assert footprint.minimum_lateral == pytest.approx(-0.40)
    assert footprint.maximum_lateral == pytest.approx(0.30)
    assert footprint.minimum_longitudinal == pytest.approx(0.40)
    assert footprint.maximum_longitudinal == pytest.approx(0.90)


def test_requires_at_least_two_finite_anchors_for_a_footprint():
    arguments = {
        "load_bbox": (10.0, 20.0, 80.0, 90.0),
        "image_width": 100,
    }
    assert build_load_footprint((), **arguments) is None
    assert (
        build_load_footprint(
            (PseudoBEVPoint(lateral=0.1, longitudinal=0.2),),
            **arguments,
        )
        is None
    )


def test_applies_pseudo_bev_lateral_scale_to_bbox_bounds():
    footprint = build_load_footprint(
        (
            PseudoBEVPoint(lateral=0.0, longitudinal=0.25),
            PseudoBEVPoint(lateral=0.0, longitudinal=0.75),
        ),
        load_bbox=(25.0, 0.0, 75.0, 100.0),
        image_width=100,
        pseudo_bev_config=PseudoBEVConfig(lateral_scale=2.0),
    )

    assert footprint == PseudoBEVRectangle(-0.5, 0.5, 0.25, 0.75)


def test_creates_rectangle_from_center_and_half_size():
    rectangle = rectangle_from_center_and_half_size(
        center_lateral=0.125,
        center_longitudinal=0.5,
        half_lateral=0.375,
        half_longitudinal=0.25,
    )

    assert rectangle == PseudoBEVRectangle(-0.25, 0.5, 0.25, 0.75)
    assert rectangle.center_lateral == 0.125
    assert rectangle.center_longitudinal == 0.5
    assert rectangle.half_lateral == 0.375
    assert rectangle.half_longitudinal == 0.25


def test_expands_footprint_by_axis_buffers():
    footprint = PseudoBEVRectangle(-0.125, 0.375, 0.25, 0.75)

    expanded = expand_footprint(
        footprint,
        ZoneBufferConfig(lateral_ratio=0.5, longitudinal_ratio=1.0),
    )

    assert expanded == PseudoBEVRectangle(-0.25, 0.5, 0.0, 1.0)


def test_builds_nested_danger_and_warning_zones():
    zones = build_load_zones(
        (
            PseudoBEVPoint(lateral=-0.25, longitudinal=0.25),
            PseudoBEVPoint(lateral=0.25, longitudinal=0.75),
        ),
        load_bbox=(25.0, 0.0, 75.0, 100.0),
        image_width=100,
        config=ZonesConfig(
            danger=ZoneBufferConfig(lateral_ratio=0.5, longitudinal_ratio=0.5),
            warning=ZoneBufferConfig(lateral_ratio=1.0, longitudinal_ratio=1.0),
        ),
    )

    assert zones is not None
    assert zones.footprint == PseudoBEVRectangle(-0.25, 0.25, 0.25, 0.75)
    assert zones.danger == PseudoBEVRectangle(-0.375, 0.375, 0.125, 0.875)
    assert zones.warning == PseudoBEVRectangle(-0.75, 0.75, -0.25, 1.25)


def test_rectangle_corners_are_clockwise_from_lower_left():
    rectangle = PseudoBEVRectangle(-0.1, 0.2, 0.4, 0.8)

    assert rectangle.corners == (
        PseudoBEVPoint(-0.1, 0.4),
        PseudoBEVPoint(-0.1, 0.8),
        PseudoBEVPoint(0.2, 0.8),
        PseudoBEVPoint(0.2, 0.4),
    )
