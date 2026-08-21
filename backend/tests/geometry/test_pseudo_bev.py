import math

import pytest
from geometry_engine.config import PseudoBEVConfig
from geometry_engine.contracts import (
    ImagePoint,
    LoadAnchorCandidate,
    PersonRepresentative,
    PseudoBEVPoint,
    RepresentativeDepth,
)
from geometry_engine.pseudo_bev import (
    project_image_point_to_pseudo_bev,
    project_load_anchors_to_pseudo_bev,
    project_person_to_pseudo_bev,
    relative_depth_to_forward,
)


def make_anchor(candidate_id: str, x: float, depth: float) -> LoadAnchorCandidate:
    return LoadAnchorCandidate(
        candidate_id=candidate_id,
        grid_x=0,
        grid_y=0,
        point=ImagePoint(x=x, y=40.0),
        patch_bbox=(0, 0, 1, 1),
        depth=depth,
        valid_depth_count=1,
        valid_depth_fraction=1.0,
    )


def test_projects_center_point_onto_zero_lateral_axis():
    projected = project_image_point_to_pseudo_bev(
        ImagePoint(x=100.0, y=50.0),
        4.0,
        image_width=200,
        depth_low=0.0,
        depth_high=8.0,
    )

    assert projected == PseudoBEVPoint(lateral=0.0, longitudinal=0.5)


def test_projects_centered_horizontal_offset_and_applies_axis_scales():
    projected = project_image_point_to_pseudo_bev(
        ImagePoint(x=50.0, y=50.0),
        4.0,
        image_width=200,
        depth_low=0.0,
        depth_high=8.0,
        config=PseudoBEVConfig(lateral_scale=2.0, longitudinal_scale=3.0),
    )

    assert projected == PseudoBEVPoint(lateral=-0.5, longitudinal=1.0)


def test_depth_below_scene_range_clamps_to_zero():
    projected = project_image_point_to_pseudo_bev(
        ImagePoint(x=0.0, y=50.0),
        -2.0,
        image_width=100,
        depth_low=0.0,
        depth_high=1.0,
    )

    assert projected == PseudoBEVPoint(lateral=-0.5, longitudinal=0.0)


@pytest.mark.parametrize(
    ("x", "depth", "expected"),
    [
        (-20.0, -1.0, PseudoBEVPoint(lateral=-0.7, longitudinal=0.0)),
        (120.0, 3.0, PseudoBEVPoint(lateral=0.7, longitudinal=1.0)),
    ],
)
def test_preserves_centered_lateral_offset_and_clamps_longitudinal(
    x: float,
    depth: float,
    expected: PseudoBEVPoint,
):
    projected = project_image_point_to_pseudo_bev(
        ImagePoint(x=x, y=50.0),
        depth,
        image_width=100,
        depth_low=0.0,
        depth_high=1.0,
    )

    assert projected == expected


def test_returns_none_for_unavailable_or_non_finite_input():
    assert (
        project_image_point_to_pseudo_bev(
            ImagePoint(x=10.0, y=20.0),
            None,
            image_width=100,
            depth_low=0.0,
            depth_high=1.0,
        )
        is None
    )
    assert (
        project_image_point_to_pseudo_bev(
            ImagePoint(x=math.nan, y=20.0),
            1.0,
            image_width=100,
            depth_low=0.0,
            depth_high=1.0,
        )
        is None
    )
    assert (
        project_image_point_to_pseudo_bev(
            ImagePoint(x=10.0, y=20.0),
            math.inf,
            image_width=100,
            depth_low=0.0,
            depth_high=1.0,
        )
        is None
    )


def test_projects_load_anchors_in_input_order_and_skips_invalid_depth():
    projected = project_load_anchors_to_pseudo_bev(
        (
            make_anchor("right", 75.0, 2.0),
            make_anchor("invalid", 50.0, math.nan),
            make_anchor("left", 25.0, 4.0),
        ),
        image_width=100,
        depth_low=0.0,
        depth_high=4.0,
    )

    assert projected == (
        PseudoBEVPoint(lateral=0.25, longitudinal=0.5),
        PseudoBEVPoint(lateral=-0.25, longitudinal=1.0),
    )


def test_projects_person_representative_and_handles_missing_depth():
    person = PersonRepresentative(
        point=ImagePoint(x=25.0, y=90.0),
        point_source="bbox_bottom_center",
        depth=RepresentativeDepth(value=2.0, source="full", quality="low"),
    )
    unavailable_person = PersonRepresentative(
        point=person.point,
        point_source=person.point_source,
        depth=RepresentativeDepth(
            value=None,
            source="unavailable",
            quality="unavailable",
        ),
    )

    assert project_person_to_pseudo_bev(
        person,
        image_width=100,
        depth_low=0.0,
        depth_high=4.0,
    ) == PseudoBEVPoint(
        lateral=-0.25,
        longitudinal=0.5,
    )
    assert (
        project_person_to_pseudo_bev(
            unavailable_person,
            image_width=100,
            depth_low=0.0,
            depth_high=4.0,
        )
        is None
    )


@pytest.mark.parametrize("image_width", [0, -1])
def test_rejects_non_positive_image_width(image_width: int):
    with pytest.raises(ValueError, match="image_width must be greater than zero"):
        project_image_point_to_pseudo_bev(
            ImagePoint(x=0.0, y=0.0),
            1.0,
            image_width=image_width,
            depth_low=0.0,
            depth_high=1.0,
        )


def test_rejects_non_integer_image_width():
    with pytest.raises(TypeError, match="image_width must be an integer"):
        project_image_point_to_pseudo_bev(
            ImagePoint(x=0.0, y=0.0),
            1.0,
            image_width=100.0,  # type: ignore[arg-type]
            depth_low=0.0,
            depth_high=1.0,
        )


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (2.0, 0.0),
        (4.0, 0.5),
        (6.0, 1.0),
        (8.0, 1.0),
    ],
)
def test_normalizes_relative_depth_to_forward(depth: float, expected: float):
    assert relative_depth_to_forward(depth, depth_low=2.0, depth_high=6.0) == expected


def test_relative_depth_to_forward_returns_nan_for_non_finite_depth():
    assert math.isnan(relative_depth_to_forward(math.nan, 0.0, 1.0))
