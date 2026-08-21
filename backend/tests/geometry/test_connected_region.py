from dataclasses import replace

import pytest
from geometry_engine.config import ConnectedRegionConfig
from geometry_engine.connected_region import find_connected_candidate_region
from geometry_engine.contracts import ImagePoint, LoadAnchorCandidate

ROOT_BBOX = (-1.0, -1.0, 10.0, 10.0)


def make_candidate(
    grid_x: int,
    grid_y: int,
    depth: float,
    *,
    is_seed_consistent: bool = True,
) -> LoadAnchorCandidate:
    return LoadAnchorCandidate(
        candidate_id=f"candidate_{grid_y}_{grid_x}",
        grid_x=grid_x,
        grid_y=grid_y,
        point=ImagePoint(x=float(grid_x), y=float(grid_y)),
        patch_bbox=(grid_x, grid_y, grid_x + 1, grid_y + 1),
        depth=depth,
        valid_depth_count=1,
        valid_depth_fraction=1.0,
        seed_depth_difference=abs(depth - 1.0) / max(abs(depth), 1.0),
        is_seed_consistent=is_seed_consistent,
    )


def candidate_ids(region):
    return {candidate.candidate_id for candidate in region}


def test_grows_through_diagonal_seed_consistent_neighbors():
    candidates = [
        make_candidate(0, 0, 1.00),
        make_candidate(1, 1, 1.03),
        make_candidate(2, 1, 1.05),
    ]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=ROOT_BBOX,
        config=ConnectedRegionConfig(
            neighbor_radius=1,
            local_neighbor_depth_tolerance=0.05,
        ),
    )

    assert tuple(region) == tuple(candidates)


def test_uses_precomputed_seed_consistency_filter():
    candidates = [
        make_candidate(0, 0, 1.00),
        make_candidate(1, 0, 1.04),
        make_candidate(2, 0, 1.08),
        make_candidate(3, 0, 1.12, is_seed_consistent=False),
    ]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=ROOT_BBOX,
        config=ConnectedRegionConfig(
            local_neighbor_depth_tolerance=0.05,
        ),
    )

    assert candidate_ids(region) == {
        "candidate_0_0",
        "candidate_0_1",
        "candidate_0_2",
    }


def test_local_gate_rejects_abrupt_seed_compatible_neighbor():
    candidates = [
        make_candidate(0, 0, 1.00),
        make_candidate(1, 0, 1.09),
    ]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=ROOT_BBOX,
        config=ConnectedRegionConfig(
            local_neighbor_depth_tolerance=0.05,
        ),
    )

    assert candidate_ids(region) == {"candidate_0_0"}


def test_returns_only_component_connected_to_best_seed_candidate():
    candidates = [
        make_candidate(3, 0, 1.02),
        make_candidate(0, 0, 1.00),
        make_candidate(4, 0, 1.03),
    ]

    region = find_connected_candidate_region(candidates, root_bbox=ROOT_BBOX)

    assert candidate_ids(region) == {"candidate_0_0"}


def test_neighbor_radius_can_bridge_one_missing_grid_position():
    candidates = [
        make_candidate(0, 0, 1.00),
        make_candidate(2, 0, 1.02),
    ]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=ROOT_BBOX,
        config=ConnectedRegionConfig(neighbor_radius=2),
    )

    assert tuple(region) == tuple(candidates)


def test_returns_empty_region_when_no_candidate_is_seed_consistent():
    candidate = make_candidate(
        0,
        0,
        1.0,
        is_seed_consistent=False,
    )

    region = find_connected_candidate_region([candidate], root_bbox=ROOT_BBOX)

    assert region == ()


def test_rejects_duplicate_grid_coordinates():
    duplicate = replace(make_candidate(0, 0, 1.0), candidate_id="duplicate")

    with pytest.raises(ValueError, match="unique grid coordinates"):
        find_connected_candidate_region(
            [make_candidate(0, 0, 1.0), duplicate],
            root_bbox=ROOT_BBOX,
        )


def test_selects_root_only_from_inside_root_bbox_then_grows_outward():
    candidates = [
        make_candidate(0, 0, 1.00),
        make_candidate(2, 0, 1.02),
        make_candidate(3, 0, 1.03),
    ]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=(1.5, -0.5, 2.5, 0.5),
    )

    assert candidate_ids(region) == {"candidate_0_2", "candidate_0_3"}


def test_returns_empty_when_inner_roi_has_no_seed_consistent_candidate():
    candidates = [make_candidate(0, 0, 1.0)]

    region = find_connected_candidate_region(
        candidates,
        root_bbox=(2.0, 2.0, 3.0, 3.0),
    )

    assert region == ()
