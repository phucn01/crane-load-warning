import numpy as np
from geometry_engine.config import LoadAnchorsConfig, RepresentativeDepthConfig
from geometry_engine.contracts import ImagePoint, LoadAnchorCandidate
from geometry_engine.load_anchors import (
    build_load_anchor_candidates,
    filter_candidates_by_seed_depth,
    generate_candidate_patches,
)


def make_load_detection(bbox=(0.0, 0.0, 15.0, 15.0)):
    x1, y1, x2, y2 = bbox
    return {
        "source_model": "test",
        "class_id": 0,
        "class_name": "hanging_object",
        "confidence": 0.9,
        "bbox": bbox,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "mask": None,
    }


def make_candidate(candidate_id: str, depth: float) -> LoadAnchorCandidate:
    return LoadAnchorCandidate(
        candidate_id=candidate_id,
        grid_x=0,
        grid_y=0,
        point=ImagePoint(x=2.0, y=2.0),
        patch_bbox=(0, 0, 5, 5),
        depth=depth,
        valid_depth_count=25,
        valid_depth_fraction=1.0,
    )


def test_generates_regular_candidate_grid_inside_load_bbox():
    depth = np.full((30, 30), 2.0, dtype=np.float32)
    config = LoadAnchorsConfig(
        patch_size=11,
        patch_stride=12,
    )

    candidates, generated_count = generate_candidate_patches(
        make_load_detection((0.0, 0.0, 30.0, 30.0)),
        depth,
        config=config,
    )

    assert generated_count == 4
    assert len(candidates) == 4
    assert candidates[0].point == ImagePoint(x=6.0, y=6.0)
    assert candidates[-1].point == ImagePoint(x=18.0, y=18.0)
    assert all(
        candidate.patch_bbox[2] - candidate.patch_bbox[0] == 11
        for candidate in candidates
    )
    assert all(candidate.depth == 2.0 for candidate in candidates)


def test_rejects_patch_without_finite_depth_values():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    config = LoadAnchorsConfig(
        patch_size=11,
        patch_stride=12,
    )

    candidates, generated_count = generate_candidate_patches(
        make_load_detection((0.0, 0.0, 10.0, 10.0)),
        depth,
        config=config,
    )

    assert generated_count == 1
    assert candidates == []


def test_accepts_patch_with_one_finite_depth_value():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[4, 4] = 1.25

    candidates, generated_count = generate_candidate_patches(
        make_load_detection((0.0, 0.0, 10.0, 10.0)),
        depth,
        config=LoadAnchorsConfig(patch_size=11, patch_stride=12),
    )

    assert generated_count == 1
    assert len(candidates) == 1
    assert candidates[0].depth == 1.25
    assert candidates[0].valid_depth_count == 1


def test_uses_median_of_all_finite_patch_depths():
    depth = np.full((12, 12), np.nan, dtype=np.float32)
    depth[1:12, 1:12] = np.arange(121, dtype=np.float32).reshape(11, 11)
    config = LoadAnchorsConfig(
        patch_size=11,
        patch_stride=12,
    )

    candidates, generated_count = generate_candidate_patches(
        make_load_detection((0.0, 0.0, 12.0, 12.0)),
        depth,
        config=config,
    )

    assert generated_count == 1
    assert len(candidates) == 1
    assert candidates[0].depth == 60.0
    assert candidates[0].valid_depth_count == 121
    assert candidates[0].patch_bbox == (1, 1, 12, 12)


def test_filters_candidates_by_symmetric_relative_seed_error():
    candidates = [
        make_candidate("near", 1.05),
        make_candidate("far", 1.50),
    ]

    annotated_candidates = filter_candidates_by_seed_depth(
        candidates,
        seed_depth=1.0,
        seed_depth_tolerance=0.10,
    )

    assert annotated_candidates[0].is_seed_consistent is True
    assert np.isclose(annotated_candidates[0].seed_depth_difference, 0.05 / 1.05)
    assert annotated_candidates[1].is_seed_consistent is False
    assert np.isclose(annotated_candidates[1].seed_depth_difference, 0.5 / 1.5)


def test_builds_candidates_with_inner_roi_seed_and_filter_summary():
    depth = np.full((15, 15), 2.0, dtype=np.float32)
    result = build_load_anchor_candidates(
        make_load_detection(),
        depth,
        representative_config=RepresentativeDepthConfig(
            minimum_valid_pixels=1,
            load_inner_size_fraction=0.2,
        ),
        anchors_config=LoadAnchorsConfig(
            patch_size=5,
            patch_stride=5,
            seed_depth_tolerance=0.1,
        ),
    )

    assert result.seed_depth == 2.0
    assert result.seed_source == "inner"
    assert result.generated_patch_count == 9
    assert result.rejected_invalid_depth_count == 0
    assert len(result.candidates) == 9
    assert len(result.consistent_candidates) == 9
