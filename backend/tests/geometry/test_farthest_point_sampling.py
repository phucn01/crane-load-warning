import pytest
from geometry_engine.config import FarthestPointSamplingConfig
from geometry_engine.contracts import ImagePoint, LoadAnchorCandidate
from geometry_engine.farthest_point_sampling import select_farthest_load_anchors


def make_candidate(
    x: float,
    seed_depth_difference: float,
    candidate_id: str,
) -> LoadAnchorCandidate:
    return LoadAnchorCandidate(
        candidate_id=candidate_id,
        grid_x=int(x),
        grid_y=0,
        point=ImagePoint(x=x, y=0.0),
        patch_bbox=(int(x), 0, int(x) + 1, 1),
        depth=1.0,
        valid_depth_count=1,
        valid_depth_fraction=1.0,
        seed_depth_difference=seed_depth_difference,
        is_seed_consistent=True,
    )


def test_starts_nearest_seed_then_spreads_across_region():
    candidates = [
        make_candidate(0.0, 0.2, "left"),
        make_candidate(5.0, 0.0, "root"),
        make_candidate(10.0, 0.1, "right"),
    ]

    selected = select_farthest_load_anchors(
        candidates,
        config=FarthestPointSamplingConfig(maximum_anchors=3),
    )

    assert [candidate.candidate_id for candidate in selected] == [
        "root",
        "left",
        "right",
    ]


def test_respects_maximum_anchor_count():
    candidates = [make_candidate(float(x), float(x), str(x)) for x in range(5)]

    selected = select_farthest_load_anchors(
        candidates,
        config=FarthestPointSamplingConfig(maximum_anchors=2),
    )

    assert len(selected) == 2


def test_stops_when_minimum_distance_cannot_be_met():
    candidates = [
        make_candidate(0.0, 0.0, "root"),
        make_candidate(2.0, 0.1, "near"),
        make_candidate(4.0, 0.2, "far"),
    ]

    selected = select_farthest_load_anchors(
        candidates,
        config=FarthestPointSamplingConfig(
            maximum_anchors=3,
            minimum_distance=3.0,
        ),
    )

    assert [candidate.candidate_id for candidate in selected] == ["root", "far"]


def test_empty_candidates_return_empty_tuple():
    assert select_farthest_load_anchors([]) == ()


def test_rejects_partial_selection_below_minimum_anchor_count():
    candidate = make_candidate(5.0, 0.0, "only")

    assert select_farthest_load_anchors([candidate]) == ()


def test_minimum_anchor_count_cannot_exceed_maximum():
    with pytest.raises(ValueError, match="minimum_anchors must not exceed"):
        FarthestPointSamplingConfig(minimum_anchors=3, maximum_anchors=2)


def test_equal_distance_tie_break_is_independent_of_input_order():
    root = make_candidate(5.0, 0.0, "root")
    left = make_candidate(0.0, 0.1, "left")
    right = make_candidate(10.0, 0.2, "right")
    config = FarthestPointSamplingConfig(maximum_anchors=2)

    forward = select_farthest_load_anchors(
        [root, left, right],
        config=config,
    )
    reversed_order = select_farthest_load_anchors(
        [right, left, root],
        config=config,
    )

    assert [candidate.candidate_id for candidate in forward] == ["root", "left"]
    assert [candidate.candidate_id for candidate in reversed_order] == [
        "root",
        "left",
    ]
