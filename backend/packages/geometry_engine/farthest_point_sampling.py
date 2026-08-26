"""Select spatially distributed load anchors from a connected candidate region."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import FarthestPointSamplingConfig
from .contracts import LoadAnchorCandidate


def select_farthest_load_anchors(
    candidates: Sequence[LoadAnchorCandidate],
    *,
    config: FarthestPointSamplingConfig | None = None,
) -> tuple[LoadAnchorCandidate, ...]:
    """Select deterministic anchors with greedy farthest-point sampling.

    Selection starts at the candidate closest to the representative seed depth.
    Every later anchor maximizes its minimum image-space distance to the anchors
    already selected. Selection stops at ``maximum_anchors`` or when no remaining
    candidate satisfies ``minimum_distance``. A partial result below
    ``minimum_anchors`` is rejected instead of fabricating load anchors.
    """

    settings = config or FarthestPointSamplingConfig()
    if not candidates:
        return ()

    first_anchor = min(
        candidates,
        key=lambda candidate: (
            _finite_difference_or_infinity(candidate.seed_depth_difference),
            candidate.point.y,
            candidate.point.x,
            candidate.candidate_id,
        ),
    )
    selected_anchors = [first_anchor]
    remaining_candidates = [
        candidate for candidate in candidates if candidate is not first_anchor
    ]

    while remaining_candidates and len(selected_anchors) < settings.maximum_anchors:
        next_candidate, distance_to_nearest_anchor = min(
            (
                (
                    candidate,
                    min(
                        _point_distance(candidate, anchor)
                        for anchor in selected_anchors
                    ),
                )
                for candidate in remaining_candidates
            ),
            key=lambda item: (
                -item[1],
                item[0].point.y,
                item[0].point.x,
                item[0].candidate_id,
            ),
        )
        if distance_to_nearest_anchor < settings.minimum_distance:
            break
        selected_anchors.append(next_candidate)
        remaining_candidates.remove(next_candidate)

    if len(selected_anchors) < settings.minimum_anchors:
        return ()
    return tuple(selected_anchors)


def _point_distance(
    first_candidate: LoadAnchorCandidate,
    second_candidate: LoadAnchorCandidate,
) -> float:
    return math.hypot(
        first_candidate.point.x - second_candidate.point.x,
        first_candidate.point.y - second_candidate.point.y,
    )


def _finite_difference_or_infinity(difference: float | None) -> float:
    if difference is None or not math.isfinite(difference):
        return math.inf
    return difference


__all__ = ["select_farthest_load_anchors"]
