"""Grow a spatially connected load region through candidate-grid neighbors."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from .config import ConnectedRegionConfig
from .contracts import LoadAnchorCandidate
from .representative_depth import relative_depth_difference

GridKey = tuple[int, int]


def find_connected_candidate_region(
    candidates: Sequence[LoadAnchorCandidate],
    *,
    root_bbox: tuple[float, float, float, float],
    config: ConnectedRegionConfig | None = None,
    epsilon: float = 1e-8,
) -> tuple[LoadAnchorCandidate, ...]:
    """Return the connected component selected from seed-consistent candidates.

    Seed-depth filtering is performed upstream by ``load_anchors``. This function
    uses only candidates whose precomputed ``is_seed_consistent`` flag is true.
    The root is selected only from eligible candidates inside ``root_bbox`` and
    has the smallest ``seed_depth_difference``. Region growing may then expand
    beyond that ROI by following grid neighbors within ``neighbor_radius`` whose
    depth is locally consistent with the current candidate.

    The returned candidates keep their original input order.
    """

    settings = config or ConnectedRegionConfig()
    _validate_epsilon(epsilon)
    _validate_bbox(root_bbox)

    candidates_by_grid = _index_candidates(candidates)
    if not candidates_by_grid:
        return ()

    eligible_keys = {
        key
        for key, candidate in candidates_by_grid.items()
        if candidate.is_seed_consistent
    }
    if not eligible_keys:
        return ()

    root_keys = {
        key
        for key in eligible_keys
        if _point_is_inside_bbox(candidates_by_grid[key], root_bbox)
    }
    if not root_keys:
        return ()

    root_key = min(
        root_keys,
        key=lambda key: (
            _finite_difference_or_infinity(
                candidates_by_grid[key].seed_depth_difference
            ),
            candidates_by_grid[key].point.y,
            candidates_by_grid[key].point.x,
            candidates_by_grid[key].candidate_id,
        ),
    )
    connected_keys = {root_key}
    pending_keys: deque[GridKey] = deque([root_key])

    while pending_keys:
        current_key = pending_keys.popleft()
        current_candidate = candidates_by_grid[current_key]
        for neighbor_key in _neighbor_keys(
            current_key,
            settings.neighbor_radius,
        ):
            if neighbor_key in connected_keys or neighbor_key not in eligible_keys:
                continue
            neighbor_candidate = candidates_by_grid.get(neighbor_key)
            if neighbor_candidate is None:
                continue
            neighbor_depth_difference = relative_depth_difference(
                current_candidate.depth,
                neighbor_candidate.depth,
                epsilon=epsilon,
            )
            if (
                neighbor_depth_difference is not None
                and neighbor_depth_difference <= settings.local_neighbor_depth_tolerance
            ):
                connected_keys.add(neighbor_key)
                pending_keys.append(neighbor_key)

    return tuple(
        candidate
        for candidate in candidates
        if (candidate.grid_y, candidate.grid_x) in connected_keys
    )


def _index_candidates(
    candidates: Sequence[LoadAnchorCandidate],
) -> dict[GridKey, LoadAnchorCandidate]:
    candidates_by_grid: dict[GridKey, LoadAnchorCandidate] = {}
    for candidate in candidates:
        key = (candidate.grid_y, candidate.grid_x)
        if key in candidates_by_grid:
            raise ValueError(
                "candidates must have unique grid coordinates; "
                f"duplicate coordinate (grid_y={key[0]}, grid_x={key[1]})"
            )
        candidates_by_grid[key] = candidate
    return candidates_by_grid


def _neighbor_keys(key: GridKey, neighbor_radius: int) -> tuple[GridKey, ...]:
    """Return surrounding grid coordinates, excluding the current cell."""

    grid_y, grid_x = key
    neighbors: list[GridKey] = []

    for row_offset in range(-neighbor_radius, neighbor_radius + 1):
        for column_offset in range(-neighbor_radius, neighbor_radius + 1):
            is_current_cell = row_offset == 0 and column_offset == 0
            if is_current_cell:
                continue

            neighbor_y = grid_y + row_offset
            neighbor_x = grid_x + column_offset
            neighbors.append((neighbor_y, neighbor_x))

    return tuple(neighbors)


def _validate_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite value greater than zero")


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("root_bbox must contain four finite values")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("root_bbox must have positive width and height")


def _point_is_inside_bbox(
    candidate: LoadAnchorCandidate,
    bbox: tuple[float, float, float, float],
) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= candidate.point.x < x2 and y1 <= candidate.point.y < y2


def _finite_difference_or_infinity(difference: float | None) -> float:
    if difference is None or not math.isfinite(difference):
        return math.inf
    return difference


__all__ = ["find_connected_candidate_region"]
