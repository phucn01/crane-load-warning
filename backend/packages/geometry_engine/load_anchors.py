"""Generate load patches and filter them against representative seed depth."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vision_engine.contracts import Detection, clip_bbox

from .config import LoadAnchorsConfig, RepresentativeDepthConfig
from .contracts import (
    ImagePoint,
    LoadAnchorCandidate,
    LoadAnchorCandidates,
)
from .representative_depth import (
    load_representative_depth,
    relative_depth_difference,
)


def build_load_anchor_candidates(
    detection: Detection,
    depth_map: NDArray[np.generic],
    *,
    representative_config: RepresentativeDepthConfig | None = None,
    anchors_config: LoadAnchorsConfig | None = None,
) -> LoadAnchorCandidates:
    """Generate and seed-filter candidate patches for one hanging object."""

    representative_depth_settings = representative_config or RepresentativeDepthConfig()
    load_anchor_settings = anchors_config or LoadAnchorsConfig()
    _validate_inputs(detection, depth_map)

    seed_depth_result = load_representative_depth(
        detection,
        depth_map,
        config=representative_depth_settings,
    )
    candidates, generated_patch_count = generate_candidate_patches(
        detection,
        depth_map,
        config=load_anchor_settings,
    )
    annotated_candidates = filter_candidates_by_seed_depth(
        candidates,
        seed_depth=seed_depth_result.value,
        seed_depth_tolerance=load_anchor_settings.seed_depth_tolerance,
    )
    return LoadAnchorCandidates(
        seed_depth=seed_depth_result.value,
        seed_source=seed_depth_result.source,
        generated_patch_count=generated_patch_count,
        rejected_invalid_depth_count=generated_patch_count - len(candidates),
        candidates=tuple(annotated_candidates),
    )


def generate_candidate_patches(
    detection: Detection,
    depth_map: NDArray[np.generic],
    *,
    config: LoadAnchorsConfig | None = None,
) -> tuple[list[LoadAnchorCandidate], int]:
    """Sample robust local depths on a regular grid inside a load bbox.

    The returned integer counts every spatial patch position before invalid-depth
    patches are rejected. The median of all finite values is used as the robust
    candidate depth.
    """

    settings = config or LoadAnchorsConfig()
    _validate_inputs(detection, depth_map)
    bbox = clip_bbox(detection["bbox"], depth_map.shape)
    if bbox is None:
        raise ValueError("hanging_object detection has an invalid bbox")

    x1, y1, x2, y2 = bbox
    patch_radius = settings.patch_radius
    x_centers = _grid_centers(x1, x2, settings.patch_stride)
    y_centers = _grid_centers(y1, y2, settings.patch_stride)

    candidates: list[LoadAnchorCandidate] = []
    generated_patch_count = 0
    for grid_y, point_y in enumerate(y_centers):
        for grid_x, point_x in enumerate(x_centers):
            generated_patch_count += 1
            patch_center_x = round(float(point_x))
            patch_center_y = round(float(point_y))
            height, width = depth_map.shape
            patch_bbox = (
                max(0, patch_center_x - patch_radius),
                max(0, patch_center_y - patch_radius),
                min(width, patch_center_x + patch_radius + 1),
                min(height, patch_center_y + patch_radius + 1),
            )
            px1, py1, px2, py2 = patch_bbox
            depth_patch = depth_map[py1:py2, px1:px2]
            valid_depths = np.asarray(depth_patch, dtype=np.float32)
            valid_depths = valid_depths[np.isfinite(valid_depths)]
            valid_depth_count = int(valid_depths.size)
            if valid_depth_count == 0:
                continue

            candidates.append(
                LoadAnchorCandidate(
                    candidate_id=f"candidate_{len(candidates) + 1:04d}",
                    grid_x=grid_x,
                    grid_y=grid_y,
                    point=ImagePoint(x=float(point_x), y=float(point_y)),
                    patch_bbox=patch_bbox,
                    depth=float(np.median(valid_depths)),
                    valid_depth_count=valid_depth_count,
                    valid_depth_fraction=float(valid_depth_count / depth_patch.size),
                )
            )
    return candidates, generated_patch_count


def filter_candidates_by_seed_depth(
    candidates: list[LoadAnchorCandidate] | tuple[LoadAnchorCandidate, ...],
    *,
    seed_depth: float | None,
    seed_depth_tolerance: float,
    epsilon: float = 1e-8,
) -> list[LoadAnchorCandidate]:
    """Annotate candidates with seed error and consistency decisions."""

    if not math.isfinite(seed_depth_tolerance) or seed_depth_tolerance < 0:
        raise ValueError("seed_depth_tolerance must be a finite non-negative value")

    annotated_candidates: list[LoadAnchorCandidate] = []
    for candidate in candidates:
        seed_depth_difference = relative_depth_difference(
            candidate.depth,
            seed_depth,
            epsilon=epsilon,
        )
        annotated_candidates.append(
            replace(
                candidate,
                seed_depth_difference=seed_depth_difference,
                is_seed_consistent=(
                    seed_depth_difference is not None
                    and seed_depth_difference <= seed_depth_tolerance
                ),
            )
        )
    return annotated_candidates


def _grid_centers(start: float, end: float, stride: int) -> NDArray[np.float64]:
    """Match the research grid: half-stride offset, midpoint for small bboxes."""

    if end - start <= stride:
        return np.asarray([(start + end) / 2.0], dtype=np.float64)
    return np.arange(start + stride / 2.0, end, stride, dtype=np.float64)


def _validate_inputs(detection: Detection, depth_map: Any) -> None:
    if detection["class_name"] != "hanging_object":
        raise ValueError(
            f"expected a hanging_object detection, got {detection['class_name']}"
        )
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a numpy.ndarray")
    if depth_map.ndim != 2:
        raise ValueError("depth_map must have shape (height, width)")
    if depth_map.shape[0] <= 0 or depth_map.shape[1] <= 0:
        raise ValueError("depth_map height and width must be greater than zero")


__all__ = [
    "build_load_anchor_candidates",
    "filter_candidates_by_seed_depth",
    "generate_candidate_patches",
]
