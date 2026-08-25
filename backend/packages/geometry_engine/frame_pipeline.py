"""Frame-level orchestration for the relative geometry engine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pipeline_timeline import PipelineTimeline, log_pipeline_operation
from vision_engine.contracts import Detection, clip_bbox

from .config import GeometryConfig
from .connected_region import find_connected_candidate_region
from .contracts import (
    GeometryFrameResult,
    LoadGeometryResult,
    PersonGeometryResult,
)
from .depth_utils import calculate_depth_normalization_range
from .farthest_point_sampling import select_farthest_load_anchors
from .load_anchors import build_load_anchor_candidates
from .pseudo_bev import (
    project_load_anchors_to_pseudo_bev,
    project_person_to_pseudo_bev,
)
from .representative_depth import (
    estimate_person_representative,
    load_representative_depth,
)
from .zones import build_load_zones


class GeometryFramePipeline:
    """Run all geometry stages for one frame without duplicating algorithms."""

    def __init__(
        self,
        config: GeometryConfig | None = None,
        *,
        timeline: PipelineTimeline | None = None,
    ) -> None:
        self.config = config or GeometryConfig()
        self.timeline = timeline

    def process(
        self,
        detections: Iterable[Detection],
        depth_map: NDArray[np.generic],
        *,
        frame_id: str,
    ) -> GeometryFrameResult:
        if self.timeline is not None:
            with self.timeline.track("geometry", "process", frame_id=frame_id):
                return self._process(detections, depth_map, frame_id=frame_id)
        return self._process(detections, depth_map, frame_id=frame_id)

    def _process(
        self,
        detections: Iterable[Detection],
        depth_map: NDArray[np.generic],
        *,
        frame_id: str,
    ) -> GeometryFrameResult:
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        _validate_depth_map(depth_map)

        with log_pipeline_operation(
            "geometry", "depth_normalization", frame_id=frame_id
        ):
            depth_low, depth_high = calculate_depth_normalization_range(
                depth_map, upper_percentile=100
            )
        image_width = int(depth_map.shape[1])
        person_detections: list[Detection] = []
        load_detections: list[Detection] = []
        for detection in detections:
            if detection["class_name"] == "person":
                person_detections.append(detection)
            elif detection["class_name"] == "hanging_object":
                load_detections.append(detection)

        persons = tuple(
            self._process_person(
                detection,
                person_id=f"person_{index:02d}",
                frame_id=frame_id,
                depth_map=depth_map,
                image_width=image_width,
                depth_low=depth_low,
                depth_high=depth_high,
            )
            for index, detection in enumerate(person_detections, start=1)
        )
        loads = tuple(
            self._process_load(
                detection,
                load_id=f"hanging_object_{index:02d}",
                frame_id=frame_id,
                depth_map=depth_map,
                image_width=image_width,
                depth_low=depth_low,
                depth_high=depth_high,
            )
            for index, detection in enumerate(load_detections, start=1)
        )
        frame_reasons: list[str] = []
        if not persons:
            frame_reasons.append("no_person_detections")
        if not loads:
            frame_reasons.append("no_load_detections")
        return GeometryFrameResult(
            frame_id=frame_id,
            depth_low=depth_low,
            depth_high=depth_high,
            persons=persons,
            loads=loads,
            quality_reasons=tuple(frame_reasons),
        )

    def _process_person(
        self,
        detection: Detection,
        *,
        person_id: str,
        frame_id: str,
        depth_map: NDArray[np.generic],
        image_width: int,
        depth_low: float,
        depth_high: float,
    ) -> PersonGeometryResult:
        with log_pipeline_operation(
            "geometry",
            "person_representative_depth",
            frame_id=frame_id,
            entity_id=person_id,
        ):
            representative = estimate_person_representative(
                detection,
                depth_map,
                config=self.config.representative_depth,
            )
        with log_pipeline_operation(
            "geometry",
            "person_pseudo_bev_projection",
            frame_id=frame_id,
            entity_id=person_id,
        ):
            pseudo_bev_point = project_person_to_pseudo_bev(
                representative,
                image_width=image_width,
                depth_low=depth_low,
                depth_high=depth_high,
                config=self.config.pseudo_bev,
            )
        mask_reliable = _has_valid_mask(detection, depth_map.shape)
        reasons: list[str] = []
        if not mask_reliable:
            reasons.append("person_mask_unavailable")
        if representative.depth.quality == "unavailable":
            reasons.append("person_depth_unavailable")
        elif representative.depth.quality == "low":
            reasons.append("person_depth_low_quality")
        if pseudo_bev_point is None:
            reasons.append("person_projection_unavailable")
        return PersonGeometryResult(
            person_id=person_id,
            confidence=float(detection["confidence"]),
            bbox=_required_bbox(detection, depth_map.shape),
            representative=representative,
            pseudo_bev_point=pseudo_bev_point,
            mask_reliable=mask_reliable,
            quality_reasons=tuple(reasons),
            track_id=_optional_track_id(detection),
        )

    def _process_load(
        self,
        detection: Detection,
        *,
        load_id: str,
        frame_id: str,
        depth_map: NDArray[np.generic],
        image_width: int,
        depth_low: float,
        depth_high: float,
    ) -> LoadGeometryResult:
        bbox = _required_bbox(detection, depth_map.shape)
        with log_pipeline_operation(
            "geometry",
            "load_representative_depth",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            representative_depth = load_representative_depth(
                detection,
                depth_map,
                config=self.config.representative_depth,
            )
        with log_pipeline_operation(
            "geometry",
            "load_anchor_candidates",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            candidate_selection = build_load_anchor_candidates(
                detection,
                depth_map,
                representative_config=self.config.representative_depth,
                anchors_config=self.config.load_anchors,
            )
        inner_bbox = _centered_bbox(
            bbox,
            self.config.representative_depth.load_inner_size_fraction,
        )
        with log_pipeline_operation(
            "geometry",
            "connected_candidate_region",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            connected_candidates = find_connected_candidate_region(
                candidate_selection.candidates,
                root_bbox=inner_bbox,
                config=self.config.connected_region,
            )
        with log_pipeline_operation(
            "geometry",
            "farthest_anchor_selection",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            final_anchors = select_farthest_load_anchors(
                connected_candidates,
                config=self.config.farthest_point_sampling,
            )
        with log_pipeline_operation(
            "geometry",
            "load_pseudo_bev_projection",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            pseudo_bev_points = project_load_anchors_to_pseudo_bev(
                final_anchors,
                image_width=image_width,
                depth_low=depth_low,
                depth_high=depth_high,
                config=self.config.pseudo_bev,
            )
        with log_pipeline_operation(
            "geometry",
            "safety_zone_construction",
            frame_id=frame_id,
            entity_id=load_id,
        ):
            safety_zones = build_load_zones(
                pseudo_bev_points,
                load_bbox=bbox,
                image_width=image_width,
                pseudo_bev_config=self.config.pseudo_bev,
                config=self.config.zones,
            )
        reasons: list[str] = []
        if representative_depth.quality == "unavailable":
            reasons.append("load_depth_unavailable")
        elif representative_depth.quality == "low":
            reasons.append("load_depth_low_quality")
        if not candidate_selection.consistent_candidates:
            reasons.append("no_seed_consistent_candidates")
        if not connected_candidates:
            reasons.append("no_connected_candidates")
        if len(final_anchors) < 2:
            reasons.append("insufficient_final_anchors")
        if safety_zones is None:
            reasons.append("zone_geometry_unavailable")
        return LoadGeometryResult(
            load_id=load_id,
            confidence=float(detection["confidence"]),
            bbox=bbox,
            representative_depth=representative_depth,
            candidate_selection=candidate_selection,
            inner_bbox=inner_bbox,
            connected_candidates=connected_candidates,
            final_anchors=final_anchors,
            pseudo_bev_points=pseudo_bev_points,
            safety_zones=safety_zones,
            quality_reasons=tuple(reasons),
            track_id=_optional_track_id(detection),
        )


def _validate_depth_map(depth_map: Any) -> None:
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a numpy.ndarray")
    if depth_map.ndim != 2:
        raise ValueError("depth_map must have shape (height, width)")
    if depth_map.shape[0] <= 0 or depth_map.shape[1] <= 0:
        raise ValueError("depth_map height and width must be greater than zero")


def _required_bbox(
    detection: Detection,
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    bbox = clip_bbox(detection["bbox"], image_shape)
    if bbox is None:
        raise ValueError(f"{detection['class_name']} detection has an invalid bbox")
    return bbox


def _centered_bbox(
    bbox: tuple[float, float, float, float],
    size_fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * size_fraction / 2.0
    half_height = (y2 - y1) * size_fraction / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _has_valid_mask(detection: Detection, image_shape: tuple[int, ...]) -> bool:
    mask = detection["mask"]
    return bool(
        isinstance(mask, np.ndarray)
        and mask.shape == tuple(image_shape[:2])
        and np.any(mask)
    )


def _optional_track_id(detection: Detection) -> str | None:
    track_id = detection.get("track_id")  # type: ignore[typeddict-item]
    return str(track_id) if track_id is not None else None


__all__ = ["GeometryFramePipeline"]
