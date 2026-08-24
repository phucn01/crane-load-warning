import numpy as np
import pytest
from geometry_engine import (
    ConnectedRegionConfig,
    FarthestPointSamplingConfig,
    GeometryConfig,
    GeometryFramePipeline,
    LoadAnchorsConfig,
    RepresentativeDepthConfig,
    ZoneBufferConfig,
    ZonesConfig,
)
from vision_engine.contracts import Detection


def detection(
    class_name: str,
    bbox: tuple[float, float, float, float],
    *,
    mask: np.ndarray | None = None,
    confidence: float = 0.9,
    track_id: str | None = None,
) -> Detection:
    x1, y1, x2, y2 = bbox
    result: Detection = {
        "source_model": "fixture",
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "mask": mask,
    }
    if track_id is not None:
        result["track_id"] = track_id  # type: ignore[typeddict-unknown-key]
    return result


def pipeline() -> GeometryFramePipeline:
    return GeometryFramePipeline(
        GeometryConfig(
            representative_depth=RepresentativeDepthConfig(
                person_bottom_fraction=0.1,
                minimum_valid_pixels=1,
                load_inner_size_fraction=0.5,
            ),
            load_anchors=LoadAnchorsConfig(
                patch_size=3,
                patch_stride=4,
                seed_depth_tolerance=1.0,
            ),
            connected_region=ConnectedRegionConfig(
                neighbor_radius=1,
                local_neighbor_depth_tolerance=1.0,
            ),
            farthest_point_sampling=FarthestPointSamplingConfig(
                maximum_anchors=4,
                minimum_distance=0.0,
            ),
            zones=ZonesConfig(
                danger=ZoneBufferConfig(0.2, 0.2),
                warning=ZoneBufferConfig(0.4, 0.4),
            ),
        )
    )


def test_processes_people_loads_and_ignores_rope_detections():
    depth_map = np.linspace(0.1, 1.0, 48 * 48, dtype=np.float32).reshape(48, 48)
    person_mask = np.zeros((48, 48), dtype=bool)
    person_mask[20:44, 2:12] = True
    detections = (
        detection("hanging_rope", (20.0, 0.0, 24.0, 20.0)),
        detection(
            "person",
            (2.0, 20.0, 12.0, 44.0),
            mask=person_mask,
            track_id="person-track-7",
        ),
        detection(
            "hanging_object",
            (8.0, 8.0, 40.0, 40.0),
            track_id="load-track-2",
        ),
    )

    result = pipeline().process(detections, depth_map, frame_id="frame-001")

    assert result.frame_id == "frame-001"
    assert result.quality_reasons == ()
    assert len(result.persons) == 1
    assert len(result.loads) == 1
    assert result.persons[0].person_id == "person_01"
    assert result.persons[0].track_id == "person-track-7"
    assert result.persons[0].pseudo_bev_point is not None
    assert result.persons[0].mask_reliable is True
    assert result.loads[0].load_id == "hanging_object_01"
    assert result.loads[0].track_id == "load-track-2"
    assert result.loads[0].connected_candidates
    assert 2 <= len(result.loads[0].final_anchors) <= 4
    assert result.loads[0].safety_zones is not None
    assert result.loads[0].quality_reasons == ()


def test_same_inputs_return_the_same_geometry_result():
    depth_map = np.linspace(0.1, 1.0, 48 * 48, dtype=np.float32).reshape(48, 48)
    detections = (detection("hanging_object", (8.0, 8.0, 40.0, 40.0)),)
    geometry_pipeline = pipeline()

    first = geometry_pipeline.process(detections, depth_map, frame_id="frame-001")
    second = geometry_pipeline.process(detections, depth_map, frame_id="frame-001")

    assert first == second


def test_empty_detections_return_explicit_frame_quality_reasons():
    result = pipeline().process(
        (),
        np.ones((8, 8), dtype=np.float32),
        frame_id="frame-empty",
    )

    assert result.persons == ()
    assert result.loads == ()
    assert result.quality_reasons == (
        "no_person_detections",
        "no_load_detections",
    )


def test_load_without_valid_depth_returns_diagnostics_instead_of_a_zone():
    depth_map = np.ones((48, 48), dtype=np.float32)
    depth_map[8:40, 8:40] = np.nan

    result = pipeline().process(
        (detection("hanging_object", (8.0, 8.0, 40.0, 40.0)),),
        depth_map,
        frame_id="frame-001",
    )

    load = result.loads[0]
    assert load.safety_zones is None
    assert load.quality_reasons == (
        "load_depth_unavailable",
        "no_seed_consistent_candidates",
        "no_connected_candidates",
        "insufficient_final_anchors",
        "zone_geometry_unavailable",
    )


def test_rejects_empty_frame_id_and_non_finite_depth_map():
    with pytest.raises(ValueError, match="frame_id must not be empty"):
        pipeline().process((), np.ones((8, 8)), frame_id="")

    with pytest.raises(ValueError, match="at least one finite value"):
        pipeline().process(
            (),
            np.full((8, 8), np.nan),
            frame_id="frame-001",
        )
