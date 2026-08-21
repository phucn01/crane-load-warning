import numpy as np
from geometry_engine.config import RepresentativeDepthConfig
from geometry_engine.representative_depth import (
    load_representative_depth,
    person_representative_depth,
    person_representative_point,
)


def make_detection(class_name: str, mask=None, bbox=(0.0, 0.0, 10.0, 10.0)):
    x1, y1, x2, y2 = bbox
    return {
        "source_model": "test",
        "class_id": 0,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox": bbox,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "mask": mask,
    }


def test_person_point_uses_bottom_mask_band_then_bbox_fallback():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 2:8] = True
    detection = make_detection("person", mask)

    point, source = person_representative_point(
        detection,
        image_shape=mask.shape,
    )

    assert (point.x, point.y) == (4.5, 9.0)
    assert source == "segmentation_bottom_band"

    detection["mask"] = None
    point, source = person_representative_point(
        detection,
        image_shape=mask.shape,
    )
    assert (point.x, point.y) == (5.0, 10.0)
    assert source == "bbox_bottom_center"


def test_person_depth_uses_mask_and_lower_roi_when_consistent():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, 2:8] = True
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[mask] = 2.0

    result = person_representative_depth(
        make_detection("person", mask),
        depth,
        minimum_valid_pixels=2,
    )

    assert result.value == 2.0
    assert result.quality == "high"
    assert result.source == "segmentation_mask:lower"
    assert result.top_lower_relative_difference == 0.0


def test_person_depth_falls_back_to_full_bbox_when_mask_is_invalid():
    depth = np.full((10, 10), 3.0, dtype=np.float32)
    wrong_shape_mask = np.ones((2, 2), dtype=bool)

    result = person_representative_depth(
        make_detection("person", wrong_shape_mask),
        depth,
    )

    assert result.value == 3.0
    assert result.source == "bbox_fallback:lower"


def test_load_depth_prefers_inner_roi_then_falls_back_to_full_bbox():
    detection = make_detection(
        "hanging_object",
        bbox=(0.0, 0.0, 20.0, 20.0),
    )
    config = RepresentativeDepthConfig(
        minimum_valid_pixels=4,
        load_inner_size_fraction=0.10,
    )
    depth = np.ones((20, 20), dtype=np.float32)
    depth[9:11, 9:11] = 4.0

    inner = load_representative_depth(detection, depth, config=config)

    assert inner.value == 4.0
    assert inner.source == "inner"
    assert inner.quality == "high"

    depth[9:11, 9:11] = np.nan
    fallback = load_representative_depth(detection, depth, config=config)

    assert fallback.value == 1.0
    assert fallback.source == "full"
    assert fallback.quality == "low"
