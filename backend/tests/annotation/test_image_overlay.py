from dataclasses import replace

import numpy as np
import pytest
from annotation_engine import render_image_overlay
from annotation_engine.image_overlay import CLASS_COLORS_BGR, RISK_COLORS_BGR
from risk_engine import RiskLevel

from .conftest import AnnotationBundle


def test_prefers_person_segment_and_uses_bboxes_as_fallback(
    annotation_bundle: AnnotationBundle,
):
    output = render_image_overlay(
        annotation_bundle.image_bgr,
        annotation_bundle.detections,
        annotation_bundle.risk_result.assessment,
    )

    assert output.shape == annotation_bundle.image_bgr.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, annotation_bundle.image_bgr)
    assert not np.array_equal(output[100, 40], annotation_bundle.image_bgr[100, 40])
    assert np.array_equal(output[100, 20], annotation_bundle.image_bgr[100, 20])
    assert _near(output[100, 30], RISK_COLORS_BGR[RiskLevel.DANGER])
    assert not np.array_equal(output[100, 90], annotation_bundle.image_bgr[100, 90])
    assert not np.array_equal(output[45, 120], annotation_bundle.image_bgr[45, 120])
    assert _near(output[10, 230], RISK_COLORS_BGR[RiskLevel.DANGER])


def test_image_renderer_is_deterministic(annotation_bundle: AnnotationBundle):
    first = render_image_overlay(
        annotation_bundle.image_bgr,
        annotation_bundle.detections,
        annotation_bundle.risk_result.assessment,
    )
    second = render_image_overlay(
        annotation_bundle.image_bgr,
        annotation_bundle.detections,
        annotation_bundle.risk_result.assessment,
    )

    assert np.array_equal(first, second)


def test_risk_color_is_applied_to_person_but_not_load(
    annotation_bundle: AnnotationBundle,
):
    original = annotation_bundle.risk_result.assessment
    warning_pair = replace(
        original.pair_assessments[0],
        level=RiskLevel.WARNING,
        matched_zone=RiskLevel.WARNING,
    )
    warning_assessment = replace(
        original,
        level=RiskLevel.WARNING,
        pair_assessments=(warning_pair,),
    )

    output = render_image_overlay(
        annotation_bundle.image_bgr,
        annotation_bundle.detections,
        warning_assessment,
    )

    assert _near(output[100, 30], RISK_COLORS_BGR[RiskLevel.WARNING])
    assert _near(output[100, 90], CLASS_COLORS_BGR["hanging_object"])


def test_rejects_invalid_camera_image(annotation_bundle: AnnotationBundle):
    with pytest.raises(ValueError, match="shape"):
        render_image_overlay(
            np.zeros((10, 10), dtype=np.uint8),
            annotation_bundle.detections,
            annotation_bundle.risk_result.assessment,
        )


def _near(pixel: np.ndarray, color: tuple[int, int, int], tolerance: int = 8) -> bool:
    return bool(np.all(np.abs(pixel.astype(int) - np.asarray(color)) <= tolerance))
