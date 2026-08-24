from unittest.mock import Mock

import numpy as np
import pytest
from annotation_engine import (
    draw_pseudo_bev_chart,
    render_pseudo_bev_chart,
    render_pseudo_bev_overlay,
)
from annotation_engine.image_overlay import RISK_COLORS_BGR
from annotation_engine.pseudo_bev_overlay import FOOTPRINT_BGR
from risk_engine import RiskLevel

from .conftest import AnnotationBundle

OLD_LOAD_ANCHOR_BGR = (83, 200, 30)


def test_pseudo_bev_snapshot_colors_with_pixel_tolerance(
    annotation_bundle: AnnotationBundle,
):
    output = render_pseudo_bev_overlay(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
        width=480,
        height=400,
    )

    assert output.shape == (400, 480, 3)
    assert output.dtype == np.uint8
    assert not _contains_color(output, OLD_LOAD_ANCHOR_BGR, tolerance=4)
    assert _contains_color(output, FOOTPRINT_BGR, tolerance=8)
    assert _contains_color(
        output,
        RISK_COLORS_BGR[RiskLevel.DANGER],
        tolerance=8,
    )
    assert np.count_nonzero(np.any(output != output[0, 0], axis=2)) > 1_000


def test_pseudo_bev_renderer_is_deterministic(annotation_bundle: AnnotationBundle):
    first = render_pseudo_bev_overlay(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
        width=320,
        height=320,
    )
    second = render_pseudo_bev_overlay(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
        width=320,
        height=320,
    )

    assert np.array_equal(first, second)


def test_draws_research_chart_from_production_geometry(
    annotation_bundle: AnnotationBundle,
):
    axis = Mock()
    axis.get_legend_handles_labels.return_value = ([object()], ["item"])

    draw_pseudo_bev_chart(
        axis,
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
    )

    assert axis.fill.call_count == 3
    assert axis.plot.call_count == 3
    assert axis.scatter.call_count == 2
    axis.set_title.assert_called_once_with("Pseudo-BEV Safety View")
    axis.set_xlabel.assert_called_once_with("Relative lateral position")
    axis.set_ylabel.assert_called_once_with("Relative longitudinal position")
    axis.legend.assert_called_once_with(loc="best")


def test_renders_research_chart_as_api_image(
    annotation_bundle: AnnotationBundle,
):
    output = render_pseudo_bev_chart(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
        width=640,
        height=480,
        dpi=100,
    )

    assert output.shape == (480, 640, 3)
    assert output.dtype == np.uint8
    assert np.count_nonzero(np.any(output != output[0, 0], axis=2)) > 1_000


def test_renders_research_style_title_and_axis_labels(
    annotation_bundle: AnnotationBundle,
    monkeypatch: pytest.MonkeyPatch,
):
    import cv2

    rendered_text: list[str] = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    render_pseudo_bev_overlay(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
        width=480,
        height=400,
    )

    assert any(text.startswith("Pseudo-BEV") for text in rendered_text)
    assert "Relative lateral position" in rendered_text
    assert "Relative longitudinal position" in rendered_text
    assert all("not metric" not in text.lower() for text in rendered_text)


def test_rejects_too_small_pseudo_bev_panel(annotation_bundle: AnnotationBundle):
    with pytest.raises(ValueError, match="at least 160"):
        render_pseudo_bev_overlay(
            annotation_bundle.geometry,
            annotation_bundle.risk_result.assessment,
            width=100,
            height=100,
        )


def _contains_color(
    image: np.ndarray,
    color: tuple[int, int, int],
    *,
    tolerance: int,
) -> bool:
    difference = np.abs(image.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return bool(np.any(np.all(difference <= tolerance, axis=2)))
