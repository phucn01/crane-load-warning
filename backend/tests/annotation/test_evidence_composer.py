import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from annotation_engine import EvidenceTraceability, OfflineEvidenceComposer
from pipeline_timeline import PipelineTimeline, TimelineStatus
from risk_engine import RiskFrameResult, RiskLevel

from .conftest import AnnotationBundle

TRACEABILITY = EvidenceTraceability(
    pipeline_version="phase-4-test",
    model_versions={"person": "yolo-test", "load": "rfdetr-test"},
    config_versions={"geometry": "geo-v1", "risk": "risk-v2"},
)


@pytest.mark.parametrize(
    "level",
    [RiskLevel.SAFE, RiskLevel.WARNING, RiskLevel.DANGER],
)
def test_writes_image_and_traceable_json_for_every_assessment(
    tmp_path: Path,
    annotation_bundle: AnnotationBundle,
    level: RiskLevel,
):
    risk_result = _with_level(annotation_bundle.risk_result, level)
    artifacts = OfflineEvidenceComposer(pseudo_bev_size=(320, 320)).write(
        image_bgr=annotation_bundle.image_bgr,
        detections=annotation_bundle.detections,
        geometry=annotation_bundle.geometry,
        risk_result=risk_result,
        context=annotation_bundle.context,
        traceability=TRACEABILITY,
        output_dir=tmp_path,
    )

    assert artifacts.evidence_image_path.is_file()
    assert artifacts.assessment_json_path.is_file()
    evidence = cv2.imread(str(artifacts.evidence_image_path), cv2.IMREAD_COLOR)
    assert evidence is not None and evidence.shape == (416, 640, 3)
    payload = json.loads(artifacts.assessment_json_path.read_text(encoding="utf-8"))
    assert payload["assessment"]["level"] == level.value
    assert payload["assessment"]["pairs"][0]["level"] == level.value
    assert payload["geometry"]["coordinate_system"] == (
        "relative_pseudo_bev_not_metric"
    )
    assert payload["traceability"] == {
        "pipeline_version": "phase-4-test",
        "model_versions": {"load": "rfdetr-test", "person": "yolo-test"},
        "config_versions": {"geometry": "geo-v1", "risk": "risk-v2"},
    }
    assert payload["artifacts"]["evidence_image"] == (
        artifacts.evidence_image_path.name
    )

def test_default_pseudo_bev_uses_wide_chart_size(
    annotation_bundle: AnnotationBundle,
):
    composer = OfflineEvidenceComposer()
    output = composer.render_pseudo_bev(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
    )
    combined = composer.compose(
        image_bgr=annotation_bundle.image_bgr,
        detections=annotation_bundle.detections,
        geometry=annotation_bundle.geometry,
        risk_result=annotation_bundle.risk_result,
        context=annotation_bundle.context,
        traceability=TRACEABILITY,
    )

    assert output.shape == (720, 960, 3)
    assert combined.shape == (816, 1920, 3)


def test_composed_evidence_is_deterministic(annotation_bundle: AnnotationBundle):
    timeline = PipelineTimeline()
    composer = OfflineEvidenceComposer(
        pseudo_bev_size=(320, 320),
        timeline=timeline,
    )
    arguments = {
        "image_bgr": annotation_bundle.image_bgr,
        "detections": annotation_bundle.detections,
        "geometry": annotation_bundle.geometry,
        "risk_result": annotation_bundle.risk_result,
        "context": annotation_bundle.context,
        "traceability": TRACEABILITY,
    }

    first = composer.compose(**arguments)
    second = composer.compose(**arguments)

    assert np.array_equal(first, second)
    assert [record.component for record in timeline.snapshot()] == [
        "annotation",
        "annotation",
    ]
    assert all(
        record.status is TimelineStatus.COMPLETED
        for record in timeline.snapshot()
    )


def test_standalone_and_combined_evidence_share_one_bev_renderer(
    annotation_bundle: AnnotationBundle,
    monkeypatch: pytest.MonkeyPatch,
):
    composer = OfflineEvidenceComposer(pseudo_bev_size=(320, 240))
    expected = np.full((240, 320, 3), 123, dtype=np.uint8)
    calls = 0

    def fake_renderer(geometry, assessment, *, width, height):
        nonlocal calls
        calls += 1
        assert geometry is annotation_bundle.geometry
        assert assessment is annotation_bundle.risk_result.assessment
        assert (width, height) == (320, 240)
        return expected.copy()

    monkeypatch.setattr(
        "annotation_engine.evidence_composer.render_pseudo_bev_chart",
        fake_renderer,
    )

    standalone = composer.render_pseudo_bev(
        annotation_bundle.geometry,
        annotation_bundle.risk_result.assessment,
    )
    composer.compose(
        image_bgr=annotation_bundle.image_bgr,
        detections=annotation_bundle.detections,
        geometry=annotation_bundle.geometry,
        risk_result=annotation_bundle.risk_result,
        context=annotation_bundle.context,
        traceability=TRACEABILITY,
    )

    assert np.array_equal(standalone, expected)
    assert calls == 2


def test_rejects_mismatched_frame_identity(annotation_bundle: AnnotationBundle):
    wrong_context = replace(annotation_bundle.context, frame_id="other-frame")

    with pytest.raises(ValueError, match="frame_id must match"):
        OfflineEvidenceComposer().compose(
            image_bgr=annotation_bundle.image_bgr,
            detections=annotation_bundle.detections,
            geometry=annotation_bundle.geometry,
            risk_result=annotation_bundle.risk_result,
            context=wrong_context,
            traceability=TRACEABILITY,
        )


def _with_level(result: RiskFrameResult, level: RiskLevel) -> RiskFrameResult:
    pair = replace(
        result.assessment.pair_assessments[0],
        level=level,
        matched_zone=None if level is RiskLevel.SAFE else level,
    )
    assessment = replace(
        result.assessment,
        level=level,
        pair_assessments=(pair,),
    )
    return replace(result, assessment=assessment)
