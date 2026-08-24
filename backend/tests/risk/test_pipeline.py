import pytest
from pipeline_timeline import PipelineTimeline, TimelineStatus
from risk_engine import (
    EventPolicy,
    EventStateMachine,
    MediaFrameContext,
    MediaType,
    PersonObservation,
    Point2D,
    Rectangle2D,
    RiskEvaluator,
    RiskFramePipeline,
    RiskLevel,
    RiskPairInput,
    ZoneGeometry,
    aggregate_frame_risk,
)

ZONES = ZoneGeometry(
    danger=Rectangle2D(25.0, 75.0, 25.0, 75.0),
    warning=Rectangle2D(0.0, 100.0, 0.0, 100.0),
)


def image_context() -> MediaFrameContext:
    return MediaFrameContext(
        upload_id="image-001",
        frame_id="frame-000000",
        frame_index=0,
        timestamp=0.0,
        media_type=MediaType.IMAGE,
    )


def video_context(frame_index: int) -> MediaFrameContext:
    return MediaFrameContext(
        upload_id="video-001",
        frame_id=f"frame-{frame_index:06d}",
        frame_index=frame_index,
        timestamp=frame_index / 10.0,
        media_type=MediaType.VIDEO,
    )


def pair(
    person_id: str,
    anchor: Point2D,
    *,
    load_id: str = "load-1",
    person_track_id: str | None = None,
    load_track_id: str | None = None,
) -> RiskPairInput:
    return RiskPairInput(
        person=PersonObservation(
            person_id=person_id,
            anchor=anchor,
            confidence=0.9,
            track_id=person_track_id,
        ),
        load_id=load_id,
        zones=ZONES,
        load_track_id=load_track_id,
    )


def test_image_mode_returns_immediate_assessment_without_temporal_event():
    timeline = PipelineTimeline()
    result = RiskFramePipeline(timeline=timeline).process(
        (pair("person-1", Point2D(50.0, 50.0)),),
        context=image_context(),
    )

    assert result.assessment.level is RiskLevel.DANGER
    assert result.assessment.contributing_person_ids == ("person-1",)
    assert result.event is None
    timing = timeline.snapshot()[0]
    assert (timing.component, timing.operation) == ("risk", "process")
    assert timing.status is TimelineStatus.COMPLETED


def test_video_mode_confirms_one_scene_event_across_frames():
    event_policy = EventPolicy(danger_enter_frames=2)
    pipeline = RiskFramePipeline(
        evaluator=RiskEvaluator(),
        state_machine=EventStateMachine(event_policy),
    )
    pairs = (pair("frame-person", Point2D(50.0, 50.0)),)

    first = pipeline.process(
        pairs,
        context=video_context(0),
    )
    second = pipeline.process(
        pairs,
        context=video_context(1),
    )

    assert first.assessment.level is RiskLevel.DANGER
    assert first.event is not None and first.event.level is RiskLevel.SAFE
    assert second.event is not None and second.event.level is RiskLevel.DANGER
    assert second.event.alert_triggered is True


def test_empty_frame_is_unreliable_warning_instead_of_safe():
    result = RiskFramePipeline().process((), context=image_context())

    assert result.assessment.level is RiskLevel.WARNING
    assert result.assessment.assessment_reliable is False
    assert result.assessment.quality_reasons == ("no_risk_assessments",)


def test_aggregate_keeps_all_highest_level_contributors():
    evaluator = RiskEvaluator()
    assessments = tuple(
        evaluator.evaluate(item.person, load_id=item.load_id, zones=item.zones)
        for item in (
            pair("person-safe", Point2D(150.0, 50.0)),
            pair("person-danger-1", Point2D(50.0, 50.0)),
            pair("person-danger-2", Point2D(60.0, 60.0)),
        )
    )

    frame = aggregate_frame_risk(assessments, frame_id="frame-001")

    assert frame.level is RiskLevel.DANGER
    assert frame.contributing_person_ids == (
        "person-danger-1",
        "person-danger-2",
    )


def test_optional_track_ids_are_propagated_without_being_required():
    result = RiskFramePipeline().process(
        (
            pair(
                "person-1",
                Point2D(50.0, 50.0),
                person_track_id="person-track-7",
                load_track_id="load-track-2",
            ),
        ),
        context=image_context(),
    )
    assessment = result.assessment.pair_assessments[0]

    assert assessment.person_track_id == "person-track-7"
    assert assessment.load_track_id == "load-track-2"


def test_media_context_rejects_invalid_frame_metadata():
    with pytest.raises(ValueError, match="frame_index must be non-negative"):
        MediaFrameContext(
            upload_id="video-001",
            frame_id="frame-invalid",
            frame_index=-1,
            timestamp=0.0,
            media_type=MediaType.VIDEO,
        )

    with pytest.raises(TypeError, match="media_type must be a MediaType"):
        MediaFrameContext(
            upload_id="video-001",
            frame_id="frame-000000",
            frame_index=0,
            timestamp=0.0,
            media_type="video",  # type: ignore[arg-type]
        )
