"""Frame-level orchestration for image uploads and scene-level video events."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pipeline_timeline import PipelineTimeline

from .contracts import (
    EventScope,
    FrameRiskAssessment,
    MediaFrameContext,
    RiskFrameResult,
    RiskLevel,
    RiskPairInput,
    RiskSignal,
    SafetyAssessment,
)
from .evaluator import RiskEvaluator
from .event_state_machine import EventStateMachine


def aggregate_frame_risk(
    assessments: Sequence[SafetyAssessment],
    *,
    frame_id: str,
) -> FrameRiskAssessment:
    """Aggregate pair assessments without assuming stable cross-frame identities."""

    if not frame_id:
        raise ValueError("frame_id must not be empty")
    pair_assessments = tuple(assessments)
    if not pair_assessments:
        return FrameRiskAssessment(
            frame_id=frame_id,
            level=RiskLevel.SAFE,
            assessment_reliable=False,
            quality_reasons=("no_risk_assessments",),
            pair_assessments=(),
        )

    highest_level = max(
        (assessment.level for assessment in pair_assessments),
        key=lambda level: level.severity,
    )
    contributors = tuple(
        assessment
        for assessment in pair_assessments
        if assessment.level is highest_level
    )
    quality_reasons = tuple(
        dict.fromkeys(
            reason
            for assessment in pair_assessments
            for reason in assessment.quality_reasons
        )
    )
    return FrameRiskAssessment(
        frame_id=frame_id,
        level=highest_level,
        assessment_reliable=all(
            assessment.assessment_reliable for assessment in pair_assessments
        ),
        quality_reasons=quality_reasons,
        pair_assessments=pair_assessments,
        contributing_person_ids=tuple(
            dict.fromkeys(assessment.person_id for assessment in contributors)
        ),
        contributing_load_ids=tuple(
            dict.fromkeys(assessment.load_id for assessment in contributors)
        ),
    )


class RiskFramePipeline:
    """Evaluate one frame immediately and optionally update a temporal event."""

    def __init__(
        self,
        evaluator: RiskEvaluator | None = None,
        state_machine: EventStateMachine | None = None,
        *,
        timeline: PipelineTimeline | None = None,
    ) -> None:
        self.evaluator = evaluator or RiskEvaluator()
        self.state_machine = state_machine or EventStateMachine()
        self.timeline = timeline

    def process(
        self,
        pair_inputs: Iterable[RiskPairInput],
        *,
        context: MediaFrameContext,
        update_temporal_event: bool = True,
    ) -> RiskFrameResult:
        """Process an image or video frame through the same frame-level API."""

        if self.timeline is not None:
            with self.timeline.track(
                "risk",
                "process",
                frame_id=context.frame_id,
            ):
                return self._process(
                    pair_inputs,
                    context=context,
                    update_temporal_event=update_temporal_event,
                )
        return self._process(
            pair_inputs,
            context=context,
            update_temporal_event=update_temporal_event,
        )

    def _process(
        self,
        pair_inputs: Iterable[RiskPairInput],
        *,
        context: MediaFrameContext,
        update_temporal_event: bool,
    ) -> RiskFrameResult:
        assessments = tuple(
            self.evaluator.evaluate(
                item.person,
                load_id=item.load_id,
                zones=item.zones,
                load_track_id=item.load_track_id,
            )
            for item in pair_inputs
        )
        frame_assessment = aggregate_frame_risk(
            assessments,
            frame_id=context.frame_id,
        )

        event = None
        # Runtime event association is deferred until tracking is introduced.
        # Callers processing independent video frames explicitly disable it.
        if context.has_temporal_event and update_temporal_event:
            event = self.state_machine.update(
                RiskSignal(
                    level=frame_assessment.level,
                    assessment_reliable=frame_assessment.assessment_reliable,
                    quality_reasons=frame_assessment.quality_reasons,
                ),
                event_key=context.scene_event_key,
                event_scope=EventScope.SCENE,
                timestamp=context.timestamp,
            )
        return RiskFrameResult(assessment=frame_assessment, event=event)


__all__ = ["RiskFramePipeline", "aggregate_frame_risk"]
