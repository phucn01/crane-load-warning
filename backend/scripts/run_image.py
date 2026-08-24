"""Run the complete safety pipeline for one image."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from annotation_engine import (
    EvidenceTraceability,
    OfflineEvidenceComposer,
    draw_pseudo_bev_chart,
    render_image_overlay,
)
from geometry_engine import (
    GeometryFramePipeline,
    GeometryFrameResult,
    PseudoBEVRectangle,
    load_geometry_config,
)
from pipeline_timeline import PipelineTimeline
from risk_engine import (
    EventStateMachine,
    FrameRiskAssessment,
    MediaFrameContext,
    MediaType,
    PersonObservation,
    Point2D,
    Rectangle2D,
    RiskEvaluator,
    RiskFramePipeline,
    RiskPairInput,
    ZoneGeometry,
    load_risk_policy,
)
from vision_engine import (
    Detection,
    VisionFramePipeline,
    build_model_manager,
    write_vision_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vision, Geometry, Risk, and Annotation for one image."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--geometry-config", required=True, type=Path)
    parser.add_argument("--risk-config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "images",
        help="Parent directory for outputs/images/<run-id>",
    )
    parser.add_argument(
        "--run-id",
        help="Optional output directory and frame ID; defaults to a UTC timestamp",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all model weights before reading the input image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = _required_file(args.image, "input image")
    models_config_path = _required_file(args.models_config, "models config")
    geometry_config_path = _required_file(args.geometry_config, "geometry config")
    risk_config_path = _required_file(args.risk_config, "risk config")
    run_id = _resolve_run_id(args.run_id)
    run_dir = args.output_root.resolve() / run_id

    models_config = _load_yaml(models_config_path)
    geometry_config = load_geometry_config(geometry_config_path)
    risk_policy = load_risk_policy(risk_config_path)
    model_manager = build_model_manager(
        models_config,
        config_dir=models_config_path.parent,
    )
    if args.preload:
        model_manager.load_all()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode input image: {image_path}")

    timeline = PipelineTimeline()
    vision_pipeline = VisionFramePipeline(model_manager, timeline=timeline)
    geometry_pipeline = GeometryFramePipeline(geometry_config, timeline=timeline)
    risk_pipeline = RiskFramePipeline(
        evaluator=RiskEvaluator(risk_policy.evaluation),
        state_machine=EventStateMachine(risk_policy.events),
        timeline=timeline,
    )
    annotation_composer = OfflineEvidenceComposer(timeline=timeline)

    vision_result = vision_pipeline.process(
        image_bgr,
        frame_id=run_id,
        image_path=image_path,
    )
    geometry_result = geometry_pipeline.process(
        vision_result.detections,
        vision_result.relative_depth.depth_map,
        frame_id=run_id,
    )
    context = MediaFrameContext(
        upload_id=image_path.stem,
        frame_id=run_id,
        frame_index=0,
        timestamp=0.0,
        media_type=MediaType.IMAGE,
    )
    risk_result = risk_pipeline.process(
        _geometry_to_risk_inputs(geometry_result),
        context=context,
    )

    model_metadata = model_manager.metadata()
    model_metadata["pipeline"] = {
        "version": "vision-geometry-risk-annotation-v1",
        "mode": "image",
        "models_config": str(models_config_path),
        "geometry_config": str(geometry_config_path),
        "risk_config": str(risk_config_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_vision_artifacts(
        vision_result,
        image_bgr=image_bgr,
        image_path=image_path,
        output_dir=run_dir,
        model_metadata=model_metadata,
    )

    traceability = EvidenceTraceability(
        pipeline_version="vision-geometry-risk-annotation-v1",
        model_versions=_model_versions(model_metadata),
        config_versions={
            "models": _config_fingerprint(models_config_path),
            "geometry": _config_fingerprint(geometry_config_path),
            "risk": _config_fingerprint(risk_config_path),
        },
    )
    annotation_preview = annotation_composer.compose(
        image_bgr=image_bgr,
        detections=vision_result.detections,
        geometry=geometry_result,
        risk_result=risk_result,
        context=context,
        traceability=traceability,
    )
    annotation_preview_path = run_dir / "annotation_preview.png"
    if not cv2.imwrite(str(annotation_preview_path), annotation_preview):
        raise OSError(f"could not write annotation preview: {annotation_preview_path}")
    final_assessment_path = run_dir / "final_safety_assessment.png"
    _write_final_assessment_chart(
        image_bgr=image_bgr,
        detections=vision_result.detections,
        geometry=geometry_result,
        assessment=risk_result.assessment,
        output_path=final_assessment_path,
    )
    evidence = annotation_composer.write(
        image_bgr=image_bgr,
        detections=vision_result.detections,
        geometry=geometry_result,
        risk_result=risk_result,
        context=context,
        traceability=traceability,
        output_dir=run_dir,
    )
    timeline.write_json(run_dir / "pipeline_timeline.json")

    detection_counts: dict[str, int] = {}
    for detection in vision_result.detections:
        class_name = detection["class_name"]
        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
    print(f"output={run_dir}")
    print(f"detections={detection_counts}")
    print(f"risk={risk_result.assessment.level.value}")
    print(f"assessment_reliable={risk_result.assessment.assessment_reliable}")
    print(f"annotation_preview={annotation_preview_path}")
    print(f"final_assessment={final_assessment_path}")
    print(f"evidence={None if evidence is None else evidence.evidence_image_path}")
    print(f"timeline={run_dir / 'pipeline_timeline.json'}")
    return 0


def _write_final_assessment_chart(
    *,
    image_bgr: Any,
    detections: tuple[Detection, ...],
    geometry: GeometryFrameResult,
    assessment: FrameRiskAssessment,
    output_path: Path,
) -> None:
    """Save the same Camera + Pseudo-BEV assessment used by the notebook."""

    camera_overlay = render_image_overlay(image_bgr, detections, assessment)
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    try:
        axes[0].imshow(cv2.cvtColor(camera_overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Camera View - Detection and Status")
        axes[0].axis("off")
        draw_pseudo_bev_chart(axes[1], geometry, assessment)
        figure.suptitle("Suspended-Load Safety Assessment", fontsize=14)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def _geometry_to_risk_inputs(
    geometry: GeometryFrameResult,
) -> tuple[RiskPairInput, ...]:
    pair_inputs: list[RiskPairInput] = []
    for person_geometry in geometry.persons:
        point = person_geometry.pseudo_bev_point
        for load_geometry in geometry.loads:
            zones = load_geometry.safety_zones
            quality_reasons = tuple(
                dict.fromkeys(
                    (*person_geometry.quality_reasons, *load_geometry.quality_reasons)
                )
            )
            pair_inputs.append(
                RiskPairInput(
                    person=PersonObservation(
                        person_id=person_geometry.person_id,
                        anchor=(
                            None
                            if point is None
                            else Point2D(point.lateral, point.longitudinal)
                        ),
                        confidence=person_geometry.confidence,
                        anchor_reliable=point is not None,
                        mask_reliable=person_geometry.mask_reliable,
                        quality_reasons=quality_reasons,
                        track_id=person_geometry.track_id,
                    ),
                    load_id=load_geometry.load_id,
                    zones=(
                        None
                        if zones is None
                        else ZoneGeometry(
                            danger=_to_risk_rectangle(zones.danger),
                            warning=_to_risk_rectangle(zones.warning),
                        )
                    ),
                    load_track_id=load_geometry.track_id,
                )
            )
    return tuple(pair_inputs)


def _to_risk_rectangle(rectangle: PseudoBEVRectangle) -> Rectangle2D:
    return Rectangle2D(
        minimum_x=rectangle.minimum_lateral,
        maximum_x=rectangle.maximum_lateral,
        minimum_y=rectangle.minimum_longitudinal,
        maximum_y=rectangle.maximum_longitudinal,
    )


def _model_versions(metadata: Mapping[str, Any]) -> dict[str, str]:
    models = metadata.get("models", {})
    if not isinstance(models, Mapping):
        return {"models": "unknown"}
    return {
        str(name): str(values.get("identifier", values.get("name", "unknown")))
        if isinstance(values, Mapping)
        else str(values)
        for name, values in models.items()
    }


def _config_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _required_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_run_id(value: str | None) -> str:
    run_id = value or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError(
            "run-id must contain only letters, numbers, dot, underscore, or dash"
        )
    return run_id


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("config root must be a YAML mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
