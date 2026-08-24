"""Frame-level vision orchestration and offline artifact writing."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pipeline_timeline import PipelineTimeline

from .contracts import (
    Detection,
    VisionFrameResult,
    detection_to_dict,
)
from .model_manager import ModelManager

CLASS_COLORS_BGR = {
    "person": (166, 166, 0),
    "hanging_object": (40, 40, 214),
    "hanging_rope": (199, 74, 140),
}


class VisionFramePipeline:
    """Run all vision models exactly once for one image or video frame."""

    def __init__(
        self,
        model_manager: ModelManager,
        *,
        timeline: PipelineTimeline | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.timeline = timeline

    def process(
        self,
        image_bgr: NDArray[np.generic],
        *,
        frame_id: str,
        image_path: str | Path | None = None,
    ) -> VisionFrameResult:
        if self.timeline is not None:
            with self.timeline.track("vision", "process", frame_id=frame_id):
                return self._process(
                    image_bgr,
                    frame_id=frame_id,
                    image_path=image_path,
                )
        return self._process(
            image_bgr,
            frame_id=frame_id,
            image_path=image_path,
        )

    def _process(
        self,
        image_bgr: NDArray[np.generic],
        *,
        frame_id: str,
        image_path: str | Path | None,
    ) -> VisionFrameResult:
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        _validate_image(image_bgr)
        load_detector = self.model_manager.get("rfdetr")
        person_segmenter = self.model_manager.get("yolo_person")
        depth_estimator = self.model_manager.get("depth_anything_v3")

        load_detections = load_detector.predict(image_bgr)
        person_detections = person_segmenter.predict(image_bgr)
        relative_depth = depth_estimator.predict(
            image_bgr,
            image_path=image_path,
        )
        return VisionFrameResult(
            frame_id=frame_id,
            detections=tuple(load_detections + person_detections),
            relative_depth=relative_depth,
        )


def write_vision_artifacts(
    result: VisionFrameResult,
    *,
    image_bgr: NDArray[np.generic],
    image_path: str | Path,
    output_dir: str | Path,
    model_metadata: dict[str, Any],
) -> Path:
    """Write detections, depth, metadata, masks, and a basic preview."""

    run_dir = Path(output_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = run_dir / "masks"

    counters: defaultdict[str, int] = defaultdict(int)
    json_detections: list[dict[str, Any]] = []
    for detection in result.detections:
        class_name = detection["class_name"]
        counters[class_name] += 1
        detection_id = f"{class_name}_{counters[class_name]:02d}"
        mask_ref: str | None = None
        mask = detection["mask"]
        if isinstance(mask, np.ndarray) and mask.shape == image_bgr.shape[:2]:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_path = masks_dir / f"{detection_id}.npy"
            np.save(mask_path, mask.astype(bool), allow_pickle=False)
            mask_ref = mask_path.relative_to(run_dir).as_posix()
        json_detections.append(
            detection_to_dict(
                detection,
                detection_id=detection_id,
                mask_ref=mask_ref,
            )
        )

    depth_path = run_dir / "relative_depth.npy"
    np.save(
        depth_path,
        result.relative_depth.depth_map.astype(np.float32),
        allow_pickle=False,
    )
    detections_payload = {
        "schema_version": "1.0",
        "frame_id": result.frame_id,
        "source_image": str(Path(image_path).resolve()),
        "image": {
            "height": int(image_bgr.shape[0]),
            "width": int(image_bgr.shape[1]),
            "channels": int(image_bgr.shape[2]),
        },
        "detections": json_detections,
        "relative_depth": {
            "artifact_ref": depth_path.name,
            **result.relative_depth.metadata.to_dict(),
        },
    }
    _write_json(run_dir / "detections.json", detections_payload)
    _write_json(run_dir / "model_metadata.json", model_metadata)

    preview = render_detection_preview(image_bgr, result.detections)
    preview_path = run_dir / "detection_preview.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"could not write detection preview: {preview_path}")
    return run_dir


def render_detection_preview(
    image_bgr: NDArray[np.generic],
    detections: Sequence[Detection],
) -> NDArray[np.uint8]:
    output = image_bgr.copy()
    for detection in detections:
        color = CLASS_COLORS_BGR.get(detection["class_name"], (255, 255, 255))
        mask = detection["mask"]
        has_person_mask = (
            detection["class_name"] == "person"
            and isinstance(mask, np.ndarray)
            and mask.shape == output.shape[:2]
            and bool(np.any(mask))
        )
        if has_person_mask:
            pixels = output[mask].astype(np.float32)
            overlay_color = np.asarray(color, dtype=np.float32)
            output[mask] = np.clip(
                0.70 * pixels + 0.30 * overlay_color, 0, 255
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)

        x1, y1, x2, y2 = (round(value) for value in detection["bbox"])
        # Draw a person bbox only when its mask is missing.
        if not has_person_mask:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_image(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("pipeline image must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be greater than zero")


__all__ = [
    "VisionFramePipeline",
    "render_detection_preview",
    "write_vision_artifacts",
]
