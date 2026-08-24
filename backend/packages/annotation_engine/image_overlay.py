"""Explain detections and frame risk on the source RGB camera view."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from risk_engine import FrameRiskAssessment, RiskLevel
from vision_engine.contracts import Detection, clip_bbox

CLASS_COLORS_BGR = {
    "person": (166, 166, 0),
    "hanging_object": (40, 40, 214),
    "hanging_rope": (199, 74, 140),
    "rope": (199, 74, 140),
}
RISK_COLORS_BGR = {
    RiskLevel.SAFE: (81, 166, 0),
    RiskLevel.WARNING: (11, 158, 245),
    RiskLevel.DANGER: (40, 40, 214),
}


def render_image_overlay(
    image_bgr: NDArray[np.generic],
    detections: Iterable[Detection],
    assessment: FrameRiskAssessment,
) -> NDArray[np.uint8]:
    """Render segmentation when available, otherwise bbox, without physical zones."""

    output = _uint8_bgr_image(image_bgr)
    person_levels = _highest_person_levels(assessment)
    counters: defaultdict[str, int] = defaultdict(int)

    for detection in detections:
        class_name = detection["class_name"]
        counters[class_name] += 1
        entity_id = f"{class_name}_{counters[class_name]:02d}"
        level = None
        if class_name == "person":
            level = person_levels.get(entity_id, assessment.level)
        _draw_detection(output, detection, entity_id=entity_id, level=level)

    _draw_frame_banner(output, assessment)
    return output


def _draw_detection(
    output: NDArray[np.uint8],
    detection: Detection,
    *,
    entity_id: str,
    level: RiskLevel | None,
) -> None:
    bbox = clip_bbox(detection["bbox"], output.shape)
    if bbox is None:
        return
    color = CLASS_COLORS_BGR.get(detection["class_name"], (230, 230, 230))
    border_color = RISK_COLORS_BGR.get(level, color)
    mask = detection["mask"]
    has_valid_mask = bool(
        isinstance(mask, np.ndarray)
        and mask.shape == output.shape[:2]
        and np.any(mask)
    )
    if has_valid_mask:
        mask_bool = mask.astype(bool)
        pixels = output[mask_bool].astype(np.float32)
        overlay_color = np.asarray(border_color, dtype=np.float32)
        output[mask_bool] = np.clip(
            0.7 * pixels + 0.3 * overlay_color,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, border_color, 2, cv2.LINE_AA)

    x1, y1, x2, y2 = (round(value) for value in bbox)
    if not has_valid_mask:
        cv2.rectangle(output, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
    label = f"{entity_id} {float(detection['confidence']):.2f}"
    if level is not None:
        label = f"{label} {level.value}"
    _draw_text_label(output, label, x=x1, y=y1, color=border_color)


def _draw_frame_banner(
    output: NDArray[np.uint8], assessment: FrameRiskAssessment
) -> None:
    color = RISK_COLORS_BGR[assessment.level]
    banner_height = min(30, output.shape[0])
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), color, -1)
    reliability = "reliable" if assessment.assessment_reliable else "unreliable"
    text = f"FRAME {assessment.level.value} | {reliability}"
    cv2.putText(
        output,
        text,
        (7, min(21, banner_height - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_text_label(
    output: NDArray[np.uint8],
    text: str,
    *,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    origin_y = max(44, y - 5)
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    )
    cv2.rectangle(
        output,
        (x, origin_y - height - baseline - 3),
        (min(output.shape[1] - 1, x + width + 4), origin_y + 2),
        color,
        -1,
    )
    cv2.putText(
        output,
        text,
        (x + 2, origin_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _highest_person_levels(
    assessment: FrameRiskAssessment,
) -> dict[str, RiskLevel]:
    levels: dict[str, RiskLevel] = {}
    for pair in assessment.pair_assessments:
        current = levels.get(pair.person_id)
        if current is None or pair.level.severity > current.severity:
            levels[pair.person_id] = pair.level
    return levels


def _uint8_bgr_image(image: Any) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image_bgr must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be greater than zero")
    return np.clip(image, 0, 255).astype(np.uint8, copy=True)


__all__ = [
    "CLASS_COLORS_BGR",
    "RISK_COLORS_BGR",
    "render_image_overlay",
]
