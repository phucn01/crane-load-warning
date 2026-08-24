"""Render relative geometry in a dedicated explanatory Pseudo-BEV panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from geometry_engine import GeometryFrameResult, PseudoBEVPoint, PseudoBEVRectangle
from numpy.typing import NDArray
from risk_engine import FrameRiskAssessment, RiskLevel

from .image_overlay import RISK_COLORS_BGR

BACKGROUND_BGR = (255, 255, 255)
GRID_BGR = (225, 225, 225)
AXIS_BGR = (90, 90, 90)
FOOTPRINT_BGR = (168, 120, 76)
WARNING_BGR = (11, 158, 245)
DANGER_BGR = (40, 39, 214)
PERSON_BGR = (166, 166, 0)
RISK_COLORS_HEX = {
    RiskLevel.SAFE: "#2CA02C",
    RiskLevel.WARNING: "#F59E0B",
    RiskLevel.DANGER: "#D62728",
}


@dataclass(frozen=True, slots=True)
class _Viewport:
    center_lateral: float
    center_longitudinal: float
    scale: float
    center_x: float
    center_y: float
    left: int
    right: int
    top: int
    bottom: int

    def project(self, point: PseudoBEVPoint) -> tuple[int, int]:
        x = self.center_x + (point.lateral - self.center_lateral) * self.scale
        y = self.center_y - (point.longitudinal - self.center_longitudinal) * self.scale
        return round(x), round(y)


def render_pseudo_bev_overlay(
    geometry: GeometryFrameResult,
    assessment: FrameRiskAssessment,
    *,
    width: int = 640,
    height: int = 640,
) -> NDArray[np.uint8]:
    """Render zones, load footprints, and representative person points."""

    if width < 160 or height < 160:
        raise ValueError("Pseudo-BEV dimensions must be at least 160 pixels")
    canvas = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)
    viewport = _build_viewport(geometry, width=width, height=height)
    _draw_plot_background(canvas, viewport)
    legend_entries: list[tuple[str, tuple[int, int, int], str]] = []

    for load_index, load in enumerate(geometry.loads, start=1):
        zones = load.safety_zones
        if zones is not None:
            _fill_rectangle(canvas, zones.warning, viewport, WARNING_BGR, 0.10)
            _draw_dashed_rectangle(canvas, zones.warning, viewport, WARNING_BGR)
            _fill_rectangle(canvas, zones.danger, viewport, DANGER_BGR, 0.18)
            _outline_rectangle(canvas, zones.danger, viewport, DANGER_BGR, 2)
            _fill_rectangle(canvas, zones.footprint, viewport, FOOTPRINT_BGR, 0.18)
            _outline_rectangle(canvas, zones.footprint, viewport, FOOTPRINT_BGR, 2)
            legend_entries.extend(
                (
                    (f"L{load_index} Warning Zone", WARNING_BGR, "dashed"),
                    (f"L{load_index} Danger Zone", DANGER_BGR, "line"),
                    (f"L{load_index} footprint", FOOTPRINT_BGR, "line"),
                )
            )

    person_levels = _person_levels(assessment)
    label_offsets = ((7, -7), (7, 14), (-74, -7), (-74, 14))
    for person_index, person in enumerate(geometry.persons, start=1):
        point = person.pseudo_bev_point
        if point is None:
            continue
        level = person_levels.get(person.person_id, RiskLevel.WARNING)
        pixel = viewport.project(point)
        risk_color = RISK_COLORS_BGR[level]
        cv2.circle(canvas, pixel, 7, PERSON_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, pixel, 8, risk_color, 2, cv2.LINE_AA)
        offset_x, offset_y = label_offsets[(person_index - 1) % len(label_offsets)]
        cv2.putText(
            canvas,
            f"P{person_index} - {level.value}",
            (pixel[0] + offset_x, pixel[1] + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            risk_color,
            2,
            cv2.LINE_AA,
        )
        legend_entries.append(
            (f"Person P{person_index} - {level.value}", risk_color, "person")
        )

    _draw_camera(canvas, viewport)
    legend_entries.append(("Camera", (25, 25, 25), "camera"))
    _draw_legend(canvas, viewport, legend_entries)
    _draw_title_and_axes(canvas, viewport)
    return canvas


def draw_pseudo_bev_chart(
    axis: Any,
    geometry: GeometryFrameResult,
    assessment: FrameRiskAssessment,
    *,
    title: str = "Pseudo-BEV Safety View",
) -> None:
    """Draw a research-style Pseudo-BEV chart on a Matplotlib-compatible axis.

    Keeping this adapter here gives notebooks the same zone semantics as the
    production evidence renderer without making Matplotlib a backend dependency.
    """

    zone_styles = (
        ("warning", "#F59E0B", 0.10, "--", "Warning Zone"),
        ("danger", "#D62728", 0.18, "-", "Danger Zone"),
        ("footprint", "#4C78A8", 0.18, "-", "Estimated footprint"),
    )
    for load_index, load in enumerate(geometry.loads, start=1):
        zones = load.safety_zones
        if zones is None:
            continue
        for field, color, alpha, line_style, label in zone_styles:
            polygon = _closed_rectangle_points(getattr(zones, field))
            axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color=color,
                alpha=alpha,
            )
            axis.plot(
                polygon[:, 0],
                polygon[:, 1],
                color=color,
                linestyle=line_style,
                linewidth=2,
                label=f"L{load_index} {label}",
            )

    person_levels = _person_levels(assessment)
    label_offsets = ((6, 6), (6, -14), (-56, 6), (-56, -14))
    for person_index, person in enumerate(geometry.persons, start=1):
        point = person.pseudo_bev_point
        if point is None:
            continue
        level = person_levels.get(person.person_id, RiskLevel.WARNING)
        risk_color = RISK_COLORS_HEX[level]
        axis.scatter(
            [point.lateral],
            [point.longitudinal],
            s=85,
            c="#00A6A6",
            edgecolors=risk_color,
            linewidths=2,
            zorder=5,
            label=f"Person P{person_index} - {level.value}",
        )
        axis.annotate(
            f"P{person_index} - {level.value}",
            (point.lateral, point.longitudinal),
            xytext=label_offsets[(person_index - 1) % len(label_offsets)],
            textcoords="offset points",
            color=risk_color,
            fontsize=9,
            fontweight="bold",
        )

    axis.scatter(
        [0.0],
        [0.0],
        marker="x",
        s=75,
        c="#191919",
        linewidths=2,
        zorder=6,
        label="Camera",
    )
    axis.set_title(title)
    axis.set_xlabel("Relative lateral position")
    axis.set_ylabel("Relative longitudinal position")
    axis.grid(alpha=0.25)
    axis.set_aspect("equal", adjustable="datalim")
    axis.autoscale_view()
    axis.margins(0.10)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best")


def _build_viewport(
    geometry: GeometryFrameResult,
    *,
    width: int,
    height: int,
) -> _Viewport:
    points = [PseudoBEVPoint(0.0, 0.0)]
    for person in geometry.persons:
        if person.pseudo_bev_point is not None:
            points.append(person.pseudo_bev_point)
    for load in geometry.loads:
        if load.safety_zones is not None:
            points.extend(load.safety_zones.warning.corners)

    lateral_values = [point.lateral for point in points]
    longitudinal_values = [point.longitudinal for point in points]
    minimum_lateral, maximum_lateral = min(lateral_values), max(lateral_values)
    minimum_longitudinal = min(longitudinal_values)
    maximum_longitudinal = max(longitudinal_values)
    lateral_span = max(maximum_lateral - minimum_lateral, 1.0e-6)
    longitudinal_span = max(maximum_longitudinal - minimum_longitudinal, 1.0e-6)
    left = max(42, round(width * 0.13))
    right = width - max(14, round(width * 0.03))
    top = max(40, round(height * 0.10))
    bottom = height - max(48, round(height * 0.12))
    plot_width = max(1.0, right - left)
    plot_height = max(1.0, bottom - top)
    scale = min(plot_width / (lateral_span * 1.2), plot_height / (longitudinal_span * 1.2))
    return _Viewport(
        center_lateral=(minimum_lateral + maximum_lateral) / 2.0,
        center_longitudinal=(minimum_longitudinal + maximum_longitudinal) / 2.0,
        scale=scale,
        center_x=(left + right) / 2.0,
        center_y=(top + bottom) / 2.0,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )


def _draw_plot_background(canvas: NDArray[np.uint8], viewport: _Viewport) -> None:
    for index in range(5):
        fraction = index / 4.0
        x = round(viewport.left + fraction * (viewport.right - viewport.left))
        y = round(viewport.top + fraction * (viewport.bottom - viewport.top))
        cv2.line(
            canvas,
            (x, viewport.top),
            (x, viewport.bottom),
            GRID_BGR,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (viewport.left, y),
            (viewport.right, y),
            GRID_BGR,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas,
        (viewport.left, viewport.top),
        (viewport.right, viewport.bottom),
        AXIS_BGR,
        1,
        cv2.LINE_AA,
    )


def _draw_camera(canvas: NDArray[np.uint8], viewport: _Viewport) -> None:
    origin = viewport.project(PseudoBEVPoint(0.0, 0.0))
    cv2.drawMarker(
        canvas,
        origin,
        (25, 25, 25),
        cv2.MARKER_TILTED_CROSS,
        12,
        2,
        cv2.LINE_AA,
    )


def _draw_title_and_axes(
    canvas: NDArray[np.uint8],
    viewport: _Viewport,
) -> None:
    _centered_text(
        canvas,
        "Pseudo-BEV Safety View",
        y=24,
        scale=0.42,
        color=(35, 35, 35),
    )
    _centered_text(
        canvas,
        "Relative lateral position",
        y=canvas.shape[0] - 8,
        scale=0.36,
        color=AXIS_BGR,
    )
    cv2.putText(
        canvas,
        "Relative longitudinal position",
        (4, viewport.top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        AXIS_BGR,
        1,
        cv2.LINE_AA,
    )


def _centered_text(
    canvas: NDArray[np.uint8],
    text: str,
    *,
    y: int,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    (text_width, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    cv2.putText(
        canvas,
        text,
        (max(3, (canvas.shape[1] - text_width) // 2), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _fill_rectangle(
    canvas: NDArray[np.uint8],
    rectangle: PseudoBEVRectangle,
    viewport: _Viewport,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    left, top, right, bottom = _rectangle_pixels(rectangle, viewport)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)


def _outline_rectangle(
    canvas: NDArray[np.uint8],
    rectangle: PseudoBEVRectangle,
    viewport: _Viewport,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    left, top, right, bottom = _rectangle_pixels(rectangle, viewport)
    cv2.rectangle(
        canvas,
        (left, top),
        (right, bottom),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_rectangle(
    canvas: NDArray[np.uint8],
    rectangle: PseudoBEVRectangle,
    viewport: _Viewport,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = _rectangle_pixels(rectangle, viewport)
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
        _draw_dashed_line(canvas, start, end, color=color, thickness=2)


def _draw_dashed_line(
    canvas: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
    thickness: int,
    dash_length: float = 9.0,
) -> None:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        return
    direction = delta / length
    position = 0.0
    while position < length:
        dash_end = min(position + dash_length, length)
        first = tuple(np.rint(start_array + direction * position).astype(int))
        second = tuple(np.rint(start_array + direction * dash_end).astype(int))
        cv2.line(canvas, first, second, color, thickness, cv2.LINE_AA)
        position += dash_length * 2.0


def _draw_legend(
    canvas: NDArray[np.uint8],
    viewport: _Viewport,
    entries: list[tuple[str, tuple[int, int, int], str]],
) -> None:
    if not entries:
        return
    line_height = 16
    maximum_entries = max(1, (viewport.bottom - viewport.top - 16) // line_height)
    visible_entries = entries[:maximum_entries]
    text_widths = [
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
        for label, _, _ in visible_entries
    ]
    legend_width = min(max(text_widths, default=80) + 42, viewport.right - viewport.left)
    legend_height = len(visible_entries) * line_height + 10
    left = viewport.left + 7
    top = viewport.bottom - legend_height - 7
    right = left + legend_width
    bottom = top + legend_height
    overlay = canvas.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0.0, dst=canvas)
    cv2.rectangle(canvas, (left, top), (right, bottom), (190, 190, 190), 1)

    for index, (label, color, style) in enumerate(visible_entries):
        center_y = top + 9 + index * line_height
        start, end = (left + 7, center_y), (left + 27, center_y)
        if style == "dashed":
            _draw_dashed_line(canvas, start, end, color=color, thickness=2, dash_length=4)
        elif style == "person":
            cv2.circle(canvas, (left + 17, center_y), 5, PERSON_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (left + 17, center_y), 6, color, 1, cv2.LINE_AA)
        elif style == "camera":
            cv2.drawMarker(
                canvas,
                (left + 17, center_y),
                color,
                cv2.MARKER_TILTED_CROSS,
                9,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (left + 33, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )


def _rectangle_pixels(
    rectangle: PseudoBEVRectangle,
    viewport: _Viewport,
) -> tuple[int, int, int, int]:
    first = viewport.project(
        PseudoBEVPoint(rectangle.minimum_lateral, rectangle.minimum_longitudinal)
    )
    second = viewport.project(
        PseudoBEVPoint(rectangle.maximum_lateral, rectangle.maximum_longitudinal)
    )
    left, right = sorted((first[0], second[0]))
    top, bottom = sorted((first[1], second[1]))
    return left, top, right, bottom


def _closed_rectangle_points(
    rectangle: PseudoBEVRectangle,
) -> NDArray[np.float32]:
    return np.asarray(
        (
            (rectangle.minimum_lateral, rectangle.minimum_longitudinal),
            (rectangle.maximum_lateral, rectangle.minimum_longitudinal),
            (rectangle.maximum_lateral, rectangle.maximum_longitudinal),
            (rectangle.minimum_lateral, rectangle.maximum_longitudinal),
            (rectangle.minimum_lateral, rectangle.minimum_longitudinal),
        ),
        dtype=np.float32,
    )


def _person_levels(assessment: FrameRiskAssessment) -> dict[str, RiskLevel]:
    levels: dict[str, RiskLevel] = {}
    for pair in assessment.pair_assessments:
        current = levels.get(pair.person_id)
        if current is None or pair.level.severity > current.severity:
            levels[pair.person_id] = pair.level
    return levels


__all__ = ["draw_pseudo_bev_chart", "render_pseudo_bev_overlay"]
