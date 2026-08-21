"""Utilities for preparing relative-depth maps for geometry processing."""

from __future__ import annotations

import math

import numpy as np


def calculate_depth_normalization_range(
    depth_map: np.ndarray,
    *,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
) -> tuple[float, float]:
    """Calculate robust shared depth bounds from one frame's finite values."""

    if (
        not math.isfinite(lower_percentile)
        or not math.isfinite(upper_percentile)
        or not 0.0 <= lower_percentile < upper_percentile <= 100.0
    ):
        raise ValueError(
            "depth percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100"
        )

    finite_depth = np.asarray(depth_map, dtype=float)
    finite_depth = finite_depth[np.isfinite(finite_depth)]
    if finite_depth.size == 0:
        raise ValueError("depth_map must contain at least one finite value")

    depth_low, depth_high = np.percentile(
        finite_depth,
        [lower_percentile, upper_percentile],
    )
    return float(depth_low), float(depth_high)


__all__ = ["calculate_depth_normalization_range"]
