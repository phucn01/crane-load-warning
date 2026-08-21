import math

import numpy as np
import pytest
from geometry_engine.depth_utils import calculate_depth_normalization_range


def test_defaults_to_full_finite_scene_depth_range():
    depth_map = np.append(np.arange(101, dtype=float), [math.nan, math.inf])

    assert calculate_depth_normalization_range(depth_map) == (0.0, 100.0)


def test_accepts_named_depth_percentiles():
    depth_map = np.arange(101, dtype=float)

    assert calculate_depth_normalization_range(
        depth_map,
        lower_percentile=0,
        upper_percentile=90,
    ) == (0.0, 90.0)


def test_requires_depth_percentiles_to_be_keyword_arguments():
    with pytest.raises(TypeError):
        calculate_depth_normalization_range(  # type: ignore[call-arg]
            np.arange(101, dtype=float),
            0,
            90,
        )


def test_rejects_depth_map_without_finite_values():
    with pytest.raises(ValueError, match="at least one finite value"):
        calculate_depth_normalization_range(np.array([math.nan, math.inf]))


def test_rejects_invalid_depth_percentiles():
    with pytest.raises(ValueError, match="depth percentiles"):
        calculate_depth_normalization_range(
            np.array([1.0]),
            lower_percentile=99.0,
            upper_percentile=1.0,
        )
