"""Depth Anything V3 adapter for one relative-depth inference per frame."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from ..contracts import DepthMapMetadata, RelativeDepthResult


@dataclass(frozen=True, slots=True)
class DepthAnythingV3Config:
    model_name: str = "depth-anything/DA3-BASE"
    device: str | None = None
    process_resolution: int = 504
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.process_resolution <= 0:
            raise ValueError("process_resolution must be greater than zero")


class DepthAnythingV3:
    """Produce a float32, image-sized, relative (non-metric) depth map."""

    def __init__(
        self,
        *,
        config: DepthAnythingV3Config | None = None,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or DepthAnythingV3Config()
        self._model = model
        self._model_factory = model_factory
        self._lock = Lock()
        self._inference_count = 0

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        factory = self._model_factory
        if factory is None:
            # Monocular depth does not need the optional pycolmap exporter.
            previous_pycolmap = sys.modules.get("pycolmap")
            sys.modules["pycolmap"] = types.ModuleType("pycolmap")
            try:
                from depth_anything_3.api import DepthAnything3
            except ModuleNotFoundError as exc:  # pragma: no cover
                missing_module = exc.name or "unknown"
                if missing_module.startswith("depth_anything_3"):
                    message = (
                        "depth-anything-3 is required for relative-depth inference"
                    )
                else:
                    message = (
                        "Depth Anything V3 dependency is missing: "
                        f"{missing_module}. Reinstall the backend project dependencies."
                    )
                raise RuntimeError(message) from exc
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Depth Anything V3 import failed: {exc}"
                ) from exc
            finally:
                if previous_pycolmap is None:
                    sys.modules.pop("pycolmap", None)
                else:
                    sys.modules["pycolmap"] = previous_pycolmap
            factory = DepthAnything3.from_pretrained

        model = factory(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
        )
        if self.config.device is not None and hasattr(model, "to"):
            model.to(device=self.config.device)
        if hasattr(model, "eval"):
            model.eval()
        self._model = model
        return model

    def load(self) -> "DepthAnythingV3":
        with self._lock:
            self._get_model()
        return self

    def predict(
        self,
        image_bgr: NDArray[np.generic],
        *,
        image_path: str | Path | None = None,
    ) -> RelativeDepthResult:
        _validate_image(image_bgr)
        source: str | NDArray[np.uint8]
        if image_path is not None:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Depth input image not found: {path}")
            source = str(path)
        else:
            source = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            model = self._get_model()
            prediction = _run_inference(
                model,
                [source],
                process_resolution=self.config.process_resolution,
            )
            self._inference_count += 1

        raw_depth = np.asarray(prediction.depth[0], dtype=np.float32)
        target_shape = image_bgr.shape[:2]
        if raw_depth.shape != target_shape:
            raw_depth = cv2.resize(
                raw_depth,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        finite = raw_depth[np.isfinite(raw_depth)]
        if finite.size == 0:
            raise ValueError(
                "Depth Anything V3 returned no finite relative-depth values"
            )
        metadata = DepthMapMetadata(
            height=target_shape[0],
            width=target_shape[1],
            dtype=str(raw_depth.dtype),
            finite_min=float(finite.min()),
            finite_max=float(finite.max()),
            finite_fraction=float(finite.size / raw_depth.size),
        )
        return RelativeDepthResult(depth_map=raw_depth, metadata=metadata)

    __call__ = predict

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "Depth Anything V3",
            "provider": "depth-anything-3",
            "identifier": self.config.model_name,
            "device": self.config.device,
            "process_resolution": self.config.process_resolution,
            "depth_convention": "relative_depth_not_metric",
            "loaded": self._model is not None,
            "inference_count": self._inference_count,
        }


def _run_inference(model: Any, sources: list[Any], *, process_resolution: int) -> Any:
    try:
        import torch
    except ImportError:
        return model.inference(sources, process_res=process_resolution)
    with torch.inference_mode():
        return model.inference(sources, process_res=process_resolution)


def _validate_image(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("depth image must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be greater than zero")


__all__ = ["DepthAnythingV3", "DepthAnythingV3Config"]
