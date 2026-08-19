"""Run the complete Phase-1 vision pipeline for one image."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine import (  # noqa: E402
    OfflineFramePipeline,
    build_model_manager,
    write_phase1_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR, YOLO26m-seg, and DA3 on one image."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="Parent directory for outputs/<run-id>",
    )
    parser.add_argument(
        "--run-id",
        help="Optional output directory name; defaults to a UTC timestamp",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all model weights before reading the input image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    config_path = args.config.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")

    config = _load_yaml(config_path)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError(
            "run-id must contain only letters, numbers, dot, underscore, or dash"
        )
    run_dir = args.output_root.resolve() / run_id

    manager = build_model_manager(config, config_dir=config_path.parent)
    if args.preload:
        manager.load_all()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode input image: {image_path}")

    pipeline = OfflineFramePipeline(manager)
    result = pipeline.process(image_bgr, image_path=image_path)
    metadata = manager.metadata()
    metadata["pipeline"] = {
        "phase": 1,
        "mode": "offline_single_image",
        "config_path": str(config_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_phase1_artifacts(
        result,
        image_bgr=image_bgr,
        image_path=image_path,
        output_dir=run_dir,
        model_metadata=metadata,
    )

    counts: dict[str, int] = {}
    for detection in result.detections:
        class_name = detection["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1
    depth_meta = result.relative_depth.metadata
    print(f"output={run_dir}")
    print(f"detections={counts}")
    print(
        "relative_depth="
        f"shape=({depth_meta.height}, {depth_meta.width}), "
        f"finite_range=({depth_meta.finite_min:.6f}, "
        f"{depth_meta.finite_max:.6f}), metric=false"
    )
    return 0


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("model config root must be a YAML mapping")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
