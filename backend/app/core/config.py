"""Environment-backed configuration for the local API process."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Settings:
    models_config: Path
    geometry_config: Path
    risk_config: Path
    evidence_root: Path
    max_upload_bytes: int = 20 * 1024 * 1024
    video_upload_root: Path | None = None
    video_output_root: Path | None = None
    max_video_upload_bytes: int = 500 * 1024 * 1024
    risk_segment_pre_roll_seconds: float = 2.0
    risk_segment_post_roll_seconds: float = 2.0
    ffmpeg_path: Path | None = None
    preload_models: bool = False
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            models_config=_environment_path(
                "CRANE_MODELS_CONFIG",
                PROJECT_ROOT / "configs" / "models.local.yaml",
            ),
            geometry_config=_environment_path(
                "CRANE_GEOMETRY_CONFIG",
                PROJECT_ROOT / "configs" / "geometry.local.yaml",
            ),
            risk_config=_environment_path(
                "CRANE_RISK_CONFIG",
                PROJECT_ROOT / "configs" / "risk-policy.example.yaml",
            ),
            evidence_root=_environment_path(
                "CRANE_EVIDENCE_ROOT",
                PROJECT_ROOT / "backend" / "storage" / "evidence",
            ),
            video_upload_root=_environment_path(
                "CRANE_VIDEO_UPLOAD_ROOT",
                PROJECT_ROOT / "backend" / "storage" / "uploads" / "videos",
            ),
            video_output_root=_environment_path(
                "CRANE_VIDEO_OUTPUT_ROOT",
                PROJECT_ROOT / "backend" / "storage" / "outputs" / "videos",
            ),
            max_video_upload_bytes=_positive_int_environment(
                "CRANE_MAX_VIDEO_UPLOAD_BYTES",
                500 * 1024 * 1024,
            ),
            risk_segment_pre_roll_seconds=_non_negative_float_environment(
                "CRANE_RISK_SEGMENT_PRE_ROLL_SECONDS",
                2.0,
            ),
            risk_segment_post_roll_seconds=_non_negative_float_environment(
                "CRANE_RISK_SEGMENT_POST_ROLL_SECONDS",
                2.0,
            ),
            ffmpeg_path=_optional_environment_path("CRANE_FFMPEG_PATH"),
            max_upload_bytes=_positive_int_environment(
                "CRANE_MAX_UPLOAD_BYTES",
                20 * 1024 * 1024,
            ),
            preload_models=_boolean_environment("CRANE_PRELOAD_MODELS", False),
            cors_origins=_origins_environment(
                "CRANE_CORS_ORIGINS",
                ("http://localhost:5173", "http://127.0.0.1:5173"),
            ),
        )


def _environment_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _optional_environment_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean_environment(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _non_negative_float_environment(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _origins_environment(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    origins = tuple(
        dict.fromkeys(
            item.strip().rstrip("/") for item in raw.split(",") if item.strip()
        )
    )
    if not origins:
        raise ValueError(f"{name} must contain at least one origin")
    if "*" in origins:
        raise ValueError(f"{name} must not use a wildcard origin")
    return origins


__all__ = ["PROJECT_ROOT", "Settings"]
