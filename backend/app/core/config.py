"""Environment-backed configuration for the local API process."""

from __future__ import annotations

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
    preload_models: bool = False

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
            max_upload_bytes=_positive_int_environment(
                "CRANE_MAX_UPLOAD_BYTES",
                20 * 1024 * 1024,
            ),
            preload_models=_boolean_environment("CRANE_PRELOAD_MODELS", False),
        )


def _environment_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


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


__all__ = ["PROJECT_ROOT", "Settings"]
