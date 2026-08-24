"""Contracts for deterministic offline safety evidence artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceTraceability:
    """Version identifiers needed to reproduce one assessment artifact."""

    pipeline_version: str
    model_versions: Mapping[str, str]
    config_versions: Mapping[str, str]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.pipeline_version:
            raise ValueError("pipeline_version must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        _validate_versions("model_versions", self.model_versions)
        _validate_versions("config_versions", self.config_versions)


@dataclass(frozen=True, slots=True)
class EvidenceArtifacts:
    """Paths written for one non-safe frame assessment."""

    evidence_image_path: Path
    assessment_json_path: Path


def _validate_versions(name: str, versions: Mapping[str, str]) -> None:
    if not isinstance(versions, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not str(key) or not str(value) for key, value in versions.items()):
        raise ValueError(f"{name} keys and values must not be empty")


__all__ = ["EvidenceArtifacts", "EvidenceTraceability"]
