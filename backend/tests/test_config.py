from __future__ import annotations

from pytest import MonkeyPatch

from app.core.config import Settings


def test_models_are_preloaded_by_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("CRANE_PRELOAD_MODELS", raising=False)

    assert Settings.from_environment().preload_models is True


def test_model_preload_can_be_disabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CRANE_PRELOAD_MODELS", "false")

    assert Settings.from_environment().preload_models is False
