from __future__ import annotations

from pytest import MonkeyPatch

from app.core.config import Settings


def test_models_are_preloaded_by_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("CRANE_PRELOAD_MODELS", raising=False)

    assert Settings.from_environment().preload_models is True


def test_model_preload_can_be_disabled(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CRANE_PRELOAD_MODELS", "false")

    assert Settings.from_environment().preload_models is False


def test_supabase_database_settings_are_loaded(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("CRANE_DATABASE_JOB_UPDATE_INTERVAL", "5")

    settings = Settings.from_environment()

    assert settings.database_url == "postgresql://example"
    assert settings.database_job_update_interval == 5
    assert settings.persistence_config is not None
