from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.services import video_transcoding_service
from app.services.video_transcoding_service import BrowserVideoConverter


def test_unavailable_ffmpeg_keeps_mp4v_with_warning(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4v-source")

    result = BrowserVideoConverter(None).convert(source)

    assert source.read_bytes() == b"mp4v-source"
    assert result.codec == "mp4v"
    assert result.browser_playback_compatible is False
    assert result.warning is not None


def test_successful_conversion_atomically_replaces_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"binary")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4v-source")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        commands.append(command)
        Path(command[-1]).write_bytes(b"h264-output")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(video_transcoding_service.subprocess, "run", fake_run)

    result = BrowserVideoConverter(executable).convert(source)

    assert source.read_bytes() == b"h264-output"
    assert result.codec == "h264"
    assert result.browser_playback_compatible is True
    assert result.warning is None
    assert "libx264" in commands[0]
    assert "yuv420p" in commands[0]
    assert "+faststart" in commands[0]


def test_failed_conversion_preserves_original_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"binary")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4v-source")

    monkeypatch.setattr(
        video_transcoding_service.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=b"failure"),
    )

    result = BrowserVideoConverter(executable).convert(source)

    assert source.read_bytes() == b"mp4v-source"
    assert result.codec == "mp4v"
    assert result.browser_playback_compatible is False
    assert result.warning is not None
