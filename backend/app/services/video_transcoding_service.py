"""Best-effort browser-compatible finalization for generated MP4 files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoCompatibility:
    codec: str
    browser_playback_compatible: bool
    warning: str | None = None


class BrowserVideoConverter:
    """Convert a released OpenCV MP4 to H.264/yuv420p/faststart in place."""

    def __init__(self, ffmpeg_executable: Path | None) -> None:
        self.ffmpeg_executable = ffmpeg_executable

    @classmethod
    def discover(cls, explicit_path: Path | None = None) -> BrowserVideoConverter:
        executable = _resolve_executable(explicit_path)
        return cls(executable)

    def convert(self, path: Path) -> VideoCompatibility:
        if self.ffmpeg_executable is None:
            LOGGER.warning(
                "=== WARNING | VIDEO_TRANSCODE_SKIPPED | "
                "REASON=FFMPEG_UNAVAILABLE ==="
            )
            return _mp4v_fallback("FFmpeg is unavailable; browser playback may fail")

        temporary = path.with_name(f".{path.stem}.h264.tmp.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            str(self.ffmpeg_executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=600,
            )
            if completed.returncode != 0 or not temporary.is_file():
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                LOGGER.warning(
                    "=== ERROR | OPERATION=VIDEO_TRANSCODE | DETAIL=%s ===",
                    detail[-500:],
                )
                return _mp4v_fallback("H.264 conversion failed; browser playback may fail")
            if temporary.stat().st_size <= 0:
                return _mp4v_fallback(
                    "H.264 conversion produced an empty file; browser playback may fail"
                )
            temporary.replace(path)
            LOGGER.info(
                "=== END | OPERATION=VIDEO_TRANSCODE | CODEC=H264 | "
                "OUTPUT_BYTES=%s ===",
                path.stat().st_size,
            )
            return VideoCompatibility(
                codec="h264",
                browser_playback_compatible=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning(
                "=== ERROR | OPERATION=VIDEO_TRANSCODE | ERROR_TYPE=%s ===",
                type(error).__name__,
            )
            return _mp4v_fallback("H.264 conversion could not run; browser playback may fail")
        finally:
            temporary.unlink(missing_ok=True)


def _resolve_executable(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if resolved.is_file():
            return resolved
        LOGGER.warning(
            "=== WARNING | FFMPEG_NOT_FOUND | PATH=%s ===", resolved
        )
    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return Path(system_executable).resolve()
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        return bundled if bundled.is_file() else None
    except (ImportError, OSError, RuntimeError):
        return None


def _mp4v_fallback(warning: str) -> VideoCompatibility:
    return VideoCompatibility(
        codec="mp4v",
        browser_playback_compatible=False,
        warning=warning,
    )


__all__ = ["BrowserVideoConverter", "VideoCompatibility"]
