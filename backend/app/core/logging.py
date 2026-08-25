"""Structured, request-aware logging and operation timing helpers."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.config import PROJECT_ROOT

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_CONFIGURED = False


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID.get()
        return True


def configure_logging() -> None:
    """Configure console and rotating-file logs once per process."""

    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("CRANE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_path = Path(
        os.getenv(
            "CRANE_LOG_FILE",
            str(PROJECT_ROOT / "backend" / "logs" / "backend.log"),
        )
    ).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
        "request_id=%(request_id)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    context_filter = _RequestContextFilter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(
        RotatingFileHandler(
            log_path,
            maxBytes=_positive_int_env("CRANE_LOG_MAX_BYTES", 10 * 1024 * 1024),
            backupCount=_positive_int_env("CRANE_LOG_BACKUP_COUNT", 5),
            encoding="utf-8",
        )
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "=== EVENT | LOGGING_CONFIGURED | LEVEL=%s | FILE=%s ===",
        level_name,
        log_path,
    )


def bind_request_id(request_id: str) -> Token[str]:
    return REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    REQUEST_ID.reset(token)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    **fields: Any,
) -> Iterator[None]:
    """Log start, success/failure, and elapsed time for one operation."""

    context = _format_field_lines(fields)
    logger.info(
        "=== START ===\n    OPERATION    : %s%s",
        operation.upper(),
        context,
    )
    started = perf_counter()
    try:
        yield
    except BaseException as error:
        duration_ms = (perf_counter() - started) * 1000.0
        logger.exception(
            "=== ERROR ===\n    OPERATION    : %s\n    DURATION_MS  : %.3f\n"
            "    ERROR_TYPE   : %s%s",
            operation.upper(),
            duration_ms,
            type(error).__name__,
            context,
        )
        raise
    else:
        duration_ms = (perf_counter() - started) * 1000.0
        logger.info(
            "=== END ===\n    OPERATION    : %s\n    DURATION_MS  : %.3f%s",
            operation.upper(),
            duration_ms,
            context,
        )


def _format_field_lines(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    return "".join(
        f"\n    {key.upper():<12} : {_safe_value(value)}"
        for key, value in sorted(fields.items())
    )


def _safe_value(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if text and not any(char.isspace() for char in text) else repr(text)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


__all__ = [
    "bind_request_id",
    "configure_logging",
    "log_operation",
    "reset_request_id",
]
