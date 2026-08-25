import logging

import pytest

from app.core.logging import log_operation


def test_log_operation_records_completion_duration_and_context(caplog):
    logger = logging.getLogger("test.operation.completed")

    with caplog.at_level(logging.INFO), log_operation(
        logger, "sample_stage", frame_id="frame-1"
    ):
        pass

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "=== START ===\n    OPERATION    : SAMPLE_STAGE\n"
        "    FRAME_ID     : frame-1" in message
        for message in messages
    )
    assert any(
        "=== END ===\n    OPERATION    : SAMPLE_STAGE\n"
        "    DURATION_MS  :" in message
        and "FRAME_ID     : frame-1" in message
        for message in messages
    )


def test_log_operation_records_failure_and_preserves_exception(caplog):
    logger = logging.getLogger("test.operation.failed")

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(ValueError, match="bad stage"),
        log_operation(logger, "sample_stage"),
    ):
        raise ValueError("bad stage")

    assert any(
        "=== ERROR ===\n    OPERATION    : SAMPLE_STAGE\n"
        "    DURATION_MS  :" in record.getMessage()
        and "ERROR_TYPE   : ValueError" in record.getMessage()
        for record in caplog.records
    )
