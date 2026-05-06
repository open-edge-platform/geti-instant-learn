# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from domain.services.schemas.model_status import (
    MODEL_STATUS_ERROR_PREFIX,
    MODEL_STATUS_MESSAGES,
    ModelState,
    ModelStatusSchema,
    sanitize_error_detail,
)


class TestModelStatusSchema:
    def test_idle_factory_sets_state_and_message(self):
        pid = uuid4()
        status = ModelStatusSchema.idle(project_id=pid)
        assert status.state == ModelState.IDLE
        assert status.project_id == pid
        assert status.message == MODEL_STATUS_MESSAGES[ModelState.IDLE]
        assert status.error is None

    def test_loading_reference_batch_factory(self):
        status = ModelStatusSchema.loading_reference_batch(project_id=uuid4(), model_name="sam3", device="cuda")
        assert status.state == ModelState.LOADING_REFERENCE_BATCH
        assert status.model_name == "sam3"
        assert status.device == "cuda"
        assert status.message == MODEL_STATUS_MESSAGES[ModelState.LOADING_REFERENCE_BATCH]

    def test_loading_model_factory_formats_message_with_descriptor(self):
        status = ModelStatusSchema.loading_model(project_id=uuid4(), model_name="sam3", device="cuda")
        assert status.state == ModelState.LOADING_MODEL
        assert status.model_name == "sam3"
        assert status.device == "cuda"
        assert "sam3" in status.message
        assert "cuda" in status.message

    def test_loading_model_factory_tolerates_missing_descriptor(self):
        status = ModelStatusSchema.loading_model()
        assert status.message  # no exception even with missing format kwargs
        assert "unknown" in status.message

    def test_ready_factory(self):
        status = ModelStatusSchema.ready(model_name="matcher", device="cpu")
        assert status.state == ModelState.READY
        assert "matcher" in status.message
        assert "cpu" in status.message

    def test_from_exception_captures_code_and_detail(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            status = ModelStatusSchema.from_exception(exc, model_name="sam3", device="cpu")
        assert status.state == ModelState.ERROR
        assert status.error is not None
        assert status.error.code == "RuntimeError"
        assert status.error.detail == "boom"
        assert status.message.startswith(MODEL_STATUS_ERROR_PREFIX)


class TestHelpers:
    def test_sanitize_error_detail_truncates_long_messages(self):
        long_msg = "x" * 1000
        result = sanitize_error_detail(long_msg)
        assert len(result) <= 501  # 500 + ellipsis
        assert result.endswith("…")

    def test_sanitize_error_detail_strips_newlines(self):
        result = sanitize_error_detail("line one\nline two")
        assert "\n" not in result
        assert result == "line one line two"
