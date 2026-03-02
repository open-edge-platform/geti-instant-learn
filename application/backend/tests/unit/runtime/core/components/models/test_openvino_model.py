# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from runtime.core.components.models.openvino_model import OpenVINOModelHandler


class TestOpenVINOModelHandler:
    def test_cleanup_frees_references_and_collects_garbage(self):
        handler = OpenVINOModelHandler(MagicMock(), MagicMock())

        with patch("runtime.core.components.models.openvino_model.gc") as mock_gc:
            handler.cleanup()

        mock_gc.collect.assert_called_once()
        assert handler._model is None
        assert handler._reference_batch is None

    def test_cleanup_is_safe_to_call_twice(self):
        handler = OpenVINOModelHandler(MagicMock(), MagicMock())

        handler.cleanup()
        handler.cleanup()

        assert handler._model is None
        assert handler._reference_batch is None
