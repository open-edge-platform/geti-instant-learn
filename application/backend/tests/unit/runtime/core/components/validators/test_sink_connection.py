from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.validators.sink_connection import SinkConnectionValidator
from runtime.errors import SinkConnectionError


class TestSinkConnectionValidator:
    @staticmethod
    @contextmanager
    def _writer_context(writer: Mock):
        try:
            yield writer
        finally:
            writer.close()

    def test_validate_calls_connect_and_cleanup(self):
        validator = SinkConnectionValidator()
        writer = Mock()
        writer.connect.return_value = None
        writer.close.return_value = None
        config = WriterConfig(broker_host="localhost", broker_port=1883, topic="test")

        with patch(
            "runtime.core.components.validators.sink_connection.StreamWriterFactory.create",
            return_value=self._writer_context(writer),
        ):
            validator.validate(config=config)

        writer.connect.assert_called_once()
        writer.close.assert_called_once()

    def test_validate_raises_resource_connection_error(self):
        validator = SinkConnectionValidator()
        writer = Mock()
        writer.connect.side_effect = ConnectionError("Failed to connect")
        writer.close.return_value = None
        config = WriterConfig(broker_host="localhost", broker_port=1883, topic="test")

        with patch(
            "runtime.core.components.validators.sink_connection.StreamWriterFactory.create",
            return_value=self._writer_context(writer),
        ):
            with pytest.raises(SinkConnectionError):
                validator.validate(config=config)

        writer.close.assert_called_once()
