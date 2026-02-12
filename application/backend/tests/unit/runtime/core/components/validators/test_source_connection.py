from unittest.mock import MagicMock, patch

import pytest

from domain.services.schemas.reader import SourceType, UsbCameraConfig
from runtime.core.components.validators.source_connection import SourceConnectionValidator
from runtime.errors import SourceConnectionError


class TestSourceConnectionValidator:
    @staticmethod
    def _create_mock_reader():
        """Create a mock reader that supports context manager protocol."""
        reader = MagicMock()
        reader.connect.return_value = None
        reader.close.return_value = None
        return reader

    def test_validate_calls_connect_and_cleanup(self):
        validator = SourceConnectionValidator()
        reader = self._create_mock_reader()
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")

        with patch(
            "runtime.core.components.validators.source_connection.StreamReaderFactory.create",
            return_value=reader,
        ):
            validator.validate(config=config)

        reader.connect.assert_called_once()
        reader.__exit__.assert_called_once()

    def test_validate_raises_source_connection_error_on_runtime_error(self):
        validator = SourceConnectionValidator()
        reader = self._create_mock_reader()
        reader.connect.side_effect = RuntimeError("Could not open video source: 0")
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")

        with patch(
            "runtime.core.components.validators.source_connection.StreamReaderFactory.create",
            return_value=reader,
        ):
            with pytest.raises(SourceConnectionError, match="Could not open video source"):
                validator.validate(config=config)

        reader.__exit__.assert_called_once()

    def test_validate_raises_source_connection_error_on_connection_error(self):
        validator = SourceConnectionValidator()
        reader = self._create_mock_reader()
        reader.connect.side_effect = ConnectionError("Failed to connect")
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")

        with patch(
            "runtime.core.components.validators.source_connection.StreamReaderFactory.create",
            return_value=reader,
        ):
            with pytest.raises(SourceConnectionError, match="Failed to connect"):
                validator.validate(config=config)

        reader.__exit__.assert_called_once()

    def test_validate_raises_source_connection_error_on_os_error(self):
        validator = SourceConnectionValidator()
        reader = self._create_mock_reader()
        reader.connect.side_effect = OSError("Device not found")
        config = UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Test Camera")

        with patch(
            "runtime.core.components.validators.source_connection.StreamReaderFactory.create",
            return_value=reader,
        ):
            with pytest.raises(SourceConnectionError, match="Device not found"):
                validator.validate(config=config)

        reader.__exit__.assert_called_once()
