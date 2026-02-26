from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.writers import mqtt_writer
from runtime.core.components.writers.mqtt_writer import MqttWriter


@pytest.fixture
def mocked_writer(monkeypatch):
    client_instance = MagicMock()
    client_factory = MagicMock(return_value=client_instance)
    monkeypatch.setattr(mqtt_writer.mqtt, "Client", client_factory)
    config = WriterConfig(broker_host="mqtt.example", topic="topic/1", broker_port=1884)
    writer = MqttWriter(config=config)
    return writer, client_instance


class TestMqttWriter:
    def test_write_publishes_when_connected(self, mocked_writer):
        writer, client = mocked_writer
        writer._client = client
        writer._connected = True
        writer.connect = MagicMock()

        writer.write(SimpleNamespace(results={"foo": "bar"}))

        writer.connect.assert_not_called()
        client.publish.assert_called_once_with("topic/1", '{"foo": "bar"}')
        assert writer._connected is True

    def test_connect_is_noop_when_already_connected(self, mocked_writer):
        writer, client = mocked_writer
        writer._connected = True

        writer.connect()

        client.connect.assert_not_called()

    def test_connect_retries_and_raises_after_failures(self, mocked_writer):
        writer, client = mocked_writer
        writer._client = client
        client.connect.side_effect = Exception("boom")

        with pytest.raises(ConnectionError, match="Failed to connect"):
            writer.connect()

        assert client.connect.call_count == mqtt_writer.MAX_RETRIES

    def test_write_raises_when_client_missing(self, mocked_writer):
        writer, _ = mocked_writer
        writer._client = None

        with pytest.raises(RuntimeError, match="client is not initialised"):
            writer.write("payload")

    def test_write_raises_when_not_connected(self, mocked_writer):
        writer, client = mocked_writer
        writer._client = client
        writer._connected = False

        with pytest.raises(RuntimeError, match="client is not connected"):
            writer.write("payload")
