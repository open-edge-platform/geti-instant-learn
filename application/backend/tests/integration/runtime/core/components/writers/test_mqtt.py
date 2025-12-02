import json
from queue import Queue
from threading import Event
from types import SimpleNamespace

import paho.mqtt.client as mqtt
import pytest
from testcontainers.mqtt import MosquittoContainer

from runtime.core.components.writers import mqtt_writer as mqtt_module
from runtime.core.components.writers.mqtt_writer import MqttWriter

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mqtt_broker():
    with MosquittoContainer(image="eclipse-mosquitto:2.0.20") as container:
        yield container.get_container_host_ip(), int(container.get_exposed_port(1883))


def _frame(payload):
    return SimpleNamespace(results=payload)


def _subscribe(host: str, port: int, topic: str):
    queue: Queue[str] = Queue()
    ready = Event()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(m_client, _userdata, _flags, _reason_code, *_):
        m_client.subscribe(topic)
        ready.set()

    def on_message(_client, _userdata, message):
        queue.put(message.payload.decode("utf-8"))

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port)
    client.loop_start()
    if not ready.wait(timeout=5):
        client.loop_stop()
        client.disconnect()
        raise TimeoutError("Subscriber failed to connect to MQTT broker")

    def cleanup():
        client.loop_stop()
        client.disconnect()

    return queue, cleanup


class TestMqtt:
    def test_publish_round_trip(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/round-trip"
        writer = MqttWriter(host=host, port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)

        try:
            message = _frame({"foo": "bar"})
            writer.write(message)
            assert queue.get(timeout=5) == json.dumps(message.results)
        finally:
            writer.close()
            teardown()

    def test_reconnect_after_close(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/reconnect"
        writer = MqttWriter(host=host, port=port, topic=topic)

        queue, teardown = _subscribe(host, port, topic)
        queue_next = None
        teardown_next = None

        try:
            writer.write(_frame("first"))
            assert queue.get(timeout=5) == json.dumps("first")
            teardown()
            writer.close()

            queue_next, teardown_next = _subscribe(host, port, topic)
            writer.write(_frame("second"))
            assert queue_next.get(timeout=5) == json.dumps("second")
        finally:
            writer.close()
            if teardown_next:
                teardown_next()

    def test_connect_without_credentials(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "imqtt/no-auth"
        writer = MqttWriter(host=host, port=port, topic=topic, auth_required=False)
        queue, teardown = _subscribe(host, port, topic)

        try:
            writer.write(_frame("anonymous-message"))
            assert queue.get(timeout=5) == json.dumps("anonymous-message")
            assert writer._connected is True
        finally:
            writer.close()
            teardown()

    def test_connect_with_credentials(self, mqtt_broker, monkeypatch):
        host, port = mqtt_broker
        topic = "mqtt/auth"
        # https://github.com/testcontainers/testcontainers-python/blob/main/modules/mqtt/testcontainers/mqtt/__init__.py#L124
        monkeypatch.setattr(mqtt_module, "MQTT_USERNAME", "integration-user", raising=False)
        monkeypatch.setattr(mqtt_module, "MQTT_PASSWORD", "integration-pass", raising=False)

        writer = MqttWriter(host=host, port=port, topic=topic, auth_required=True)
        queue, teardown = _subscribe(host, port, topic)

        try:
            writer.write(_frame("authenticated-message"))
            assert queue.get(timeout=5) == json.dumps("authenticated-message")
            assert writer._connected is True
            assert writer._client._username.decode("utf-8") == "integration-user"
            assert writer._client._password.decode("utf-8") == "integration-pass"
        finally:
            writer.close()
            teardown()
