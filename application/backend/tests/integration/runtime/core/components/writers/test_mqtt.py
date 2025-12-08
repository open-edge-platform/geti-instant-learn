import json
import socket
import time
from queue import Queue
from threading import Event
from types import SimpleNamespace

import paho.mqtt.client as mqtt
import pytest
from testcontainers.mqtt import MosquittoContainer

from domain.services.schemas.writer import WriterConfig
from runtime.core.components.writers.mqtt_writer import MqttWriter

pytestmark = pytest.mark.integration


@pytest.fixture()
def mqtt_broker():
    with MosquittoContainer(image="eclipse-mosquitto:2.0.20") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(1883))
        # Wait for the broker to accept connections
        for _ in range(10):
            try:
                with socket.create_connection((host, port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("MQTT broker did not start in time")
        yield host, port


def _frame(payload):
    return SimpleNamespace(results=payload)


def mqtt_config(broker_host: str, broker_port: int, topic: str, auth_required: bool = False) -> WriterConfig:
    return WriterConfig(
        broker_host=broker_host,
        broker_port=broker_port,
        topic=topic,
        auth_required=auth_required,
    )


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
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        writer = MqttWriter(config=config)
        queue, teardown = _subscribe(host, port, topic)

        try:
            message = _frame({"foo": "bar"})
            writer.write(message)
            assert queue.get(timeout=5) == json.dumps(message.results)
        finally:
            writer.close()
            teardown()

    def test_connect_without_credentials(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/no-auth"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        writer = MqttWriter(config=config)
        queue, teardown = _subscribe(host, port, topic)

        try:
            writer.write(_frame("anonymous-message"))
            assert queue.get(timeout=5) == json.dumps("anonymous-message")
            assert writer._connected is True
        finally:
            writer.close()
            teardown()

    def test_connect_with_credentials(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/auth"
        # https://github.com/testcontainers/testcontainers-python/blob/main/modules/mqtt/testcontainers/mqtt/__init__.py#L124
        username = "integration-user"
        password = "integration-pass"

        config = mqtt_config(broker_host=host, broker_port=port, topic=topic, auth_required=True)
        writer = MqttWriter(config=config, username=username, password=password)
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
