import json
import socket
import time
from queue import Queue
from threading import Event

import numpy as np
import paho.mqtt.client as mqtt
import pytest
from instantlearn.data.base.prediction import Prediction
from testcontainers.mqtt import MosquittoContainer

from domain.services.schemas.processor import OutputData
from domain.services.schemas.writer import WriterConfig
from runtime.core.components.writers.mqtt_writer import MqttWriter

pytestmark = pytest.mark.integration
# Expected top-level keys in every serialised prediction dict.
_EXPECTED_PAYLOAD_KEYS = {"masks", "scores", "label_ids", "label_names", "boxes"}


@pytest.fixture()
def mqtt_broker():
    with MosquittoContainer(image="eclipse-mosquitto:2.0.22") as container:
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


def mqtt_test_data():
    prediction = Prediction(
        masks=np.ones((1, 2, 2), dtype=np.uint8),
        scores=np.array([0.9], dtype=np.float32),
        label_ids=np.array([0], dtype=np.int32),
        label_names=np.array(["object"], dtype=object),
        boxes=np.array([[0, 0, 2, 2]], dtype=np.float32),
    )
    return OutputData(frame=np.full((1), 1), results=[prediction])


class TestMqtt:
    def test_publish_round_trip(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/round-trip"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)
        try:
            with MqttWriter(config=config) as writer:
                message = mqtt_test_data()
                writer.connect()
                writer.write(message)
                assert json.loads(queue.get(timeout=5)) == message.to_list()
        finally:
            teardown()

    def test_connect_without_credentials(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/no-auth"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)
        try:
            with MqttWriter(config=config) as writer:
                message = mqtt_test_data()
                writer.connect()
                writer.write(message)
                assert json.loads(queue.get(timeout=5)) == message.to_list()
                assert writer._connected is True
        finally:
            teardown()

    def test_connect_with_credentials(self, mqtt_broker):
        host, port = mqtt_broker
        topic = "mqtt/auth"
        # https://github.com/testcontainers/testcontainers-python/blob/main/modules/mqtt/testcontainers/mqtt/__init__.py#L124
        username = "integration-user"
        password = "integration-pass"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic, auth_required=True)
        queue, teardown = _subscribe(host, port, topic)
        try:
            with MqttWriter(config=config, username=username, password=password) as writer:
                message = mqtt_test_data()
                writer.connect()
                writer.write(message)
                assert json.loads(queue.get(timeout=5)) == message.to_list()
                assert writer._connected is True
                assert writer._client._username.decode("utf-8") == "integration-user"
                assert writer._client._password.decode("utf-8") == "integration-pass"
        finally:
            teardown()

    def test_connect_invalid_host_port_reports_error(self):
        host = "127.0.0.1"
        port = 1  # closed port for fast connection refusal
        topic = "mqtt/invalid-connection"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        with MqttWriter(config=config) as writer:
            with pytest.raises(ConnectionError):
                writer.connect()
            assert writer._connected is False

    def test_payload_keys_match_prediction_schema(self, mqtt_broker):
        """Each serialised prediction must expose exactly the documented field set.
        Pins the contract: {"masks", "scores", "label_ids", "label_names", "boxes"}.
        No legacy pred_* keys, no extra fields.
        """
        host, port = mqtt_broker
        topic = "mqtt/payload-keys"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)
        try:
            with MqttWriter(config=config) as writer:
                message = mqtt_test_data()
                writer.connect()
                writer.write(message)
                payload = json.loads(queue.get(timeout=5))
            assert isinstance(payload, list), "Top-level MQTT payload must be a list of predictions"
            assert len(payload) == 1, "Expected exactly one prediction in the payload"
            item = payload[0]
            assert set(item.keys()) == _EXPECTED_PAYLOAD_KEYS, (
                f"Prediction keys {set(item.keys())} != {_EXPECTED_PAYLOAD_KEYS}"
            )
        finally:
            teardown()

    def test_payload_values_match_prediction_data(self, mqtt_broker):
        """Verify the serialised values round-trip correctly for all fields."""
        host, port = mqtt_broker
        topic = "mqtt/payload-values"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)
        masks = np.ones((1, 2, 2), dtype=np.uint8)
        scores = np.array([0.9], dtype=np.float32)
        label_ids = np.array([0], dtype=np.int32)
        label_names = np.array(["object"], dtype=object)
        boxes = np.array([[0, 0, 2, 2]], dtype=np.float32)
        prediction = Prediction(
            masks=masks,
            scores=scores,
            label_ids=label_ids,
            label_names=label_names,
            boxes=boxes,
        )
        message = OutputData(frame=np.zeros((4, 4, 3), dtype=np.uint8), results=[prediction])
        try:
            with MqttWriter(config=config) as writer:
                writer.connect()
                writer.write(message)
                payload = json.loads(queue.get(timeout=5))
            item = payload[0]
            assert item["masks"] == masks.tolist()
            assert item["scores"] == pytest.approx(scores.tolist(), rel=1e-5)
            assert item["label_ids"] == label_ids.tolist()
            assert item["label_names"] == label_names.tolist()
            assert item["boxes"] == [pytest.approx(row, rel=1e-5) for row in boxes.tolist()]
        finally:
            teardown()

    def test_payload_boxes_empty_when_prediction_has_no_boxes(self, mqtt_broker):
        """When a prediction has no boxes, the serialised 'boxes' field is an empty list."""
        host, port = mqtt_broker
        topic = "mqtt/payload-no-boxes"
        config = mqtt_config(broker_host=host, broker_port=port, topic=topic)
        queue, teardown = _subscribe(host, port, topic)
        prediction = Prediction(
            masks=np.ones((1, 2, 2), dtype=np.uint8),
            scores=np.array([0.9], dtype=np.float32),
            label_ids=np.array([0], dtype=np.int32),
            label_names=np.array(["object"], dtype=object),
            boxes=None,  # No boxes
        )
        message = OutputData(frame=np.zeros((4, 4, 3), dtype=np.uint8), results=[prediction])
        try:
            with MqttWriter(config=config) as writer:
                writer.connect()
                writer.write(message)
                payload = json.loads(queue.get(timeout=5))
            item = payload[0]
            assert item["boxes"] == [], "boxes must serialize to [] when the prediction has no boxes"
        finally:
            teardown()
