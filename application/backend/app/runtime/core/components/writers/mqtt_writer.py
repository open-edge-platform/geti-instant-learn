#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time

import paho.mqtt.client as mqtt

from domain.services.schemas.processor import OutputData
from runtime.core.components.base import StreamWriter

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECT_TIMEOUT = 10
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "username")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "password")


class MqttWriter(StreamWriter):
    def __init__(self, host: str, topic: str, port: int = 1883, auth_required: bool = False) -> None:
        self.broker_host = host
        self.broker_port = port
        self.topic = topic
        self.auth_required = auth_required

        self._client: mqtt.Client | None = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self.auth_required:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        else:
            logger.info("MQTT authentication is disabled")
        self._connected: bool = False

    def _connect(self) -> None:
        if self._client is None or self._connected:
            return
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port} (attempt {attempt + 1})"
                )
                self._client.connect(self.broker_host, self.broker_port)
                self._connected = True
                self._client.loop_start()
                return
            except Exception:
                logger.exception("Connection failed")
                time.sleep(RETRY_DELAY * (attempt + 1))
        raise ConnectionError("Failed to connect to MQTT broker")

    def write(self, data: OutputData) -> None:
        """Publish `data` to the configured MQTT topic."""
        if self._client is None:
            raise RuntimeError("MQTT client is not initialised")

        if not self._connected:
            self._connect()

        logger.info(f"Publishing data to MQTT topic: {self.topic}")
        payload = json.dumps(data.results)
        self._client.publish(self.topic, payload)

    def close(self) -> None:
        if self._client is None:
            self._connected = False
            return
        err = self._client.loop_stop()
        if err != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Error stopping MQTT loop: {mqtt.error_string(err)}")

        err = self._client.disconnect()
        if err != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Error disconnecting MQTT client: {mqtt.error_string(err)}")
        self._connected = False
