#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import json
import logging
import time

import paho.mqtt.client as mqtt

from domain.services.schemas.processor import OutputData
from domain.services.schemas.writer import WriterConfig
from runtime.core.components.base import StreamWriter

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECT_TIMEOUT = 10


class MqttWriter(StreamWriter):
    def __init__(
        self,
        config: WriterConfig,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._config = config
        self.username = username
        self.password = password

        self._client: mqtt.Client | None = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self._config.auth_required:
            self._client.username_pw_set(self.username, self.password)
        else:
            logger.info("MQTT authentication is disabled")
        self._connected: bool = False

    def _connect(self) -> None:
        if self._client is None or self._connected:
            return
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"Connecting to MQTT broker at {self._config.broker_host}:{self._config.broker_port} "
                    f"(attempt {attempt + 1})"
                )
                self._client.connect(self._config.broker_host, self._config.broker_port)
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

        logger.info(f"Publishing data to MQTT topic: {self._config.topic}")
        payload = json.dumps(data.results)
        self._client.publish(self._config.topic, payload)

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
        self._client = None
        self._connected = False
