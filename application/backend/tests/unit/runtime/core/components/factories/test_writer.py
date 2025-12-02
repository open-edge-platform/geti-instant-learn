#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.writer import MqttConfig, WriterType
from runtime.core.components.factories.writer import StreamWriterFactory
from runtime.core.components.writers.mqtt_writer import MqttWriter
from runtime.core.components.writers.noop_writer import NoOpWriter


class TestWriter:
    def test_factory_returns_mqtt_writer(self):
        mqtt_config = MqttConfig(sink_type=WriterType.MQTT)

        result = StreamWriterFactory.create(mqtt_config)

        assert isinstance(result, MqttWriter)
        assert result._config.broker_host == mqtt_config.broker_host
        assert result._config.broker_port == mqtt_config.broker_port
        assert result._config.topic == mqtt_config.topic
        assert result._config.auth_required == mqtt_config.auth_required

    def test_factory_returns_noop_writer_for_other_config(self):
        result = StreamWriterFactory.create(None)

        assert isinstance(result, NoOpWriter)
