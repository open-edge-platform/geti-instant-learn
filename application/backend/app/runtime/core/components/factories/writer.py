#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os

from domain.services.schemas.writer import DatasetConfig, MqttConfig, WriterConfig
from runtime.core.components.base import StreamWriter
from runtime.core.components.writers.mqtt_writer import MqttWriter
from runtime.core.components.writers.noop_writer import NoOpWriter


class StreamWriterFactory:
    """
    A factory for creating StreamWriter instances based on a configuration.

    This class decouples the application from the concrete implementation of
    the StreamWriter, allowing for different writer types to be instantiated
    based on the provided configuration.
    """

    @classmethod
    def create(cls, config: WriterConfig | None) -> StreamWriter:
        match config:
            case MqttConfig() as config:
                return MqttWriter(
                    config=config,
                    username=os.getenv("MQTT_USERNAME", "username"),
                    password=os.getenv("MQTT_PASSWORD", "password"),
                )
            case DatasetConfig() as config:
                try:
                    from runtime.core.components.writers.dataset_writer import DatasetWriter
                except ImportError as e:
                    raise RuntimeError(
                        "Requires datumaro. Install with: uv sync --extra dataset"
                    ) from e
                return DatasetWriter(config=config)
            case _:
                return NoOpWriter()
