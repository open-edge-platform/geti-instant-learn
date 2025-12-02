#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class WriterType(StrEnum):
    MQTT = "mqtt"


class MqttConfig(BaseModel):
    sink_type: Literal[WriterType.MQTT] = WriterType.MQTT
    name: str = "MQTT Broker"
    broker_host: str = "localhost"
    broker_port: int = 1883
    topic: str = "predictions"
    auth_required: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "sink_type": "mqtt",
                "name": "MQTT Broker",
                "broker_host": "localhost",
                "broker_port": 1883,
                "topic": "predictions",
                "auth_required": True,
            }
        }
    }


WriterConfig = Annotated[MqttConfig, Field(discriminator="sink_type")]
