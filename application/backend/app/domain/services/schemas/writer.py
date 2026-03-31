#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class WriterType(StrEnum):
    MQTT = "mqtt"
    DATASET = "dataset"


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

class DatasetConfig(BaseModel):
    sink_type: Literal[WriterType.DATASET] = WriterType.DATASET
    name: str = "Dataset Writer"
    output_dir: str = "/path/to/output/dataset"
    dataset_format: str | None = None
    max_frames: int | None = Field(default=None, ge=1)
    export_chunk_size: int | None = Field(default=None, ge=1)
    category_mapping: dict[int, str] | None = None
    frame_trace: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "sink_type": "dataset",
                "name": "Dataset Writer",
                "output_dir": "/path/to/output/dataset",
                "dataset_format": None,
                "max_frames": None,
                "export_chunk_size": None,
                "category_mapping": None,
                "frame_trace": False
            }
        }
    }

WriterConfig = Annotated[MqttConfig | DatasetConfig, Field(discriminator="sink_type")]
