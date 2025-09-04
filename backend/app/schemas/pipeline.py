#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel

from backend.app.schemas.processor import ProcessorConfig
from backend.app.schemas.sink import SinkConfig
from backend.app.schemas.source import SourceConfig


class PipelineConfig(BaseModel):
    pipeline_id: str
    source_config: Optional[SourceConfig]
    processor_config: Optional[ProcessorConfig]
    sink_config: Optional[SinkConfig]
