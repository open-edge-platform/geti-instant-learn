#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from backend.app.runtime.schemas.pipeline import PipelineConfig
from backend.app.runtime.schemas.sink import SinkConfig
from backend.app.runtime.schemas.source import SourceConfig


class ProjectConfig(BaseModel):
    project_id: str
    source_config: SourceConfig
    pipeline_config: PipelineConfig
    sink_config: SinkConfig
