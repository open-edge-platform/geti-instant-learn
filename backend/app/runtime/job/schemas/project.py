#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from runtime.core.components.schemas.processor import ProcessorConfig
from runtime.core.components.schemas.reader import ReaderConfig
from runtime.core.components.schemas.writer import WriterConfig


class ProjectConfig(BaseModel):
    project_id: str
    reader: ReaderConfig
    writer: WriterConfig
    processor: ProcessorConfig
