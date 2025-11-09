# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel

from runtime.core.components.schemas.processor import ModelConfig
from runtime.core.components.schemas.reader import ReaderConfig
from runtime.core.components.schemas.writer import WriterConfig


class PipelineConfig(BaseModel):
    project_id: UUID
    reader: ReaderConfig | None = None
    writer: WriterConfig | None = None
    processor: ModelConfig | None = None
