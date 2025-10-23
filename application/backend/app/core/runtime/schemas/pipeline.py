# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel

from core.components.schemas.processor import ModelConfig
from core.components.schemas.reader import ReaderConfig
from core.components.schemas.writer import WriterConfig


class PipelineConfig(BaseModel):
    project_id: UUID
    reader: ReaderConfig | None = None
    writer: WriterConfig | None = None
    processor: ModelConfig | None = None
