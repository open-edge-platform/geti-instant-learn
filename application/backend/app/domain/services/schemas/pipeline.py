# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field

from domain.db.models import PromptType
from domain.services.schemas.device import DEVICE_STR_PATTERN
from domain.services.schemas.processor import ModelConfig
from domain.services.schemas.reader import ReaderConfig
from domain.services.schemas.writer import WriterConfig


class PipelineConfig(BaseModel):
    project_id: UUID
    device: str = Field(default="auto", pattern=DEVICE_STR_PATTERN)
    prompt_mode: PromptType = PromptType.VISUAL
    reader: ReaderConfig | None = None
    writer: WriterConfig | None = None
    processor: ModelConfig | None = None
