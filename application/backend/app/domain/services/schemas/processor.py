# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
import torch
from getiprompt.components.encoders.timm import AVAILABLE_IMAGE_ENCODERS
from getiprompt.utils.constants import SAMModelName
from pydantic import BaseModel, Field, field_validator

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class ModelType(StrEnum):
    MATCHER = "matcher"


ALLOWED_SAM_MODELS: tuple[SAMModelName, ...] = (
    SAMModelName.SAM_HQ,
    SAMModelName.SAM_HQ_TINY,
)


class MatcherConfig(BaseModel):
    model_type: Literal[ModelType.MATCHER] = ModelType.MATCHER
    num_foreground_points: int = Field(default=5, gt=0, lt=100)
    num_background_points: int = Field(default=3, ge=0, lt=10)
    confidence_threshold: float = Field(default=0.38, gt=0.0, lt=1.0)
    precision: str = Field(default="bf16", description="Model precision")
    sam_model: SAMModelName = Field(default=SAMModelName.SAM_HQ_TINY)
    encoder_model: str = Field(default="dinov3_small")
    use_mask_refinement: bool = Field(default=False)

    @field_validator("sam_model", mode="before")
    @classmethod
    def validate_sam_model(cls, value: Any) -> SAMModelName:
        candidate = value if isinstance(value, SAMModelName) else SAMModelName(value)
        if candidate not in ALLOWED_SAM_MODELS:
            allowed = ", ".join(model.value for model in ALLOWED_SAM_MODELS)
            raise ValueError(f"Supported SAM model must be one of [{allowed}], got '{candidate.value}'")
        return candidate

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        if v not in AVAILABLE_IMAGE_ENCODERS:
            raise ValueError(f"Supported encoder must be one of {list(AVAILABLE_IMAGE_ENCODERS.keys())}, got '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "matcher",
                "num_foreground_points": 5,
                "num_background_points": 3,
                "confidence_threshold": 0.38,
                "precision": "bf16",
                "sam_model": "SAM-HQ-tiny",
                "encoder_model": "dinov3_small",
                "use_mask_refinement": False,
            }
        }
    }


ModelConfig = Annotated[MatcherConfig, Field(discriminator="model_type")]


@dataclass(kw_only=True)
class InputData:
    timestamp: int  # processing date-time in epoch milliseconds.
    frame: np.ndarray  # frame loaded as numpy array in RGB HWC format (H, W, 3) with dtype=uint8
    context: dict[str, Any]  # unstructured metadata about the source of the frame (camera ID, video file, etc.)


@dataclass(kw_only=True)
class OutputData:
    results: list[dict[str, torch.Tensor]]
    frame: np.ndarray  # frame loaded as numpy array in RGB HWC format (H, W, 3) with dtype=uint8


class ProcessorSchema(BaseIDSchema):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)


class ProcessorListSchema(PaginatedResponse):
    models: list[ProcessorSchema]


class ProcessorCreateSchema(BaseIDPayload):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)


class ProcessorUpdateSchema(BaseModel):
    config: ModelConfig
    active: bool
    name: str = Field(max_length=80, min_length=1)
