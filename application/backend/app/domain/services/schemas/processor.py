# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
import torch
from getiprompt.components.encoders.timm import AVAILABLE_IMAGE_ENCODERS
from getiprompt.utils.constants import SAMModelName
from pydantic import BaseModel, Field, field_validator, model_serializer

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class SupportedModelsSchema(BaseModel):
    sam_models: list[str]
    encoder_models: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "sam_models": ["SAM-HQ", "SAM-HQ-tiny"],
                "encoder_models": [
                    "dinov3_small",
                    "dinov3_small_plus",
                    "dinov3_base",
                    "dinov3_large",
                    "dinov3_huge",
                ],
            }
        }
    }


class ModelType(StrEnum):
    MATCHER = "matcher"


class MatcherConfig(BaseModel):
    model_type: Literal[ModelType.MATCHER] = ModelType.MATCHER
    num_foreground_points: int = Field(default=40, gt=0, lt=100)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    confidence_threshold: float = Field(default=0.38, gt=0.0, lt=1.0)
    precision: str = Field(default="bf16", description="Model precision")
    sam_model: SAMModelName = Field(default=SAMModelName.SAM_HQ_TINY)
    encoder_model: str = Field(default="dinov3_large")

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        if v not in AVAILABLE_IMAGE_ENCODERS:
            raise ValueError(f"Supported encoder must be one of {list(AVAILABLE_IMAGE_ENCODERS.keys())}, got '{v}'")
        return v

    @field_validator("sam_model", mode="before")
    @classmethod
    def validate_sam_model(cls, v: str | SAMModelName) -> SAMModelName:
        """Validate SAM model is supported."""
        # Convert string to enum if needed
        if isinstance(v, str):
            try:
                v = SAMModelName(v)
            except ValueError:
                allowed = {SAMModelName.SAM_HQ.value, SAMModelName.SAM_HQ_TINY.value}
                raise ValueError(f"Supported sam model must be one of {list(allowed)}, got '{v}'")

        # Validate enum value is in supported set
        if v not in {SAMModelName.SAM_HQ, SAMModelName.SAM_HQ_TINY}:
            allowed = {SAMModelName.SAM_HQ.value, SAMModelName.SAM_HQ_TINY.value}
            raise ValueError(f"Supported sam model must be one of {list(allowed)}, got '{v.value}'")

        return v

    @model_serializer(mode="wrap")
    def serialize_sam_model(self, serializer: Any) -> dict[str, Any]:
        """Serialize enum to string for JSON compatibility."""
        data = serializer(self)
        data["sam_model"] = self.sam_model.value
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "matcher",
                "num_foreground_points": 40,
                "num_background_points": 2,
                "confidence_threshold": 0.38,
                "precision": "bf16",
                "sam_model": "SAM-HQ-tiny",
                "encoder_model": "dinov3_large",
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
    labels_colors: dict[str, tuple[int, int, int]] | None = None  # mapping from label IDs to RGB color tuples


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
