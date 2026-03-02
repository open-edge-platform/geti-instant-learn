# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
from instantlearn.components.encoders.timm import AVAILABLE_IMAGE_ENCODERS
from instantlearn.utils.constants import SAMModelName
from pydantic import BaseModel, Field, field_validator

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


class ModelType(StrEnum):
    MATCHER = "matcher"
    PERDINO = "perdino"
    SOFT_MATCHER = "soft_matcher"


ALLOWED_SAM_MODELS: tuple[SAMModelName, ...] = (
    SAMModelName.SAM_HQ,
    SAMModelName.SAM_HQ_TINY,
)


class BaseModelConfig(BaseModel):
    """Base configuration class with common validators for all model types."""

    sam_model: SAMModelName = Field(default=SAMModelName.SAM_HQ_TINY)
    encoder_model: str = Field(default="dinov3_small")
    precision: str = Field(default="bf16", description="Model precision")
    use_nms: bool = Field(default=True)

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


class PerDinoConfig(BaseModelConfig):
    model_type: Literal[ModelType.PERDINO] = ModelType.PERDINO
    num_foreground_points: int = Field(default=80, gt=0, lt=300)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    num_grid_cells: int = Field(default=16, gt=0)
    point_selection_threshold: float = Field(default=0.65, gt=0.0, lt=1.0)
    confidence_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "perdino",
                "encoder_model": "dinov3_small",
                "sam_model": "SAM-HQ-tiny",
                "num_foreground_points": 80,
                "num_background_points": 2,
                "num_grid_cells": 16,
                "point_selection_threshold": 0.65,
                "confidence_threshold": 0.42,
                "precision": "bf16",
                "use_nms": True,
            }
        }
    }


class MatcherConfig(BaseModelConfig):
    model_type: Literal[ModelType.MATCHER] = ModelType.MATCHER
    num_foreground_points: int = Field(default=5, gt=0, lt=300)
    num_background_points: int = Field(default=3, ge=0, lt=10)
    confidence_threshold: float = Field(default=0.38, gt=0.0, lt=1.0)
    use_mask_refinement: bool = Field(default=False)

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
                "use_nms": True,
            }
        }
    }


class SoftMatcherConfig(BaseModelConfig):
    model_type: Literal[ModelType.SOFT_MATCHER] = ModelType.SOFT_MATCHER
    num_foreground_points: int = Field(default=40, gt=0, lt=300)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    confidence_threshold: float = Field(default=0.42, gt=0.0, lt=1.0)
    use_sampling: bool = Field(default=False)
    use_spatial_sampling: bool = Field(default=False)
    approximate_matching: bool = Field(default=False)
    softmatching_score_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)
    softmatching_bidirectional: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "soft_matcher",
                "sam_model": "SAM-HQ-tiny",
                "encoder_model": "dinov3_small",
                "num_foreground_points": 40,
                "num_background_points": 2,
                "confidence_threshold": 0.42,
                "use_sampling": False,
                "use_spatial_sampling": False,
                "approximate_matching": False,
                "softmatching_score_threshold": 0.4,
                "softmatching_bidirectional": False,
                "precision": "bf16",
                "use_nms": True,
            }
        }
    }


ModelConfig = Annotated[PerDinoConfig | MatcherConfig | SoftMatcherConfig, Field(discriminator="model_type")]


@dataclass(kw_only=True)
class InputData:
    timestamp: int  # processing date-time in epoch milliseconds.
    frame: np.ndarray  # frame loaded as numpy array in RGB HWC format (H, W, 3) with dtype=uint8
    context: dict[str, Any]  # unstructured metadata about the source of the frame (camera ID, video file, etc.)


@dataclass(kw_only=True)
class OutputData:
    results: list[dict[str, np.ndarray]]
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
