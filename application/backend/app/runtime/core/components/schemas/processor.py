#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field


class ModelType(StrEnum):
    MATCHER = "matcher"


class MatcherConfig(BaseModel):
    model_type: Literal[ModelType.MATCHER] = ModelType.MATCHER
    num_foreground_points: int = 40
    num_background_points: int = 2
    mask_similarity_threshold: float = 0.38
    precision: str = "bf16"

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "matcher",
                "num_foreground_points": 40,
                "num_background_points": 2,
                "mask_similarity_threshold": 0.38,
                "precision": "bf16",
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
    # the rest will be defined later.
