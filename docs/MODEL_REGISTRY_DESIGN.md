# Model Registry and API Redesign

## Table of Contents

- [Model Registry and API Redesign](#model-registry-and-api-redesign)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Problem Statement](#problem-statement)
    - [Current State](#current-state)
    - [Limitations](#limitations)
  - [Proposed Solution](#proposed-solution)
    - [Two Separate Concerns](#two-separate-concerns)
  - [Library Changes](#library-changes)
    - [1. Create Model Registry](#1-create-model-registry)
    - [2. Update Existing Constants (Optional)](#2-update-existing-constants-optional)
  - [Application Backend Changes](#application-backend-changes)
    - [1. Update Schema Definitions](#1-update-schema-definitions)
    - [2. Add Pipeline Config Classes](#2-add-pipeline-config-classes)
    - [3. Update Service Layer](#3-update-service-layer)
    - [4. Update API Endpoint](#4-update-api-endpoint)
  - [Application Frontend Changes](#application-frontend-changes)
    - [1. Update API Types](#1-update-api-types)
    - [2. Update API Client](#2-update-api-client)
    - [3. Update UI Components](#3-update-ui-components)
  - [Data Flow](#data-flow)
  - [Migration Strategy](#migration-strategy)
    - [Phase 1: Add Registry (Non-breaking)](#phase-1-add-registry-non-breaking)
    - [Phase 2: Add New Endpoint (Non-breaking)](#phase-2-add-new-endpoint-non-breaking)
    - [Phase 3: Migrate Frontend](#phase-3-migrate-frontend)
    - [Phase 4: Deprecate Old Endpoint](#phase-4-deprecate-old-endpoint)
  - [Adding New Models](#adding-new-models)
    - [Adding a New Model to Registry](#adding-a-new-model-to-registry)
    - [Adding a New Pipeline Type](#adding-a-new-pipeline-type)
  - [API Examples](#api-examples)
    - [GET /projects/{id}/models/supported](#get-projectsidmodelssupported)
    - [GET /projects/{id}/models/supported?type=segmenter](#get-projectsidmodelssupportedtypesegmenter)
    - [POST /projects/{id}/models (Matcher Pipeline)](#post-projectsidmodels-matcher-pipeline)
    - [POST /projects/{id}/models (SoftMatcher Pipeline)](#post-projectsidmodels-softmatcher-pipeline)
  - [Benefits](#benefits)
  - [Future Considerations](#future-considerations)

## Overview

This document outlines the design for refactoring the model registry and supported models API to support a more extensible and maintainable architecture. The goal is to move from hardcoded model categories to a structured, metadata-rich model registry.

## Problem Statement

### Current State

Models are defined in scattered locations with implicit metadata:

| Location | Content |
|----------|---------|
| `library/src/getiprompt/utils/constants.py` | `SAMModelName` enum |
| `library/src/getiprompt/components/encoders/timm.py` | `AVAILABLE_IMAGE_ENCODERS` dict |
| `application/backend/app/domain/services/schemas/processor.py` | `ALLOWED_SAM_MODELS` tuple |

The API returns a flat structure:

```json
{
  "sam_models": ["SAM-HQ", "SAM-HQ-tiny"],
  "encoder_models": ["dinov3_small", "dinov3_large", ...]
}
```

### Limitations

1. **Schema changes required** for each new model category (tracker, VLM, detector)
2. **No metadata** - frontend can't filter by size, capability, or modality
3. **Hardcoded categories** - adding SAM3 tracker requires API contract change
4. **Frontend must understand naming conventions** - "SAM-HQ-tiny" implies size but isn't explicit

## Proposed Solution

### Two Separate Concerns

| Concept | Purpose | Schema |
|---------|---------|--------|
| **Model Registry** (`ModelMetadata`) | Catalog of available models with metadata | "What's on the menu?" |
| **Pipeline Config** (`ModelConfig`) | Configuration for a specific pipeline instance | "How do I run this?" |

## Library Changes

### 1. Create Model Registry

Create a new file: `library/src/getiprompt/utils/model_registry.py`

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Centralized model registry - single source of truth for available models."""

from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    """Model type classification."""
    ENCODER = "encoder"
    SEGMENTER = "segmenter"
    TRACKER = "tracker"
    DETECTOR = "detector"
    VLM = "vlm"


class Modality(StrEnum):
    """Input/output modalities."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PromptType(StrEnum):
    """Supported prompt types."""
    POINT = "point"
    BOX = "box"
    MASK = "mask"
    TEXT = "text"
    IMAGE = "image"


class Capability(StrEnum):
    """Model capabilities."""
    ENCODING = "encoding"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    TRACKING = "tracking"
    DESCRIPTION = "description"


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a single model in the registry."""
    id: str
    type: ModelType
    family: str
    size: str
    modalities: tuple[Modality, ...]
    prompts: tuple[PromptType, ...]
    capabilities: tuple[Capability, ...]
    # Internal details (not exposed to API by default)
    weights_url: str | None = None
    hf_model_id: str | None = None
    config_filename: str | None = None


# =============================================================================
# MODEL REGISTRY - SINGLE SOURCE OF TRUTH
# =============================================================================

MODEL_REGISTRY: tuple[ModelMetadata, ...] = (
    # -------------------------------------------------------------------------
    # Segmenters (SAM family)
    # -------------------------------------------------------------------------
    ModelMetadata(
        id="sam-hq",
        type=ModelType.SEGMENTER,
        family="SAM-HQ",
        size="base",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
    ),
    ModelMetadata(
        id="sam-hq-tiny",
        type=ModelType.SEGMENTER,
        family="SAM-HQ",
        size="tiny",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
    ),
    # -------------------------------------------------------------------------
    # Encoders (DINOv3 family)
    # -------------------------------------------------------------------------
    ModelMetadata(
        id="dinov3-small",
        type=ModelType.ENCODER,
        family="DINOv3",
        size="small",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="timm/vit_small_patch16_dinov3.lvd1689m",
    ),
    ModelMetadata(
        id="dinov3-small-plus",
        type=ModelType.ENCODER,
        family="DINOv3",
        size="small-plus",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="timm/vit_small_plus_patch16_dinov3.lvd1689m",
    ),
    ModelMetadata(
        id="dinov3-base",
        type=ModelType.ENCODER,
        family="DINOv3",
        size="base",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="timm/vit_base_patch16_dinov3.lvd1689m",
    ),
    ModelMetadata(
        id="dinov3-large",
        type=ModelType.ENCODER,
        family="DINOv3",
        size="large",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="timm/vit_large_patch16_dinov3.lvd1689m",
    ),
    ModelMetadata(
        id="dinov3-huge",
        type=ModelType.ENCODER,
        family="DINOv3",
        size="huge",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="timm/vit_huge_plus_patch16_dinov3.lvd1689m",
    ),
    # -------------------------------------------------------------------------
    # Future: Trackers (SAM3 family)
    # -------------------------------------------------------------------------
    # ModelMetadata(
    #     id="sam3-large",
    #     type=ModelType.TRACKER,
    #     family="SAM3",
    #     size="large",
    #     modalities=(Modality.IMAGE, Modality.VIDEO),
    #     prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
    #     capabilities=(Capability.SEGMENTATION, Capability.TRACKING),
    # ),
    # -------------------------------------------------------------------------
    # Future: VLMs (Qwen family)
    # -------------------------------------------------------------------------
    # ModelMetadata(
    #     id="qwen-vl-7b",
    #     type=ModelType.VLM,
    #     family="Qwen-VL",
    #     size="7b",
    #     modalities=(Modality.IMAGE, Modality.TEXT),
    #     prompts=(PromptType.TEXT, PromptType.IMAGE),
    #     capabilities=(Capability.DETECTION, Capability.DESCRIPTION),
    # ),
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model(model_id: str) -> ModelMetadata | None:
    """Get a model by ID."""
    return next((m for m in MODEL_REGISTRY if m.id == model_id), None)


def get_models_by_type(model_type: ModelType) -> list[ModelMetadata]:
    """Get all models of a specific type."""
    return [m for m in MODEL_REGISTRY if m.type == model_type]


def get_models_by_capability(capability: Capability) -> list[ModelMetadata]:
    """Get all models with a specific capability."""
    return [m for m in MODEL_REGISTRY if capability in m.capabilities]


def get_models_by_family(family: str) -> list[ModelMetadata]:
    """Get all models from a specific family."""
    return [m for m in MODEL_REGISTRY if m.family == family]


def get_available_types() -> list[str]:
    """Get list of available model types."""
    return list({m.type.value for m in MODEL_REGISTRY})


def get_available_families() -> list[str]:
    """Get list of available model families."""
    return list({m.family for m in MODEL_REGISTRY})


def get_available_sizes() -> list[str]:
    """Get list of available model sizes."""
    return list({m.size for m in MODEL_REGISTRY})


def get_available_capabilities() -> list[str]:
    """Get list of available capabilities."""
    return list({cap.value for m in MODEL_REGISTRY for cap in m.capabilities})


def is_valid_model(model_id: str, model_type: ModelType | None = None) -> bool:
    """Check if a model ID is valid, optionally filtered by type."""
    model = get_model(model_id)
    if model is None:
        return False
    if model_type is not None and model.type != model_type:
        return False
    return True
```

Update `library/src/getiprompt/utils/__init__.py`:

```python
from .model_registry import (
    MODEL_REGISTRY,
    ModelMetadata,
    ModelType,
    Modality,
    PromptType,
    Capability,
    get_model,
    get_models_by_type,
    get_models_by_capability,
    is_valid_model,
)
```

### 2. Remove Legacy Enums

Remove `SAMModelName` enum and `AVAILABLE_IMAGE_ENCODERS` dict from `constants.py`. The registry is now the single source of truth.

**Before (constants.py):**
```python
class SAMModelName(Enum):
    SAM_HQ = "SAM-HQ"
    SAM_HQ_TINY = "SAM-HQ-tiny"
    # ...

AVAILABLE_IMAGE_ENCODERS = {
    "dinov3_small": "timm/vit_small_patch16_dinov3.lvd1689m",
    # ...
}
```

**After:** Delete these. Use registry functions instead:
```python
from getiprompt.utils.model_registry import get_model, get_models_by_type, ModelType

# Get all segmenters
segmenters = get_models_by_type(ModelType.SEGMENTER)

# Get a specific model's metadata
sam_model = get_model("sam-hq-tiny")
```

### 3. Update Model Classes to Use String IDs

**Before (Matcher):**
```python
from getiprompt.utils.constants import SAMModelName

class Matcher(Model):
    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        encoder_model: str = "dinov3_large",
        ...
    ):
```

**After (Matcher):**
```python
from getiprompt.utils.model_registry import get_model, get_models_by_type, ModelType

class Matcher(Model):
    DEFAULT_SAM = "sam-hq-tiny"
    DEFAULT_ENCODER = "dinov3-large"

    def __init__(
        self,
        sam: str = DEFAULT_SAM,
        encoder_model: str = DEFAULT_ENCODER,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        mask_similarity_threshold: float | None = 0.38,
        use_mask_refinement: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        
        # Validate models exist in registry
        self._validate_model(sam, ModelType.SEGMENTER, "sam")
        self._validate_model(encoder_model, ModelType.ENCODER, "encoder_model")

        self.sam_predictor = SAMPredictor(
            sam,
            backend=Backend.PYTORCH,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )
        # ... rest of init

    @staticmethod
    def _validate_model(model_id: str, expected_type: ModelType, param_name: str) -> None:
        """Validate a model ID against the registry."""
        model = get_model(model_id)
        if model is None:
            valid = [m.id for m in get_models_by_type(expected_type)]
            raise ValueError(f"{param_name} must be one of {valid}, got '{model_id}'")
        if model.type != expected_type:
            raise ValueError(
                f"{param_name} must be a {expected_type.value}, "
                f"but '{model_id}' is a {model.type.value}"
            )
```

### 4. Update SAMPredictor and ImageEncoder

These components need to look up model metadata from the registry:

```python
# SAMPredictor
class SAMPredictor:
    def __init__(self, model_id: str, ...):
        model = get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model: {model_id}")
        
        self.weights_url = model.weights_url
        self.config_filename = model.config_filename
        # ... load model

# ImageEncoder
class ImageEncoder:
    def __init__(self, model_id: str, ...):
        model = get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model: {model_id}")
        
        self.hf_model_id = model.hf_model_id
        # ... load model
```

## Application Backend Changes

### 1. Update Schema Definitions

Update `application/backend/app/domain/services/schemas/processor.py`:

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

from getiprompt.utils.model_registry import (
    MODEL_REGISTRY,
    ModelType as LibModelType,
    is_valid_model,
)

from domain.services.schemas.base import BaseIDPayload, BaseIDSchema, PaginatedResponse


# =============================================================================
# MODEL METADATA SCHEMA (for GET /models/supported)
# =============================================================================

class ModelMetadataSchema(BaseModel):
    """Schema describing a single model in the registry."""
    id: str
    type: str
    family: str
    size: str
    modalities: list[str]
    prompts: list[str]
    capabilities: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "sam-hq-tiny",
                "type": "segmenter",
                "family": "SAM-HQ",
                "size": "tiny",
                "modalities": ["image"],
                "prompts": ["point", "box", "mask"],
                "capabilities": ["segmentation"],
            }
        }
    }


class SupportedModelsSchema(BaseModel):
    """Response schema for supported models endpoint."""
    models: list[ModelMetadataSchema]
    # Filter options for frontend
    types: list[str]
    families: list[str]
    sizes: list[str]
    capabilities: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "models": [
                    {
                        "id": "sam-hq-tiny",
                        "type": "segmenter",
                        "family": "SAM-HQ",
                        "size": "tiny",
                        "modalities": ["image"],
                        "prompts": ["point", "box", "mask"],
                        "capabilities": ["segmentation"],
                    },
                    {
                        "id": "dinov3-large",
                        "type": "encoder",
                        "family": "DINOv3",
                        "size": "large",
                        "modalities": ["image"],
                        "prompts": [],
                        "capabilities": ["encoding"],
                    },
                ],
                "types": ["encoder", "segmenter"],
                "families": ["SAM-HQ", "DINOv3"],
                "sizes": ["tiny", "large"],
                "capabilities": ["segmentation", "encoding"],
            }
        }
    }


# =============================================================================
# PIPELINE CONFIGURATION SCHEMAS
# =============================================================================

class PipelineType(StrEnum):
    """Pipeline type discriminator."""
    MATCHER = "matcher"
    SOFT_MATCHER = "soft_matcher"
    PER_DINO = "per_dino"
    # Future pipeline types
    # TRACKER = "tracker"
    # VLM_DETECTOR = "vlm_detector"


class BaseModelConfig(BaseModel):
    """Common fields across all pipeline configs."""
    sam_model: str = Field(default="sam-hq-tiny")
    encoder_model: str = Field(default="dinov3-large")
    precision: str = Field(default="bf16", description="Model precision")

    @field_validator("sam_model")
    @classmethod
    def validate_sam_model(cls, v: str) -> str:
        if not is_valid_model(v, LibModelType.SEGMENTER):
            valid = [m.id for m in MODEL_REGISTRY if m.type == LibModelType.SEGMENTER]
            raise ValueError(f"sam_model must be one of {valid}, got '{v}'")
        return v

    @field_validator("encoder_model")
    @classmethod
    def validate_encoder_model(cls, v: str) -> str:
        if not is_valid_model(v, LibModelType.ENCODER):
            valid = [m.id for m in MODEL_REGISTRY if m.type == LibModelType.ENCODER]
            raise ValueError(f"encoder_model must be one of {valid}, got '{v}'")
        return v


class MatcherConfig(BaseModelConfig):
    """Configuration for Matcher pipeline."""
    model_type: Literal[PipelineType.MATCHER] = PipelineType.MATCHER
    num_foreground_points: int = Field(default=40, gt=0, lt=100)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    confidence_threshold: float = Field(default=0.38, gt=0.0, lt=1.0)
    mask_similarity_threshold: float | None = Field(default=0.42)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "matcher",
                "sam_model": "sam-hq-tiny",
                "encoder_model": "dinov3-large",
                "precision": "bf16",
                "num_foreground_points": 40,
                "num_background_points": 2,
                "confidence_threshold": 0.38,
            }
        }
    }


class SoftMatcherConfig(BaseModelConfig):
    """Configuration for SoftMatcher pipeline."""
    model_type: Literal[PipelineType.SOFT_MATCHER] = PipelineType.SOFT_MATCHER
    num_foreground_points: int = Field(default=40, gt=0, lt=100)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    mask_similarity_threshold: float | None = Field(default=0.42)
    # SoftMatcher-specific parameters
    softmatching_score_threshold: float = Field(default=0.4)
    softmatching_bidirectional: bool = Field(default=False)
    use_sampling: bool = Field(default=False)
    use_spatial_sampling: bool = Field(default=False)
    approximate_matching: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "soft_matcher",
                "sam_model": "sam-hq-tiny",
                "encoder_model": "dinov3-large",
                "precision": "bf16",
                "softmatching_score_threshold": 0.4,
                "approximate_matching": False,
            }
        }
    }


class PerDinoConfig(BaseModelConfig):
    """Configuration for PerDino pipeline."""
    model_type: Literal[PipelineType.PER_DINO] = PipelineType.PER_DINO
    num_foreground_points: int = Field(default=40, gt=0, lt=100)
    num_background_points: int = Field(default=2, ge=0, lt=10)
    mask_similarity_threshold: float | None = Field(default=0.42)
    # PerDino-specific parameters
    num_grid_cells: int = Field(default=16)
    similarity_threshold: float = Field(default=0.65)

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "per_dino",
                "sam_model": "sam-hq-tiny",
                "encoder_model": "dinov3-large",
                "precision": "bf16",
                "num_grid_cells": 16,
                "similarity_threshold": 0.65,
            }
        }
    }


# Discriminated union of all pipeline configs
ModelConfig = Annotated[
    MatcherConfig | SoftMatcherConfig | PerDinoConfig,
    Field(discriminator="model_type")
]


# =============================================================================
# PROCESSOR SCHEMAS (unchanged structure)
# =============================================================================

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
```

### 2. Add Pipeline Config Classes

Already included in the schema above. Each pipeline type has its own config class with specific parameters.

### 3. Update Service Layer

Update `application/backend/app/domain/services/model.py`:

```python
from getiprompt.utils.model_registry import (
    MODEL_REGISTRY,
    ModelType,
    get_available_types,
    get_available_families,
    get_available_sizes,
    get_available_capabilities,
)

from domain.services.schemas.processor import (
    ModelMetadataSchema,
    SupportedModelsSchema,
)


class ModelService:
    # ... existing methods ...

    def supported_models(
        self,
        project_id: UUID,
        model_type: str | None = None,
        family: str | None = None,
        capability: str | None = None,
    ) -> SupportedModelsSchema:
        """
        Get supported models with optional filtering.

        Args:
            project_id: Project ID (for access validation).
            model_type: Filter by model type (encoder, segmenter, etc.).
            family: Filter by model family (SAM-HQ, DINOv3, etc.).
            capability: Filter by capability (segmentation, encoding, etc.).

        Returns:
            SupportedModelsSchema with models list and filter options.

        Raises:
            ResourceNotFoundError: If the project does not exist.
        """
        # Validate project exists
        project = self.project_repository.get_by_id(project_id)
        if not project:
            logger.error(f"Project not found id={project_id}")
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id)
            )

        # Filter models
        models = list(MODEL_REGISTRY)
        
        if model_type:
            models = [m for m in models if m.type.value == model_type]
        if family:
            models = [m for m in models if m.family == family]
        if capability:
            models = [m for m in models if capability in [c.value for c in m.capabilities]]

        # Convert to schema
        model_schemas = [
            ModelMetadataSchema(
                id=m.id,
                type=m.type.value,
                family=m.family,
                size=m.size,
                modalities=[mod.value for mod in m.modalities],
                prompts=[p.value for p in m.prompts],
                capabilities=[c.value for c in m.capabilities],
            )
            for m in models
        ]

        return SupportedModelsSchema(
            models=model_schemas,
            types=get_available_types(),
            families=get_available_families(),
            sizes=get_available_sizes(),
            capabilities=get_available_capabilities(),
        )
```

### 4. Update API Endpoint

Update `application/backend/app/api/endpoints/models.py`:

```python
@projects_router.get(
    path="/{project_id}/models/supported",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully retrieved supported models.",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "id": "sam-hq-tiny",
                                "type": "segmenter",
                                "family": "SAM-HQ",
                                "size": "tiny",
                                "modalities": ["image"],
                                "prompts": ["point", "box", "mask"],
                                "capabilities": ["segmentation"],
                            }
                        ],
                        "types": ["encoder", "segmenter"],
                        "families": ["SAM-HQ", "DINOv3"],
                        "sizes": ["tiny", "large"],
                        "capabilities": ["segmentation", "encoding"],
                    }
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project not found",
        },
    },
)
def supported_models(
    project_id: UUID,
    model_service: ModelServiceDep,
    type: Annotated[str | None, Query(description="Filter by model type")] = None,
    family: Annotated[str | None, Query(description="Filter by model family")] = None,
    capability: Annotated[str | None, Query(description="Filter by capability")] = None,
) -> SupportedModelsSchema:
    """
    Retrieve supported models with optional filtering.

    Query parameters allow filtering by type, family, or capability.
    Response includes filter options for building UI dropdowns.
    """
    return model_service.supported_models(
        project_id=project_id,
        model_type=type,
        family=family,
        capability=capability,
    )
```

## Application Frontend Changes

### 1. Update API Types

Create/update `application/ui/src/api/types/models.ts`:

```typescript
// Model metadata from registry
export interface ModelMetadata {
  id: string;
  type: "encoder" | "segmenter" | "tracker" | "detector" | "vlm";
  family: string;
  size: string;
  modalities: ("image" | "video" | "text")[];
  prompts: ("point" | "box" | "mask" | "text" | "image")[];
  capabilities: ("encoding" | "segmentation" | "detection" | "tracking" | "description")[];
}

// Response from GET /models/supported
export interface SupportedModelsResponse {
  models: ModelMetadata[];
  types: string[];
  families: string[];
  sizes: string[];
  capabilities: string[];
}

// Pipeline config types (discriminated union)
export type PipelineType = "matcher" | "soft_matcher" | "per_dino";

export interface BaseModelConfig {
  sam_model: string;
  encoder_model: string;
  precision: string;
}

export interface MatcherConfig extends BaseModelConfig {
  model_type: "matcher";
  num_foreground_points: number;
  num_background_points: number;
  confidence_threshold: number;
  mask_similarity_threshold?: number;
}

export interface SoftMatcherConfig extends BaseModelConfig {
  model_type: "soft_matcher";
  num_foreground_points: number;
  num_background_points: number;
  mask_similarity_threshold?: number;
  softmatching_score_threshold: number;
  softmatching_bidirectional: boolean;
  use_sampling: boolean;
  use_spatial_sampling: boolean;
  approximate_matching: boolean;
}

export interface PerDinoConfig extends BaseModelConfig {
  model_type: "per_dino";
  num_foreground_points: number;
  num_background_points: number;
  mask_similarity_threshold?: number;
  num_grid_cells: number;
  similarity_threshold: number;
}

export type ModelConfig = MatcherConfig | SoftMatcherConfig | PerDinoConfig;
```

### 2. Update API Client

Update `application/ui/src/api/models.ts`:

```typescript
import { SupportedModelsResponse } from "./types/models";

export interface GetSupportedModelsParams {
  type?: string;
  family?: string;
  capability?: string;
}

export async function getSupportedModels(
  projectId: string,
  params?: GetSupportedModelsParams
): Promise<SupportedModelsResponse> {
  const searchParams = new URLSearchParams();
  if (params?.type) searchParams.set("type", params.type);
  if (params?.family) searchParams.set("family", params.family);
  if (params?.capability) searchParams.set("capability", params.capability);

  const query = searchParams.toString();
  const url = `/projects/${projectId}/models/supported${query ? `?${query}` : ""}`;

  const response = await fetch(url);
  return response.json();
}
```

### 3. Update UI Components

Example component for model selection:

```typescript
// ModelSelector.tsx
import { useState, useEffect } from "react";
import { getSupportedModels, SupportedModelsResponse, ModelMetadata } from "../api/models";

interface ModelSelectorProps {
  projectId: string;
  onSelect: (segmenter: string, encoder: string) => void;
}

export function ModelSelector({ projectId, onSelect }: ModelSelectorProps) {
  const [data, setData] = useState<SupportedModelsResponse | null>(null);
  const [selectedSegmenter, setSelectedSegmenter] = useState<string>("");
  const [selectedEncoder, setSelectedEncoder] = useState<string>("");

  useEffect(() => {
    getSupportedModels(projectId).then(setData);
  }, [projectId]);

  if (!data) return <div>Loading...</div>;

  const segmenters = data.models.filter((m) => m.type === "segmenter");
  const encoders = data.models.filter((m) => m.type === "encoder");

  return (
    <div>
      <label>
        SAM Model:
        <select
          value={selectedSegmenter}
          onChange={(e) => {
            setSelectedSegmenter(e.target.value);
            onSelect(e.target.value, selectedEncoder);
          }}
        >
          <option value="">Select...</option>
          {segmenters.map((m) => (
            <option key={m.id} value={m.id}>
              {m.family} ({m.size})
            </option>
          ))}
        </select>
      </label>

      <label>
        Encoder Model:
        <select
          value={selectedEncoder}
          onChange={(e) => {
            setSelectedEncoder(e.target.value);
            onSelect(selectedSegmenter, e.target.value);
          }}
        >
          <option value="">Select...</option>
          {encoders.map((m) => (
            <option key={m.id} value={m.id}>
              {m.family} ({m.size})
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  SOURCE OF TRUTH                                                    │
│  library/src/getiprompt/utils/model_registry.py                     │
│                                                                     │
│  MODEL_REGISTRY = (                                                 │
│    ModelMetadata(id="sam-hq-tiny", type=SEGMENTER, ...),           │
│    ModelMetadata(id="dinov3-large", type=ENCODER, ...),            │
│  )                                                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ imported by
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BACKEND SERVICE                                                    │
│  app/domain/services/model.py                                       │
│                                                                     │
│  from getiprompt.utils.model_registry import MODEL_REGISTRY         │
│                                                                     │
│  def supported_models(self, project_id, type=None, ...):            │
│      models = [m for m in MODEL_REGISTRY if ...]                   │
│      return SupportedModelsSchema(models=..., types=..., ...)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ serialized to JSON
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  API RESPONSE                                                       │
│  GET /projects/{id}/models/supported                                │
│                                                                     │
│  {                                                                  │
│    "models": [{"id": "sam-hq-tiny", "type": "segmenter", ...}],    │
│    "types": ["encoder", "segmenter"],                              │
│    "families": ["SAM-HQ", "DINOv3"],                               │
│    ...                                                              │
│  }                                                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ fetched by
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FRONTEND                                                           │
│  ui/src/components/ModelSelector.tsx                                │
│                                                                     │
│  const { models, types } = await getSupportedModels(projectId);     │
│  const segmenters = models.filter(m => m.type === "segmenter");    │
│  // Render dropdowns dynamically                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Migration Strategy

### Phase 1: Add Registry (Non-breaking)

1. Create `model_registry.py` in library
2. Add `ModelMetadata` dataclass and `MODEL_REGISTRY` tuple
3. Add helper functions (`get_model`, `get_models_by_type`, etc.)

### Phase 2: Update Library Models

1. Update `Matcher`, `SoftMatcher`, `PerDino` to use string IDs
2. Add `_validate_model()` method for registry validation
3. Update `SAMPredictor` and `ImageEncoder` to look up metadata from registry
4. Remove `SAMModelName` enum and `AVAILABLE_IMAGE_ENCODERS` from `constants.py`
5. Update all tests to use string IDs

### Phase 3: Update Backend

1. Add new schema classes (`ModelMetadataSchema`, updated `SupportedModelsSchema`)
2. Update service layer to use registry
3. Update validators to use `is_valid_model()` from registry

### Phase 4: Update Frontend

1. Update TypeScript types
2. Update API client
3. Update components to use new response structure

## Adding New Models

### Adding a New Model to Registry

Just add to `MODEL_REGISTRY`:

```python
# In model_registry.py
ModelMetadata(
    id="sam3-large",
    type=ModelType.TRACKER,
    family="SAM3",
    size="large",
    modalities=(Modality.IMAGE, Modality.VIDEO),
    prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
    capabilities=(Capability.SEGMENTATION, Capability.TRACKING),
),
```

**No schema changes required** - frontend automatically sees the new model.

### Adding a New Pipeline Type

1. Add config class in `processor.py`:

```python
class TrackerConfig(BaseModelConfig):
    model_type: Literal[PipelineType.TRACKER] = PipelineType.TRACKER
    tracker_model: str
    # tracker-specific params
```

2. Update the union:

```python
ModelConfig = Annotated[
    MatcherConfig | SoftMatcherConfig | PerDinoConfig | TrackerConfig,
    Field(discriminator="model_type")
]
```

3. Add model factory handling in service layer.

## API Examples

### GET /projects/{id}/models/supported

```json
{
  "models": [
    {
      "id": "sam-hq-tiny",
      "type": "segmenter",
      "family": "SAM-HQ",
      "size": "tiny",
      "modalities": ["image"],
      "prompts": ["point", "box", "mask"],
      "capabilities": ["segmentation"]
    },
    {
      "id": "dinov3-large",
      "type": "encoder",
      "family": "DINOv3",
      "size": "large",
      "modalities": ["image"],
      "prompts": [],
      "capabilities": ["encoding"]
    }
  ],
  "types": ["encoder", "segmenter"],
  "families": ["SAM-HQ", "DINOv3"],
  "sizes": ["tiny", "base", "large"],
  "capabilities": ["segmentation", "encoding"]
}
```

### GET /projects/{id}/models/supported?type=segmenter

```json
{
  "models": [
    {
      "id": "sam-hq",
      "type": "segmenter",
      "family": "SAM-HQ",
      "size": "base",
      ...
    },
    {
      "id": "sam-hq-tiny",
      "type": "segmenter",
      "family": "SAM-HQ",
      "size": "tiny",
      ...
    }
  ],
  "types": ["encoder", "segmenter"],
  ...
}
```

### POST /projects/{id}/models (Matcher Pipeline)

```json
{
  "name": "My Matcher",
  "active": true,
  "config": {
    "model_type": "matcher",
    "sam_model": "sam-hq-tiny",
    "encoder_model": "dinov3-large",
    "precision": "bf16",
    "num_foreground_points": 40,
    "num_background_points": 2,
    "confidence_threshold": 0.38
  }
}
```

### POST /projects/{id}/models (SoftMatcher Pipeline)

```json
{
  "name": "My SoftMatcher",
  "active": true,
  "config": {
    "model_type": "soft_matcher",
    "sam_model": "sam-hq-tiny",
    "encoder_model": "dinov3-large",
    "precision": "bf16",
    "softmatching_score_threshold": 0.5,
    "approximate_matching": true
  }
}
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Adding model types | Schema change required | Just add to registry |
| Filtering | Not possible | `?type=encoder&family=DINOv3` |
| Frontend dropdowns | Hardcoded categories | Dynamic from response |
| Model metadata | Implicit in names | Explicit fields |
| Validation | Hardcoded lists | Registry-based |
| Type safety | Limited | Full discriminated unions |

## Future Considerations

1. **Model versioning**: Add `version` field to `ModelMetadata`
2. **Resource requirements**: Add `memory_mb`, `compute_requirements`
3. **Precision support**: Add `supported_precisions: list[str]`
4. **Model dependencies**: Track which models work together
5. **Admin UI**: Interface for managing registry (if moved to database)
6. **Model downloads**: Track download status, cache state
