# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-memory repository for supported model metadata."""

from domain.services.schemas.annotation import AnnotationType
from domain.services.schemas.processor import (
    MatcherConfig,
    ModelType,
    PerDinoConfig,
    Sam3Config,
    SoftMatcherConfig,
    SupportedModelMetadataSchema,
    SupportedPromptType,
)

PROMPT_TYPE_TO_ANNOTATION_TYPE: dict[SupportedPromptType, AnnotationType] = {
    SupportedPromptType.VISUAL_POLYGON: AnnotationType.POLYGON,
    SupportedPromptType.VISUAL_RECTANGLE: AnnotationType.RECTANGLE,
}

_SUPPORTED_MODELS_METADATA: dict[ModelType, SupportedModelMetadataSchema] = {
    ModelType.MATCHER: SupportedModelMetadataSchema(
        default_config=MatcherConfig(), supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON]
    ),
    ModelType.PERDINO: SupportedModelMetadataSchema(
        default_config=PerDinoConfig(),
        supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON],
    ),
    ModelType.SOFT_MATCHER: SupportedModelMetadataSchema(
        default_config=SoftMatcherConfig(),
        supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON],
    ),
    ModelType.SAM3: SupportedModelMetadataSchema(
        default_config=Sam3Config(),
        supported_prompt_types=[SupportedPromptType.TEXT, SupportedPromptType.VISUAL_RECTANGLE],
    ),
}


class SupportedModelRepository:
    """Read-only repository exposing supported model metadata."""

    def get_all(self) -> list[SupportedModelMetadataSchema]:
        """Return metadata for all supported models."""
        return list(_SUPPORTED_MODELS_METADATA.values())

    def get_by_model_type(self, model_type: ModelType) -> SupportedModelMetadataSchema | None:
        """Return metadata for a single model type, or None if unknown."""
        return _SUPPORTED_MODELS_METADATA.get(model_type)

    def get_supported_annotation_types(self, model_type: ModelType) -> set[AnnotationType]:
        """Derive supported annotation types for a model."""
        metadata = _SUPPORTED_MODELS_METADATA.get(model_type)
        if metadata is None:
            return set()
        return {
            PROMPT_TYPE_TO_ANNOTATION_TYPE[pt]
            for pt in metadata.supported_prompt_types
            if pt in PROMPT_TYPE_TO_ANNOTATION_TYPE
        }
