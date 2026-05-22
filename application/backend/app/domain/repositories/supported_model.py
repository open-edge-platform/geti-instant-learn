# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-memory repository for supported model metadata."""

from domain.db.models import PromptType
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
        default_config=MatcherConfig(),
        supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON],
        display_name="Matcher",
    ),
    ModelType.PERDINO: SupportedModelMetadataSchema(
        default_config=PerDinoConfig(),
        supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON],
        display_name="PerDINO",
    ),
    ModelType.SOFT_MATCHER: SupportedModelMetadataSchema(
        default_config=SoftMatcherConfig(),
        supported_prompt_types=[SupportedPromptType.VISUAL_POLYGON],
        display_name="SoftMatcher",
    ),
    ModelType.SAM3: SupportedModelMetadataSchema(
        default_config=Sam3Config(),
        supported_prompt_types=[SupportedPromptType.TEXT, SupportedPromptType.VISUAL_RECTANGLE],
        display_name="SAM3",
    ),
}

_VISUAL_PROMPT_TYPES = {SupportedPromptType.VISUAL_POLYGON, SupportedPromptType.VISUAL_RECTANGLE}

# the model+mode that will be set active when a new project is created
DEFAULT_ACTIVE_MODEL: tuple[ModelType, PromptType] = (ModelType.SOFT_MATCHER, PromptType.VISUAL)


class SupportedModelRepository:
    """Read-only repository exposing supported model metadata."""

    @staticmethod
    def get_all() -> list[SupportedModelMetadataSchema]:
        """Return metadata for all supported models."""
        return list(_SUPPORTED_MODELS_METADATA.values())

    @staticmethod
    def get_by_model_type(model_type: ModelType) -> SupportedModelMetadataSchema | None:
        """Return metadata for a single model type, or None if unknown."""
        return _SUPPORTED_MODELS_METADATA.get(model_type)

    @staticmethod
    def get_all_model_mode_pairs() -> list[tuple[ModelType, PromptType]]:
        """Return all (model_type, prompt_mode) pairs that should exist per project.

        Single-mode models yield one pair. Dual-mode models (e.g. SAM3) yield two pairs.
        """
        pairs: list[tuple[ModelType, PromptType]] = []
        for model_type, metadata in _SUPPORTED_MODELS_METADATA.items():
            supports_visual = bool(_VISUAL_PROMPT_TYPES & set(metadata.supported_prompt_types))
            supports_text = SupportedPromptType.TEXT in metadata.supported_prompt_types
            if supports_visual:
                pairs.append((model_type, PromptType.VISUAL))
            if supports_text:
                pairs.append((model_type, PromptType.TEXT))
        return pairs

    @staticmethod
    def get_supported_annotation_types(model_type: ModelType) -> set[AnnotationType]:
        """Derive supported annotation types for a model."""
        metadata = _SUPPORTED_MODELS_METADATA.get(model_type)
        if metadata is None:
            return set()
        return {
            PROMPT_TYPE_TO_ANNOTATION_TYPE[pt]
            for pt in metadata.supported_prompt_types
            if pt in PROMPT_TYPE_TO_ANNOTATION_TYPE
        }

    @staticmethod
    def model_type_supports_prompt_mode(model_type: ModelType, prompt_mode: PromptType) -> bool:
        """Check if a model type supports the given prompt mode.

        Args:
            model_type: The model type to check.
            prompt_mode: The prompt mode to filter by.

        Returns:
            True if the model type supports the prompt mode.
        """
        metadata = _SUPPORTED_MODELS_METADATA.get(model_type)
        if metadata is None:
            return False
        if prompt_mode == PromptType.VISUAL:
            return bool(_VISUAL_PROMPT_TYPES & set(metadata.supported_prompt_types))
        if prompt_mode == PromptType.TEXT:
            return SupportedPromptType.TEXT in metadata.supported_prompt_types
        return False
