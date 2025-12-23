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
    hf_revision: str | None = None  # HuggingFace model revision/commit hash
    config_filename: str | None = None
    sha_sum: str | None = None  # SHA256 checksum for download verification


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
        sha_sum="a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
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
        sha_sum="0f32c075ccdd870ae54db2f7630e7a0878ede5a2b06d05d6fe02c65a82fb7196",
    ),
    # SAM2 family
    ModelMetadata(
        id="sam2-tiny",
        type=ModelType.SEGMENTER,
        family="SAM2",
        size="tiny",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        config_filename="sam2.1_hiera_t.yaml",
        sha_sum="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    ),
    ModelMetadata(
        id="sam2-small",
        type=ModelType.SEGMENTER,
        family="SAM2",
        size="small",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        config_filename="sam2.1_hiera_s.yaml",
        sha_sum="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    ),
    ModelMetadata(
        id="sam2-base",
        type=ModelType.SEGMENTER,
        family="SAM2",
        size="base",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        config_filename="sam2.1_hiera_b+.yaml",
        sha_sum="a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
    ),
    ModelMetadata(
        id="sam2-large",
        type=ModelType.SEGMENTER,
        family="SAM2",
        size="large",
        modalities=(Modality.IMAGE,),
        prompts=(PromptType.POINT, PromptType.BOX, PromptType.MASK),
        capabilities=(Capability.SEGMENTATION,),
        weights_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        config_filename="sam2.1_hiera_l.yaml",
        sha_sum="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ),
    # -------------------------------------------------------------------------
    # Encoders (DINOv2 family - HuggingFace)
    # -------------------------------------------------------------------------
    ModelMetadata(
        id="dinov2-small",
        type=ModelType.ENCODER,
        family="DINOv2",
        size="small",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="facebook/dinov2-with-registers-small",
        hf_revision="0d9846e56b43a21fa46d7f3f5070f0506a5795a9",
    ),
    ModelMetadata(
        id="dinov2-base",
        type=ModelType.ENCODER,
        family="DINOv2",
        size="base",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="facebook/dinov2-with-registers-base",
        hf_revision="a1d738ccfa7ae170945f210395d99dde8adb1805",
    ),
    ModelMetadata(
        id="dinov2-large",
        type=ModelType.ENCODER,
        family="DINOv2",
        size="large",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="facebook/dinov2-with-registers-large",
        hf_revision="e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
    ),
    ModelMetadata(
        id="dinov2-giant",
        type=ModelType.ENCODER,
        family="DINOv2",
        size="giant",
        modalities=(Modality.IMAGE,),
        prompts=(),
        capabilities=(Capability.ENCODING,),
        hf_model_id="facebook/dinov2-with-registers-giant",
        hf_revision="8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb",
    ),
    # -------------------------------------------------------------------------
    # Encoders (DINOv3 family - TIMM)
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


def get_local_filename(model_id: str) -> str:
    """Get the local filename for a model's weights.

    Derives filename from the weights URL.

    Args:
        model_id: The model ID.

    Returns:
        The local filename for storing model weights.

    Raises:
        ValueError: If model not found or has no weights_url.
    """
    model = get_model(model_id)
    if model is None:
        msg = f"Model '{model_id}' not found in registry"
        raise ValueError(msg)
    if model.weights_url is None:
        msg = f"Model '{model_id}' has no weights_url"
        raise ValueError(msg)
    return model.weights_url.split("/")[-1]