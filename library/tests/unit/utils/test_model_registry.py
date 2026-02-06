# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model_registry module."""

import pytest

from instantlearn.registry import (
    MODEL_REGISTRY,
    Capability,
    Modality,
    ModelMetadata,
    ModelType,
    PromptType,
    get_available_capabilities,
    get_available_families,
    get_available_sizes,
    get_available_types,
    get_local_filename,
    get_model,
    get_models_by_capability,
    get_models_by_family,
    get_models_by_type,
    is_valid_model,
)


class TestStrEnums:
    """Tests for StrEnum classes."""

    def test_model_type_is_str_enum(self) -> None:
        """ModelType values can be compared with strings."""
        assert ModelType.ENCODER == "encoder"
        assert ModelType.SEGMENTER == "segmenter"
        assert ModelType.TRACKER == "tracker"

    def test_modality_is_str_enum(self) -> None:
        """Modality values can be compared with strings."""
        assert Modality.IMAGE == "image"
        assert Modality.VIDEO == "video"
        assert Modality.TEXT == "text"

    def test_prompt_type_is_str_enum(self) -> None:
        """PromptType values can be compared with strings."""
        assert PromptType.POINT == "point"
        assert PromptType.BOX == "box"
        assert PromptType.MASK == "mask"
        assert PromptType.TEXT == "text"
        assert PromptType.IMAGE == "image"

    def test_capability_is_str_enum(self) -> None:
        """Capability values can be compared with strings."""
        assert Capability.ENCODING == "encoding"
        assert Capability.SEGMENTATION == "segmentation"
        assert Capability.TRACKING == "tracking"
        assert Capability.DESCRIPTION == "description"

    def test_str_enum_conversion(self) -> None:
        """StrEnum can be constructed from string."""
        assert ModelType("encoder") == ModelType.ENCODER
        assert Capability("segmentation") == Capability.SEGMENTATION


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_model_metadata_is_frozen(self) -> None:
        """ModelMetadata instances are immutable."""
        model = get_model("sam-hq")
        assert model is not None
        with pytest.raises(AttributeError):
            model.id = "new-id"  # type: ignore[misc]

    def test_model_metadata_required_fields(self) -> None:
        """ModelMetadata requires all non-optional fields."""
        metadata = ModelMetadata(
            id="test-model",
            type=ModelType.ENCODER,
            family="Test",
            size="small",
            modalities=(Modality.IMAGE,),
            prompts=(),
            capabilities=(Capability.ENCODING,),
        )
        assert metadata.id == "test-model"
        assert metadata.weights_url is None
        assert metadata.hf_model_id is None


class TestModelRegistry:
    """Tests for MODEL_REGISTRY tuple."""

    def test_registry_is_not_empty(self) -> None:
        """Registry contains models."""
        assert len(MODEL_REGISTRY) > 0

    def test_registry_is_tuple(self) -> None:
        """Registry is immutable tuple."""
        assert isinstance(MODEL_REGISTRY, tuple)

    def test_all_entries_are_model_metadata(self) -> None:
        """All registry entries are ModelMetadata instances."""
        for model in MODEL_REGISTRY:
            assert isinstance(model, ModelMetadata)

    def test_model_ids_are_unique(self) -> None:
        """All model IDs are unique."""
        ids = [m.id for m in MODEL_REGISTRY]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"

    def test_registry_contains_expected_models(self) -> None:
        """Registry contains expected model families."""
        families = {m.family for m in MODEL_REGISTRY}
        assert "SAM-HQ" in families
        assert "SAM2" in families
        assert "DINOv2" in families


class TestGetModel:
    """Tests for get_model function."""

    def test_get_existing_model(self) -> None:
        """get_model returns metadata for existing model."""
        model = get_model("sam-hq")
        assert model is not None
        assert model.id == "sam-hq"
        assert model.family == "SAM-HQ"

    def test_get_nonexistent_model(self) -> None:
        """get_model returns None for nonexistent model."""
        model = get_model("nonexistent-model")
        assert model is None

    def test_get_model_case_sensitive(self) -> None:
        """get_model is case-sensitive."""
        assert get_model("SAM-HQ") is None
        assert get_model("sam-hq") is not None


class TestGetModelsByType:
    """Tests for get_models_by_type function."""

    def test_get_segmenters(self) -> None:
        """get_models_by_type returns all segmenters."""
        segmenters = get_models_by_type(ModelType.SEGMENTER)
        assert len(segmenters) > 0
        assert all(m.type == ModelType.SEGMENTER for m in segmenters)

    def test_get_encoders(self) -> None:
        """get_models_by_type returns all encoders."""
        encoders = get_models_by_type(ModelType.ENCODER)
        assert len(encoders) > 0
        assert all(m.type == ModelType.ENCODER for m in encoders)

    def test_get_models_by_type_with_string(self) -> None:
        """get_models_by_type works with string argument (StrEnum)."""
        # This works because ModelType is a StrEnum
        segmenters_enum = get_models_by_type(ModelType.SEGMENTER)
        segmenters_str = get_models_by_type("segmenter")  # type: ignore[arg-type]
        assert segmenters_enum == segmenters_str

    def test_get_trackers_empty(self) -> None:
        """get_models_by_type returns empty list for type with no models."""
        trackers = get_models_by_type(ModelType.TRACKER)
        assert trackers == []


class TestGetModelsByCapability:
    """Tests for get_models_by_capability function."""

    def test_get_segmentation_capable(self) -> None:
        """get_models_by_capability returns models with segmentation."""
        models = get_models_by_capability(Capability.SEGMENTATION)
        assert len(models) > 0
        assert all(Capability.SEGMENTATION in m.capabilities for m in models)

    def test_get_encoding_capable(self) -> None:
        """get_models_by_capability returns models with encoding."""
        models = get_models_by_capability(Capability.ENCODING)
        assert len(models) > 0
        assert all(Capability.ENCODING in m.capabilities for m in models)


class TestGetModelsByFamily:
    """Tests for get_models_by_family function."""

    def test_get_sam2_family(self) -> None:
        """get_models_by_family returns all SAM2 models."""
        models = get_models_by_family("SAM2")
        assert len(models) > 0
        assert all(m.family == "SAM2" for m in models)

    def test_get_dinov2_family(self) -> None:
        """get_models_by_family returns all DINOv2 models."""
        models = get_models_by_family("DINOv2")
        assert len(models) > 0
        assert all(m.family == "DINOv2" for m in models)

    def test_get_nonexistent_family(self) -> None:
        """get_models_by_family returns empty list for unknown family."""
        models = get_models_by_family("NonexistentFamily")
        assert models == []


class TestGetAvailableFunctions:
    """Tests for get_available_* functions."""

    def test_get_available_types(self) -> None:
        """get_available_types returns list of type strings."""
        types = get_available_types()
        assert isinstance(types, list)
        assert "encoder" in types
        assert "segmenter" in types

    def test_get_available_families(self) -> None:
        """get_available_families returns list of family strings."""
        families = get_available_families()
        assert isinstance(families, list)
        assert "SAM-HQ" in families
        assert "DINOv2" in families

    def test_get_available_sizes(self) -> None:
        """get_available_sizes returns list of size strings."""
        sizes = get_available_sizes()
        assert isinstance(sizes, list)
        assert "tiny" in sizes
        assert "small" in sizes
        assert "base" in sizes
        assert "large" in sizes

    def test_get_available_capabilities(self) -> None:
        """get_available_capabilities returns list of capability strings."""
        caps = get_available_capabilities()
        assert isinstance(caps, list)
        assert "encoding" in caps
        assert "segmentation" in caps


class TestIsValidModel:
    """Tests for is_valid_model function."""

    def test_valid_model(self) -> None:
        """is_valid_model returns True for existing model."""
        assert is_valid_model("sam-hq") is True
        assert is_valid_model("dinov2-base") is True

    def test_invalid_model(self) -> None:
        """is_valid_model returns False for nonexistent model."""
        assert is_valid_model("nonexistent") is False

    def test_valid_model_with_matching_type(self) -> None:
        """is_valid_model returns True when type matches."""
        assert is_valid_model("sam-hq", ModelType.SEGMENTER) is True
        assert is_valid_model("dinov2-base", ModelType.ENCODER) is True

    def test_valid_model_with_wrong_type(self) -> None:
        """is_valid_model returns False when type doesn't match."""
        assert is_valid_model("sam-hq", ModelType.ENCODER) is False
        assert is_valid_model("dinov2-base", ModelType.SEGMENTER) is False


class TestGetLocalFilename:
    """Tests for get_local_filename function."""

    def test_get_filename_from_url(self) -> None:
        """get_local_filename extracts filename from weights_url."""
        filename = get_local_filename("sam-hq")
        assert filename == "sam_hq_vit_h.pth"

    def test_get_filename_sam2(self) -> None:
        """get_local_filename works for SAM2 models."""
        filename = get_local_filename("sam2-tiny")
        assert filename == "sam2.1_hiera_tiny.pt"

    def test_get_filename_nonexistent_model(self) -> None:
        """get_local_filename raises ValueError for nonexistent model."""
        with pytest.raises(ValueError, match="not found in registry"):
            get_local_filename("nonexistent")

    def test_get_filename_no_weights_url(self) -> None:
        """get_local_filename raises ValueError for model without weights_url."""
        # DINOv2 models use hf_model_id, not weights_url
        with pytest.raises(ValueError, match="has no weights_url"):
            get_local_filename("dinov2-base")
