# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified Dino model."""

from unittest.mock import patch

import pytest

from getiprompt.models.dino import Dino, DinoSize, DinoVersion


class TestDinoVersion:
    """Test DinoVersion enum."""

    @staticmethod
    def test_version_values() -> None:
        """Test that version enum has correct values."""
        assert DinoVersion.V2.value == "v2"
        assert DinoVersion.V3.value == "v3"

    @staticmethod
    def test_version_from_string() -> None:
        """Test creating version from string."""
        assert DinoVersion("v2") == DinoVersion.V2
        assert DinoVersion("v3") == DinoVersion.V3


class TestDinoSize:
    """Test DinoSize enum."""

    @staticmethod
    def test_size_values() -> None:
        """Test that size enum has correct values."""
        assert DinoSize.SMALL.value == "small"
        assert DinoSize.BASE.value == "base"
        assert DinoSize.LARGE.value == "large"
        assert DinoSize.GIANT.value == "giant"
        assert DinoSize.SMALL_PLUS.value == "small_plus"
        assert DinoSize.HUGE.value == "huge"

    @staticmethod
    def test_size_from_string() -> None:
        """Test creating size from string."""
        assert DinoSize.from_str("small") == DinoSize.SMALL
        assert DinoSize.from_str("base") == DinoSize.BASE
        assert DinoSize.from_str("large") == DinoSize.LARGE
        assert DinoSize.from_str("giant") == DinoSize.GIANT
        assert DinoSize.from_str("small_plus") == DinoSize.SMALL_PLUS
        assert DinoSize.from_str("huge") == DinoSize.HUGE

    @staticmethod
    def test_size_from_string_case_insensitive() -> None:
        """Test that size from string is case insensitive."""
        assert DinoSize.from_str("SMALL") == DinoSize.SMALL
        assert DinoSize.from_str("Large") == DinoSize.LARGE


class TestDinoModel:
    """Test the unified Dino model."""

    @staticmethod
    def test_invalid_size_for_version() -> None:
        """Test that invalid sizes are rejected for each version."""
        # Test DinoV2 with DinoV3-only size
        with pytest.raises(ValueError, match="Size huge is not valid for DinoV2"):
            Dino(version="v2", size="huge")

        # Test DinoV3 with DinoV2-only size
        with pytest.raises(ValueError, match="Size giant is not valid for DinoV3"):
            Dino(version="v3", size="giant")

    @staticmethod
    def test_model_id_mapping() -> None:
        """Test that model IDs are correctly mapped."""
        from getiprompt.models.dino import DINO_V2_MODEL_IDS, DINO_V3_MODEL_IDS

        # Test DinoV2 model IDs
        assert DINO_V2_MODEL_IDS["small"] == "facebook/dinov2-with-registers-small"
        assert DINO_V2_MODEL_IDS["base"] == "facebook/dinov2-with-registers-base"
        assert DINO_V2_MODEL_IDS["large"] == "facebook/dinov2-with-registers-large"
        assert DINO_V2_MODEL_IDS["giant"] == "facebook/dinov2-with-registers-giant"

        # Test DinoV3 model IDs
        assert DINO_V3_MODEL_IDS["small"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
        assert DINO_V3_MODEL_IDS["small_plus"] == "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        assert DINO_V3_MODEL_IDS["base"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
        assert DINO_V3_MODEL_IDS["large"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
        assert DINO_V3_MODEL_IDS["huge"] == "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    @staticmethod
    def test_string_parameter_conversion() -> None:
        """Test that string parameters are correctly converted to enums."""
        try:
            model = Dino(version="v2", size="large")
            assert model.version == DinoVersion.V2
            assert model.size == DinoSize.LARGE
        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_default_parameters() -> None:
        """Test that default parameters work correctly."""
        try:
            model = Dino()
            assert model.version == DinoVersion.V3  # Default is V3
            assert model.size == DinoSize.LARGE
        except ValueError as e:
            # This is expected if model files are not available or user doesn't have access
            if "User does not have access to the weights" in str(e):
                pass
            else:
                raise

    @staticmethod
    def test_huggingface_error_handling() -> None:
        """Test that HuggingFace access errors are handled properly."""
        with patch("getiprompt.models.dino.AutoModel") as mock_model:
            mock_model.from_pretrained.side_effect = OSError("You are trying to access a gated repo.")

            with pytest.raises(ValueError, match="User does not have access to the weights"):
                Dino(version="v3", size="large")

    @staticmethod
    def test_validate_size_for_version() -> None:
        """Test the size validation logic."""
        from getiprompt.models.dino import Dino

        # Test DinoV2 valid sizes
        valid_v2_sizes = [DinoSize.SMALL, DinoSize.BASE, DinoSize.LARGE, DinoSize.GIANT]
        for size in valid_v2_sizes:
            # This should not raise an error
            Dino._validate_size_for_version(DinoVersion.V2, size)  # noqa: SLF001

        # Test DinoV3 valid sizes
        valid_v3_sizes = [
            DinoSize.SMALL,
            DinoSize.SMALL_PLUS,
            DinoSize.BASE,
            DinoSize.LARGE,
            DinoSize.HUGE,
        ]
        for size in valid_v3_sizes:
            # This should not raise an error
            Dino._validate_size_for_version(DinoVersion.V3, size)  # noqa: SLF001

        # Test invalid combinations
        with pytest.raises(ValueError, match="Size huge is not valid for DinoV2"):
            Dino._validate_size_for_version(DinoVersion.V2, DinoSize.HUGE)  # noqa: SLF001

        with pytest.raises(ValueError, match="Size giant is not valid for DinoV3"):
            Dino._validate_size_for_version(DinoVersion.V3, DinoSize.GIANT)  # noqa: SLF001

    @staticmethod
    def test_get_model_id() -> None:
        """Test the model ID mapping logic."""
        from getiprompt.models.dino import Dino

        # Test DinoV2 model IDs
        assert Dino._get_model_id(DinoVersion.V2, DinoSize.SMALL) == "facebook/dinov2-with-registers-small"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V2, DinoSize.BASE) == "facebook/dinov2-with-registers-base"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V2, DinoSize.LARGE) == "facebook/dinov2-with-registers-large"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V2, DinoSize.GIANT) == "facebook/dinov2-with-registers-giant"  # noqa: SLF001

        # Test DinoV3 model IDs
        assert Dino._get_model_id(DinoVersion.V3, DinoSize.SMALL) == "facebook/dinov3-vits16-pretrain-lvd1689m"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V3, DinoSize.SMALL_PLUS) == "facebook/dinov3-vits16plus-pretrain-lvd1689m"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V3, DinoSize.BASE) == "facebook/dinov3-vitb16-pretrain-lvd1689m"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V3, DinoSize.LARGE) == "facebook/dinov3-vitl16-pretrain-lvd1689m"  # noqa: SLF001
        assert Dino._get_model_id(DinoVersion.V3, DinoSize.HUGE) == "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # noqa: SLF001

        # Test invalid version
        with pytest.raises(ValueError, match="Unsupported version"):
            Dino._get_model_id("invalid", DinoSize.LARGE)  # noqa: SLF001

    @staticmethod
    def test_ignore_token_lengths() -> None:
        """Test that ignore token lengths are correct for each version."""
        # Test DinoV2 ignore token length (5 for registers)
        assert Dino._get_ignore_token_length(DinoVersion.V2) == 5  # noqa: SLF001

        # Test DinoV3 ignore token length (1 for CLS only)
        assert Dino._get_ignore_token_length(DinoVersion.V3) == 1  # noqa: SLF001

    @staticmethod
    def test_model_loading_logic() -> None:
        """Test the model loading logic without actual loading."""
        # Test that the model ID selection works correctly
        v2_id = Dino._get_model_id(DinoVersion.V2, DinoSize.LARGE)  # noqa: SLF001
        assert v2_id == "facebook/dinov2-with-registers-large"

        v3_id = Dino._get_model_id(DinoVersion.V3, DinoSize.LARGE)  # noqa: SLF001
        assert v3_id == "facebook/dinov3-vitl16-pretrain-lvd1689m"


if __name__ == "__main__":
    pytest.main([__file__])
