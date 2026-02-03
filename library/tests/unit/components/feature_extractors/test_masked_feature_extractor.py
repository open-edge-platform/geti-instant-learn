# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedFeatureExtractor class."""

import pytest
import torch

from instantlearn.components.feature_extractors import MaskedFeatureExtractor
from instantlearn.components.feature_extractors.reference_features import ReferenceFeatures


class TestMaskedFeatureExtractor:
    """Test cases for MaskedFeatureExtractor class."""

    def test_init(self) -> None:
        """Test MaskedFeatureExtractor initialization."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )
        pytest.assume(isinstance(extractor, MaskedFeatureExtractor))
        pytest.assume(extractor.input_size == 224)
        pytest.assume(extractor.patch_size == 14)
        pytest.assume(extractor.device == "cpu")

    def test_forward_returns_reference_features(self) -> None:
        """Test that forward returns ReferenceFeatures dataclass."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(isinstance(ref_features, ReferenceFeatures))
        pytest.assume(isinstance(ref_features.ref_embeddings, torch.Tensor))
        pytest.assume(isinstance(ref_features.masked_ref_embeddings, torch.Tensor))
        pytest.assume(isinstance(ref_features.flatten_ref_masks, torch.Tensor))
        pytest.assume(isinstance(ref_features.category_ids, list))

    def test_forward_single_image_single_mask(self) -> None:
        """Test MaskedFeatureExtractor with single image and single mask."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        # Check category_ids
        pytest.assume(ref_features.category_ids == [1])

        # Check tensor shapes: [C, ...]
        num_categories = 1
        pytest.assume(ref_features.ref_embeddings.shape == (num_categories, total_patches, embedding_dim))
        pytest.assume(ref_features.masked_ref_embeddings.shape == (num_categories, 1, embedding_dim))
        pytest.assume(ref_features.flatten_ref_masks.shape == (num_categories, total_patches))

    def test_forward_multiple_images_same_categories(self) -> None:
        """Test MaskedFeatureExtractor with multiple images, same categories per image.

        Note: Current implementation requires all categories to have the same number
        of reference images to enable torch.stack in ReferenceFeatures.
        """
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 2
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        num_masks = 2
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        masks[0, 1, 150:200, 150:200] = True
        masks[1, 0, 30:80, 30:80] = True
        masks[1, 1, 100:150, 100:150] = True

        # Each category appears once per image (2 refs total per category)
        category_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        # Categories should be sorted: [1, 2]
        pytest.assume(ref_features.category_ids == [1, 2])

        num_categories = 2
        pytest.assume(ref_features.ref_embeddings.shape[0] == num_categories)
        pytest.assume(ref_features.masked_ref_embeddings.shape[0] == num_categories)
        pytest.assume(ref_features.flatten_ref_masks.shape[0] == num_categories)

        # Each category has 2 reference images (one from each batch sample)
        pytest.assume(ref_features.ref_embeddings.shape == (num_categories, total_patches * 2, embedding_dim))
        pytest.assume(ref_features.flatten_ref_masks.shape == (num_categories, total_patches * 2))

    def test_forward_empty_mask(self) -> None:
        """Test MaskedFeatureExtractor with empty mask."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(ref_features.category_ids == [1])
        # Empty mask should produce empty masked embeddings but still have ref embeddings
        pytest.assume(ref_features.masked_ref_embeddings[0].shape == (0, embedding_dim))
        pytest.assume(ref_features.flatten_ref_masks[0].sum() == 0)
        pytest.assume(ref_features.ref_embeddings[0].shape == (total_patches, embedding_dim))

    def test_forward_feature_extraction_correctness(self) -> None:
        """Test that embeddings are correctly extracted from masked regions."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        mask_height, mask_width = 224, 224
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(ref_features.category_ids == [1])
        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))

    def test_forward_different_input_sizes(self) -> None:
        """Test MaskedFeatureExtractor with different input sizes."""
        input_size = 336
        patch_size = 14
        patches_per_dim = input_size // patch_size

        extractor = MaskedFeatureExtractor(
            input_size=input_size,
            patch_size=patch_size,
            device="cpu",
        )

        batch_size = 1
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = input_size, input_size

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 100:200, 100:200] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(ref_features.category_ids == [1])
        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))
        pytest.assume(ref_features.flatten_ref_masks.shape == (1, total_patches))

    def test_forward_different_embedding_dims(self) -> None:
        """Test MaskedFeatureExtractor with different embedding dimensions."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 1024
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))

    def test_forward_large_mask_region(self) -> None:
        """Test MaskedFeatureExtractor with a large mask region."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 20:200, 20:200] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(ref_features.category_ids == [1])
        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))

    def test_forward_same_class_id_multiple_masks(self) -> None:
        """Test MaskedFeatureExtractor with multiple masks for the same class."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        num_masks = 2
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        masks[0, 1, 150:200, 150:200] = True
        category_ids = torch.tensor([[1, 1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        # Both masks have same category, so aggregated
        pytest.assume(ref_features.category_ids == [1])
        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))
        # flatten_ref_masks should have 2x patches (one per mask)
        pytest.assume(ref_features.flatten_ref_masks.shape == (1, total_patches * 2))

    def test_forward_masked_embeddings_normalized(self) -> None:
        """Test that masked reference embeddings are normalized."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = 224, 224

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        # Check that masked_ref_embeddings are normalized (unit norm)
        norm = ref_features.masked_ref_embeddings[0].norm(dim=-1)
        pytest.assume(torch.allclose(norm, torch.ones_like(norm), atol=1e-5))

    @pytest.mark.parametrize(("input_size", "patch_size"), [(224, 14), (336, 14), (224, 16)])
    def test_forward_different_configurations(self, input_size: int, patch_size: int) -> None:
        """Test MaskedFeatureExtractor with different input_size and patch_size configurations."""
        extractor = MaskedFeatureExtractor(
            input_size=input_size,
            patch_size=patch_size,
            device="cpu",
        )

        patches_per_dim = input_size // patch_size
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        batch_size = 1
        mask_height, mask_width = input_size, input_size

        embeddings = torch.randn(batch_size, total_patches, embedding_dim)
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        region_size = input_size // 4
        masks[0, 0, region_size : 2 * region_size, region_size : 2 * region_size] = True
        category_ids = torch.tensor([[1]], dtype=torch.long)

        ref_features = extractor(embeddings, masks, category_ids)

        pytest.assume(isinstance(ref_features, ReferenceFeatures))
        pytest.assume(ref_features.category_ids == [1])
        pytest.assume(ref_features.masked_ref_embeddings.shape == (1, 1, embedding_dim))
        pytest.assume(ref_features.flatten_ref_masks.shape == (1, total_patches))
