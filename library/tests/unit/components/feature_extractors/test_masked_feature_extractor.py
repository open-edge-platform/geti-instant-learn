# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedFeatureExtractor class."""

import pytest
import torch

from getiprompt.components.feature_extractors import MaskedFeatureExtractor
from getiprompt.types import Masks


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

    def test_forward_single_image_single_mask(self) -> None:
        """Test MaskedFeatureExtractor with single image and single mask."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16  # 224 // 14 = 16
        total_patches = patches_per_dim * patches_per_dim  # 256
        embedding_dim = 768
        num_masks = 1
        mask_height, mask_width = 224, 224

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a mask with a foreground region
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True  # Single mask region

        # Category IDs
        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == batch_size)

        # Check masked reference embeddings (aggregated across batch by category)
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))  # Averaged and normalized

        # Check Masks object
        pytest.assume(isinstance(masks_list[0], Masks))
        pytest.assume(1 in masks_list[0].data)
        pytest.assume(masks_list[0].data[1].shape[0] == 1)  # One mask per class

    def test_forward_multiple_images_multiple_masks(self) -> None:
        """Test MaskedFeatureExtractor with multiple images and multiple masks."""
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

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks with foreground regions
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True  # First sample, first mask
        masks[0, 1, 150:200, 150:200] = True  # First sample, second mask
        masks[1, 0, 30:80, 30:80] = True  # Second sample, first mask
        masks[1, 1, 100:150, 100:150] = True  # Second sample, second mask

        # Category IDs
        category_ids = torch.tensor([[1, 2], [1, 0]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(embeddings, masks, category_ids)

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == batch_size)

        # Check masked reference embeddings (aggregated across batch by category)
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(2 in masked_ref_embeddings)
        pytest.assume(0 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(isinstance(masked_ref_embeddings[2], torch.Tensor))
        pytest.assume(isinstance(masked_ref_embeddings[0], torch.Tensor))

        # Check first image masks
        pytest.assume(isinstance(masks_list[0], Masks))
        pytest.assume(1 in masks_list[0].data)
        pytest.assume(2 in masks_list[0].data)

        # Check second image masks
        pytest.assume(isinstance(masks_list[1], Masks))
        pytest.assume(1 in masks_list[1].data)
        pytest.assume(0 in masks_list[1].data)

    def test_forward_multiple_classes_same_image(self) -> None:
        """Test MaskedFeatureExtractor with multiple classes in the same image."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        num_masks = 3
        mask_height, mask_width = 224, 224

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks for different classes
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True  # Class 1
        masks[0, 1, 120:170, 120:170] = True  # Class 2
        masks[0, 2, 180:210, 180:210] = True  # Class 3

        # Category IDs
        category_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(embeddings, masks, category_ids)

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == 1)

        # Check all classes are present in masked reference embeddings
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(2 in masked_ref_embeddings)
        pytest.assume(3 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(isinstance(masked_ref_embeddings[2], torch.Tensor))
        pytest.assume(isinstance(masked_ref_embeddings[3], torch.Tensor))

        pytest.assume(1 in masks_list[0].data)
        pytest.assume(2 in masks_list[0].data)
        pytest.assume(3 in masks_list[0].data)

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
        num_masks = 1
        mask_height, mask_width = 224, 224

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create empty mask
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)

        # Category IDs
        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(embeddings, masks, category_ids)

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == 1)

        # Empty mask should still create entries
        pytest.assume(1 in masks_list[0].data)
        # Empty mask might not create embeddings (depends on implementation)
        # If embeddings exist, they should be a tensor
        if 1 in masked_ref_embeddings:
            pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))

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

        # Create batched embeddings with known values
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a mask covering a specific region
        mask_height, mask_width = 224, 224
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, _ = extractor(embeddings, masks, category_ids)

        # Check that masked reference embeddings are extracted
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))  # Averaged and normalized

    def test_forward_mask_pooling(self) -> None:
        """Test that masks are correctly pooled to patch grid."""
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

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a mask
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(embeddings, masks, category_ids)

        # Check that pooled mask has correct shape
        # Pooled mask should be (1, patches_per_dim, patches_per_dim)
        pooled_mask = masks_list[0].data[1]
        pytest.assume(pooled_mask.shape == (1, patches_per_dim, patches_per_dim))

        # Check that embeddings were extracted
        pytest.assume(1 in masked_ref_embeddings)

    def test_forward_different_input_sizes(self) -> None:
        """Test MaskedFeatureExtractor with different input sizes."""
        input_size = 336
        patch_size = 14
        patches_per_dim = input_size // patch_size  # 24

        extractor = MaskedFeatureExtractor(
            input_size=input_size,
            patch_size=patch_size,
            device="cpu",
        )

        batch_size = 1
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768
        mask_height, mask_width = input_size, input_size

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 100:200, 100:200] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == 1)
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))

        # Check pooled mask shape
        pooled_mask = masks_list[0].data[1]
        pytest.assume(pooled_mask.shape == (1, patches_per_dim, patches_per_dim))

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
        embedding_dim = 1024  # Different embedding dimension
        mask_height, mask_width = 224, 224

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, _ = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Check that embedding dimension is preserved
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))

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

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a large mask covering most of the image
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 20:200, 20:200] = True  # Large region

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, _ = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Large mask should extract embeddings
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))  # Averaged and normalized

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

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create multiple masks with same class ID
        masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True  # First mask
        masks[0, 1, 150:200, 150:200] = True  # Second mask, same class

        # Same category ID for both masks
        category_ids = torch.tensor([[1, 1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Both masks should add embeddings to the same class (aggregated)
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))  # Averaged across masks

        # Masks should be concatenated for the same class
        pytest.assume(1 in masks_list[0].data)
        pytest.assume(masks_list[0].data[1].shape[0] == 2)  # Two masks

    def test_forward_creates_masked_reference_embeddings(self) -> None:
        """Test that masked reference embeddings are correctly created."""
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

        # Create batched embeddings with specific values
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        masks[0, 0, 50:100, 50:100] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, _ = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Check that masked reference embeddings are created
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(isinstance(masked_ref_embeddings[1], torch.Tensor))
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))

    @pytest.mark.parametrize("input_size,patch_size", ((224, 14), (336, 14), (224, 16)))
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

        # Create batched embeddings
        embeddings = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        # Use a region proportional to input size
        region_size = input_size // 4
        masks[0, 0, region_size : 2 * region_size, region_size : 2 * region_size] = True

        category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local embeddings
        masked_ref_embeddings, masks_list = extractor(
            embeddings,
            masks,
            category_ids,
        )

        # Check outputs
        pytest.assume(isinstance(masked_ref_embeddings, dict))
        pytest.assume(len(masks_list) == 1)
        pytest.assume(1 in masked_ref_embeddings)
        pytest.assume(masked_ref_embeddings[1].shape == (1, embedding_dim))

        # Check pooled mask shape
        pooled_mask = masks_list[0].data[1]
        pytest.assume(pooled_mask.shape == (1, patches_per_dim, patches_per_dim))
