# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaskedFeatureExtractor class."""

import pytest
import torch

from getiprompt.components.feature_extractors import MaskedFeatureExtractor
from getiprompt.types import Features, Masks


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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a mask with a foreground region
        batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True  # Single mask region

        # Category IDs
        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == batch_size)
        pytest.assume(len(masks_list) == batch_size)

        # Check Features object
        pytest.assume(isinstance(features_list[0], Features))
        pytest.assume(features_list[0].global_features.shape == (total_patches, embedding_dim))
        pytest.assume(1 in features_list[0].local_features)
        pytest.assume(len(features_list[0].local_features[1]) > 0)

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks with foreground regions
        batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True  # First sample, first mask
        batched_masks[0, 1, 150:200, 150:200] = True  # First sample, second mask
        batched_masks[1, 0, 30:80, 30:80] = True  # Second sample, first mask
        batched_masks[1, 1, 100:150, 100:150] = True  # Second sample, second mask

        # Category IDs
        batched_category_ids = torch.tensor([[1, 2], [1, 0]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == batch_size)
        pytest.assume(len(masks_list) == batch_size)

        # Check first image
        pytest.assume(isinstance(features_list[0], Features))
        pytest.assume(1 in features_list[0].local_features)
        pytest.assume(2 in features_list[0].local_features)
        pytest.assume(isinstance(masks_list[0], Masks))
        pytest.assume(1 in masks_list[0].data)
        pytest.assume(2 in masks_list[0].data)

        # Check second image
        pytest.assume(isinstance(features_list[1], Features))
        pytest.assume(1 in features_list[1].local_features)
        pytest.assume(0 in features_list[1].local_features)
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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks for different classes
        batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True  # Class 1
        batched_masks[0, 1, 120:170, 120:170] = True  # Class 2
        batched_masks[0, 2, 180:210, 180:210] = True  # Class 3

        # Category IDs
        batched_category_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == 1)
        pytest.assume(len(masks_list) == 1)

        # Check all classes are present
        pytest.assume(1 in features_list[0].local_features)
        pytest.assume(2 in features_list[0].local_features)
        pytest.assume(3 in features_list[0].local_features)

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create empty mask
        batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)

        # Category IDs
        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == 1)
        pytest.assume(len(masks_list) == 1)

        # Empty mask should still create entries
        pytest.assume(1 in masks_list[0].data)
        # Local features might be empty or have zero features
        pytest.assume(1 in features_list[0].local_features)

    def test_forward_feature_extraction_correctness(self) -> None:
        """Test that features are correctly extracted from masked regions."""
        extractor = MaskedFeatureExtractor(
            input_size=224,
            patch_size=14,
            device="cpu",
        )

        batch_size = 1
        patches_per_dim = 16
        total_patches = patches_per_dim * patches_per_dim
        embedding_dim = 768

        # Create batched features with known values
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)
        # Store a reference to the original features
        original_features = batched_features[0].clone()

        # Create a mask covering a specific region
        mask_height, mask_width = 224, 224
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, _ = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check that global features are preserved
        pytest.assume(torch.equal(features_list[0].global_features, original_features))

        # Check that local features are extracted (should have some features)
        pytest.assume(1 in features_list[0].local_features)
        pytest.assume(len(features_list[0].local_features[1]) > 0)
        local_features = features_list[0].local_features[1][0]
        pytest.assume(local_features.shape[1] == embedding_dim)

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a mask
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        _, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check that pooled mask has correct shape
        # Pooled mask should be (1, patches_per_dim, patches_per_dim)
        pooled_mask = masks_list[0].data[1]
        pytest.assume(pooled_mask.shape == (1, patches_per_dim, patches_per_dim))

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 100:200, 100:200] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == 1)
        pytest.assume(len(masks_list) == 1)
        pytest.assume(features_list[0].global_features.shape == (total_patches, embedding_dim))

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, _ = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check that embedding dimension is preserved
        pytest.assume(features_list[0].global_features.shape[1] == embedding_dim)
        pytest.assume(1 in features_list[0].local_features)
        local_features = features_list[0].local_features[1][0]
        pytest.assume(local_features.shape[1] == embedding_dim)

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create a large mask covering most of the image
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 20:200, 20:200] = True  # Large region

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, _ = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Large mask should extract more features
        pytest.assume(1 in features_list[0].local_features)
        local_features = features_list[0].local_features[1][0]
        pytest.assume(local_features.shape[0] > 0)  # Should have features

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create multiple masks with same class ID
        batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True  # First mask
        batched_masks[0, 1, 150:200, 150:200] = True  # Second mask, same class

        # Same category ID for both masks
        batched_category_ids = torch.tensor([[1, 1]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Both masks should add features to the same class
        pytest.assume(1 in features_list[0].local_features)
        # Should have 2 entries (one per mask)
        pytest.assume(len(features_list[0].local_features[1]) == 2)

        # Masks should be concatenated for the same class
        pytest.assume(1 in masks_list[0].data)
        pytest.assume(masks_list[0].data[1].shape[0] == 2)  # Two masks

    def test_forward_preserves_global_features(self) -> None:
        """Test that global features are preserved and not modified."""
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

        # Create batched features with specific values
        original_features = torch.randn(total_patches, embedding_dim)
        batched_features = original_features.unsqueeze(0)

        # Create masks
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        batched_masks[0, 0, 50:100, 50:100] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, _ = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check that global features are unchanged
        pytest.assume(torch.equal(features_list[0].global_features, original_features))

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

        # Create batched features
        batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        # Create masks
        batched_masks = torch.zeros(batch_size, 1, mask_height, mask_width, dtype=torch.bool)
        # Use a region proportional to input size
        region_size = input_size // 4
        batched_masks[0, 0, region_size : 2 * region_size, region_size : 2 * region_size] = True

        batched_category_ids = torch.tensor([[1]], dtype=torch.long)

        # Extract local features
        features_list, masks_list = extractor(
            batched_features,
            batched_masks,
            batched_category_ids,
        )

        # Check outputs
        pytest.assume(len(features_list) == 1)
        pytest.assume(len(masks_list) == 1)
        pytest.assume(features_list[0].global_features.shape == (total_patches, embedding_dim))

        # Check pooled mask shape
        pooled_mask = masks_list[0].data[1]
        pytest.assume(pooled_mask.shape == (1, patches_per_dim, patches_per_dim))
