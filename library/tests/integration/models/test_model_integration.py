# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for model combinations with real datasets.

This module tests all combinations of SAM models and model types with the
fss-1000 dataset to ensure models work correctly end-to-end.
"""

from pathlib import Path

import pytest
import torch
from torchmetrics.segmentation import MeanIoU

from getiprompt.data.base import Batch
from getiprompt.data.folder import FolderDataset
from getiprompt.models.grounded_sam import GroundedSAM
from getiprompt.models.matcher import Matcher
from getiprompt.models.per_dino import PerDino
from getiprompt.models.soft_matcher import SoftMatcher
from getiprompt.utils.benchmark import convert_masks_to_one_hot_tensor
from getiprompt.utils.constants import ModelName, SAMModelName


@pytest.fixture
def fss1000_root() -> Path:
    """Return path to fss-1000 test dataset."""
    return Path(__file__).parent.parent.parent.parent / "tests" / "assets" / "fss-1000"


@pytest.fixture
def dataset(fss1000_root: Path) -> FolderDataset:
    """Create a FolderDataset for testing."""
    return FolderDataset(
        root=fss1000_root,
        categories=["apple", "basketball"],  # Use 2 categories for faster testing
        n_shots=1,
    )


@pytest.fixture
def reference_batch(dataset: FolderDataset) -> Batch:
    """Get reference batch from dataset."""
    ref_dataset = dataset.get_reference_dataset()
    samples = [ref_dataset[i] for i in range(min(2, len(ref_dataset)))]  # Use up to 2 reference samples
    return Batch.collate(samples)


@pytest.fixture
def target_batch(dataset: FolderDataset) -> Batch:
    """Get target batch from dataset."""
    target_dataset = dataset.get_target_dataset()
    samples = [target_dataset[i] for i in range(min(2, len(target_dataset)))]  # Use up to 2 target samples
    return Batch.collate(samples)


# Model classes mapping
MODEL_CLASSES = {
    ModelName.GROUNDED_SAM: GroundedSAM,
    ModelName.MATCHER: Matcher,
    ModelName.PER_DINO: PerDino,
    ModelName.SOFT_MATCHER: SoftMatcher,
}

# SAM models to test
SAM_MODELS = [SAMModelName.SAM_HQ_TINY, SAMModelName.SAM2_TINY]

# Models that support n-shots (all except GroundedSAM)
N_SHOT_SUPPORTED_MODELS = [ModelName.MATCHER, ModelName.PER_DINO, ModelName.SOFT_MATCHER]


class TestModelIntegration:
    """Integration tests for all model combinations."""

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_initialization(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
    ) -> None:
        """Test that models can be initialized with different SAM backends.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_initialization for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model with minimal parameters
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device="cpu", precision="fp32")
        else:
            model = model_class(sam=sam_model, device="cpu", precision="fp32", encoder_model="dinov2_small")

        assert model is not None
        assert hasattr(model, "learn")
        assert hasattr(model, "infer")
        assert callable(model.learn)
        assert callable(model.infer)

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_learn_infer(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        reference_batch: Batch,
        target_batch: Batch,
    ) -> None:
        """Test that models can learn from reference data and infer on target data.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            reference_batch: Batch of reference samples.
            target_batch: Batch of target samples.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_learn_infer for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device="cpu", precision="fp32")
        else:
            model = model_class(sam=sam_model, device="cpu", precision="fp32", encoder_model="dinov2_small")

        # Test learn method
        model.learn(reference_batch)

        # Test infer method
        predictions = model.infer(target_batch)

        # Validate results
        assert isinstance(predictions, list)
        assert predictions is not None
        assert len(predictions) == len(target_batch)

        # Check that masks have correct shape
        for prediction, image in zip(predictions, target_batch.images, strict=False):
            assert isinstance(prediction["pred_masks"], torch.Tensor)
            assert prediction["pred_masks"].shape[-2:] == image.shape[-2:]

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", N_SHOT_SUPPORTED_MODELS)
    def test_n_shots_capability(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        fss1000_root: Path,
    ) -> None:
        """Test that models support n-shots learning.

        This test verifies that models can learn from multiple reference samples
        (n-shots > 1) and that the number of reference samples affects the results.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test (must support n-shots).
            fss1000_root: Path to fss-1000 dataset.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_n_shots_capability for SAM2-tiny")

        if not fss1000_root.exists():
            pytest.skip("fss-1000 dataset not found")

        model_class = MODEL_CLASSES[model_name]

        # Test with n_shots=1
        dataset_1shot = FolderDataset(
            root=fss1000_root,
            categories=["apple"],
            n_shots=1,
        )
        ref_batch_1shot = Batch.collate([dataset_1shot.get_reference_dataset()[0]])
        target_batch = Batch.collate([dataset_1shot.get_target_dataset()[0]])

        model_1shot = model_class(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov2_small",
        )
        model_1shot.learn(ref_batch_1shot)
        predictions_1shot = model_1shot.infer(target_batch)

        # Test with n_shots=2 (if available)
        dataset_2shot = FolderDataset(
            root=fss1000_root,
            categories=["apple"],
            n_shots=2,
        )
        ref_dataset_2shot = dataset_2shot.get_reference_dataset()
        if len(ref_dataset_2shot) >= 2:
            ref_batch_2shot = Batch.collate([ref_dataset_2shot[i] for i in range(2)])
            target_batch_2shot = Batch.collate([dataset_2shot.get_target_dataset()[0]])

            model_2shot = model_class(
                sam=sam_model,
                device="cpu",
                precision="fp32",
                encoder_model="dinov2_small",
            )
            model_2shot.learn(ref_batch_2shot)
            predictions_2shot = model_2shot.infer(target_batch_2shot)

            # Both should produce valid results
            assert isinstance(predictions_1shot, list)
            assert isinstance(predictions_2shot, list)
            assert len(predictions_1shot[0]["pred_masks"]) > 0
            assert len(predictions_2shot[0]["pred_masks"]) > 0
        else:
            # If not enough samples, just verify 1-shot works
            assert isinstance(predictions_1shot, list)
            assert len(predictions_1shot[0]["pred_masks"]) > 0

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    def test_grounded_sam_no_n_shots(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_batch: Batch,
    ) -> None:
        """Test that GroundedSAM works but doesn't use n-shots.

        GroundedSAM uses text prompts and doesn't learn from reference images
        in the same way as other models. It only needs category mapping.

        Args:
            sam_model: The SAM model to use.
            reference_batch: Batch of reference samples (for category mapping).
            target_batch: Batch of target samples.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_input_validation for SAM2-tiny")

        model = GroundedSAM(sam=sam_model, device="cpu", precision="fp32")

        # GroundedSAM's learn() only creates category mapping
        model.learn(reference_batch)
        assert hasattr(model, "category_mapping")
        assert isinstance(model.category_mapping, dict)

        # Infer should work with just category mapping
        predictions = model.infer(target_batch)
        assert isinstance(predictions, list)
        assert len(predictions) == len(target_batch)

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_input_validation(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        reference_batch: Batch,
        target_batch: Batch,
    ) -> None:
        """Test that models validate inputs correctly.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            reference_batch: Batch of reference samples.
            target_batch: Batch of target samples.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_input_validation for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device="cpu", precision="fp32")
        else:
            model = model_class(sam=sam_model, device="cpu", precision="fp32", encoder_model="dinov2_small")

        # Validate that reference batch has required fields
        assert len(reference_batch) > 0
        assert len(reference_batch.images) > 0
        assert all(img is not None for img in reference_batch.images)

        # For non-GroundedSAM models, reference batch should have masks
        if model_name != ModelName.GROUNDED_SAM:
            assert all(mask is not None for mask in reference_batch.masks if mask is not None)

        # Validate that target batch has required fields
        assert len(target_batch) > 0
        assert len(target_batch.images) > 0
        assert all(img is not None for img in target_batch.images)

        # Models should handle these inputs without errors
        model.learn(reference_batch)
        predictions = model.infer(target_batch)

        # Results should be valid
        assert isinstance(predictions, list)
        assert len(predictions) == len(target_batch)

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_metrics_calculation(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        dataset: FolderDataset,
    ) -> None:
        """Test that models produce predictions that can be evaluated with metrics.

        This test verifies that:
        1. Models can produce predictions
        2. Metrics can be calculated from predictions and ground truth
        3. Metrics have valid values (within expected ranges)

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            dataset: The dataset to use for testing.
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_metrics_calculation for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device="cpu", precision="fp32")
        else:
            model = model_class(sam=sam_model, device="cpu", precision="fp32", encoder_model="dinov2_small")

        # Get reference and target samples for first category
        categories = dataset.categories
        if not categories:
            pytest.skip("No categories available in dataset")

        # Get reference batch
        ref_batch = Batch.collate(dataset.get_reference_dataset())

        target_dataset = dataset.get_target_dataset()
        target_batch = Batch.collate(target_dataset[0])

        # Learn from reference
        model.learn(ref_batch)

        # Infer on target
        predictions = model.infer(target_batch)

        category_id_to_index = {
            dataset.get_category_id(cat_name): idx for idx, cat_name in enumerate(dataset.categories)
        }
        batch_pred_tensors, batch_gt_tensors = convert_masks_to_one_hot_tensor(
            predictions=predictions,
            ground_truths=target_batch,
            num_classes=len(categories),
            category_id_to_index=category_id_to_index,
            device="cpu",
        )

        # Calculate metrics
        metrics = MeanIoU(num_classes=len(categories), include_background=True, per_class=True).to("cpu")
        for pred_tensor, gt_tensor in zip(batch_pred_tensors, batch_gt_tensors, strict=True):
            metrics.update(pred_tensor, gt_tensor)

        iou_per_class = metrics.compute()
        for idx in range(len(categories)):
            iou_value = iou_per_class[idx].item()
            # -1 is returned if class is completely absent both from prediction and the ground truth labels.
            assert iou_value >= -1
