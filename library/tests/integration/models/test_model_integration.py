# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for model combinations with real datasets.

This module tests all combinations of SAM models and model types with the
fss-1000 dataset to ensure models work correctly end-to-end.
"""

from pathlib import Path

import pytest

from getiprompt.data.base import Batch
from getiprompt.data.folder import FolderDataset
from getiprompt.metrics import SegmentationMetrics
from getiprompt.models.grounded_sam import GroundedSAM
from getiprompt.models.matcher import Matcher
from getiprompt.models.per_dino import PerDino
from getiprompt.models.soft_matcher import SoftMatcher
from getiprompt.types import Masks, Results
from getiprompt.utils.constants import ModelName, SAMModelName
from getiprompt.utils.utils import masks_to_custom_masks


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
        device: str = "cpu",
    ) -> None:
        """Test that models can be initialized with different SAM backends.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_initialization for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model with minimal parameters
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device=device, precision="fp32")
        else:
            model = model_class(sam=sam_model, device=device, precision="fp32", encoder_model="dinov3_small")

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
        device: str = "cpu",
    ) -> None:
        """Test that models can learn from reference data and infer on target data.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            reference_batch: Batch of reference samples.
            target_batch: Batch of target samples.
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_learn_infer for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device=device, precision="fp32")
        else:
            model = model_class(sam=sam_model, device=device, precision="fp32", encoder_model="dinov3_small")

        # Test learn method
        model.learn(reference_batch)

        # Test infer method
        results = model.infer(target_batch)

        # Validate results
        assert isinstance(results, Results)
        assert results.masks is not None
        assert len(results.masks) == len(target_batch)

        # Check that masks have correct shape
        for mask in results.masks:
            assert mask is not None
            assert isinstance(mask, Masks)

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_outputs_structure(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        reference_batch: Batch,
        target_batch: Batch,
        device: str = "cpu",
    ) -> None:
        """Test that model outputs have the correct structure.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            reference_batch: Batch of reference samples.
            target_batch: Batch of target samples.
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_outputs_structure for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device=device, precision="fp32")
        else:
            model = model_class(sam=sam_model, device=device, precision="fp32", encoder_model="dinov3_small")

        # Run learn and infer
        model.learn(reference_batch)
        results = model.infer(target_batch)

        # Validate Results structure
        assert isinstance(results, Results)
        assert hasattr(results, "masks")

        # Check masks
        assert results.masks is not None
        assert isinstance(results.masks, list)
        assert len(results.masks) == len(target_batch)

        # Model-specific output checks
        if model_name == ModelName.GROUNDED_SAM:
            # GroundedSAM should have box_prompts and used_boxes
            assert hasattr(results, "box_prompts")
            assert hasattr(results, "used_boxes")
        else:
            # Other models should have point_prompts and used_points
            assert hasattr(results, "point_prompts")
            assert hasattr(results, "used_points")
            # They may also have similarities (set as dynamic attribute)
            # Check if it exists, but don't fail if it doesn't
            if hasattr(results, "similarities"):
                assert results.similarities is not None

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", N_SHOT_SUPPORTED_MODELS)
    def test_n_shots_capability(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        fss1000_root: Path,
        device: str = "cpu",
    ) -> None:
        """Test that models support n-shots learning.

        This test verifies that models can learn from multiple reference samples
        (n-shots > 1) and that the number of reference samples affects the results.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test (must support n-shots).
            fss1000_root: Path to fss-1000 dataset.
            device: Device to run on (default: cpu for testing).
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
            device=device,
            precision="fp32",
            encoder_model="dinov3_small",
        )
        model_1shot.learn(ref_batch_1shot)
        results_1shot = model_1shot.infer(target_batch)

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
                device=device,
                precision="fp32",
                encoder_model="dinov3_small",
            )
            model_2shot.learn(ref_batch_2shot)
            results_2shot = model_2shot.infer(target_batch_2shot)

            # Both should produce valid results
            assert isinstance(results_1shot, Results)
            assert isinstance(results_2shot, Results)
            assert len(results_1shot.masks) > 0
            assert len(results_2shot.masks) > 0
        else:
            # If not enough samples, just verify 1-shot works
            assert isinstance(results_1shot, Results)
            assert len(results_1shot.masks) > 0

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    def test_grounded_sam_no_n_shots(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_batch: Batch,
        device: str = "cpu",
    ) -> None:
        """Test that GroundedSAM works but doesn't use n-shots.

        GroundedSAM uses text prompts and doesn't learn from reference images
        in the same way as other models. It only needs category mapping.

        Args:
            sam_model: The SAM model to use.
            reference_batch: Batch of reference samples (for category mapping).
            target_batch: Batch of target samples.
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_input_validation for SAM2-tiny")

        model = GroundedSAM(sam=sam_model, device=device, precision="fp32")

        # GroundedSAM's learn() only creates category mapping
        model.learn(reference_batch)
        assert hasattr(model, "category_mapping")
        assert isinstance(model.category_mapping, dict)

        # Infer should work with just category mapping
        results = model.infer(target_batch)
        assert isinstance(results, Results)
        assert results.masks is not None
        assert len(results.masks) == len(target_batch)

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_input_validation(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        reference_batch: Batch,
        target_batch: Batch,
        device: str = "cpu",
    ) -> None:
        """Test that models validate inputs correctly.

        Args:
            sam_model: The SAM model to use.
            model_name: The model type to test.
            reference_batch: Batch of reference samples.
            target_batch: Batch of target samples.
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_input_validation for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device=device, precision="fp32")
        else:
            model = model_class(sam=sam_model, device=device, precision="fp32", encoder_model="dinov3_small")

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
        results = model.infer(target_batch)

        # Results should be valid
        assert isinstance(results, Results)
        assert results.masks is not None

    @pytest.mark.parametrize("sam_model", SAM_MODELS)
    @pytest.mark.parametrize("model_name", ModelName)
    def test_model_metrics_calculation(
        self,
        sam_model: SAMModelName,
        model_name: ModelName,
        dataset: FolderDataset,
        device: str = "cpu",
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
            device: Device to run on (default: cpu for testing).
        """
        # TODO(Eugene): SAM2 is currently not supported due to a bug in the SAM2 model.
        # https://github.com/open-edge-platform/geti-prompt/issues/367
        if sam_model == SAMModelName.SAM2_TINY:
            pytest.skip("Skipping test_model_metrics_calculation for SAM2-tiny")

        model_class = MODEL_CLASSES[model_name]

        # Initialize model
        if model_name == ModelName.GROUNDED_SAM:
            model = model_class(sam=sam_model, device=device, precision="fp32")
        else:
            model = model_class(sam=sam_model, device=device, precision="fp32", encoder_model="dinov3_small")

        # Get reference and target samples for first category
        categories = dataset.categories
        if not categories:
            pytest.skip("No categories available in dataset")

        category_name = categories[0]
        category_id = dataset.get_category_id(category_name)

        # Get reference batch
        ref_dataset = dataset.get_reference_dataset(category=category_name)
        if len(ref_dataset) == 0:
            pytest.skip(f"No reference samples for category: {category_name}")
        ref_batch = Batch.collate([ref_dataset[0]])

        # Get target batch with ground truth masks
        target_dataset = dataset.get_target_dataset(category=category_name)
        if len(target_dataset) == 0:
            pytest.skip(f"No target samples for category: {category_name}")
        target_batch = Batch.collate([target_dataset[0]])

        # Ensure target batch has ground truth masks
        if target_batch.masks[0] is None:
            pytest.skip("No ground truth masks available for target sample")

        # Learn from reference
        model.learn(ref_batch)

        # Infer on target
        results = model.infer(target_batch)

        # Validate predictions exist
        assert isinstance(results, Results)
        assert results.masks is not None
        assert len(results.masks) == len(target_batch)

        # Convert ground truth masks to Masks objects
        gt_masks = masks_to_custom_masks(
            target_batch.masks,
            class_id=category_id,
        )

        # Create metrics calculator
        metrics_calculator = SegmentationMetrics(categories=[category_name])

        # Calculate metrics
        metrics_calculator(
            predictions=results.masks,
            ground_truths=gt_masks,
            mapping={category_id: category_name},
        )

        # Get metrics
        metrics = metrics_calculator.get_metrics()

        # Validate metrics structure
        assert isinstance(metrics, dict)
        assert "category" in metrics
        assert "iou" in metrics
        assert "f1score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "accuracy" in metrics
        assert "dice" in metrics
        assert "jaccard" in metrics
        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics

        # Check that metrics were calculated for the category
        if len(metrics["category"]) > 0:
            # Find metrics for our category (skip background if present)
            category_metrics = None
            for i, cat in enumerate(metrics["category"]):
                if cat == category_name:
                    category_metrics = i
                    break

            if category_metrics is not None:
                # Validate metric values are within expected ranges [0, 1]
                iou = metrics["iou"][category_metrics]
                f1score = metrics["f1score"][category_metrics]
                precision = metrics["precision"][category_metrics]
                recall = metrics["recall"][category_metrics]
                accuracy = metrics["accuracy"][category_metrics]
                dice = metrics["dice"][category_metrics]
                jaccard = metrics["jaccard"][category_metrics]

                # All metrics should be between 0 and 1
                assert 0.0 <= iou <= 1.0, f"IoU should be in [0, 1], got {iou}"
                assert 0.0 <= f1score <= 1.0, f"F1 score should be in [0, 1], got {f1score}"
                assert 0.0 <= precision <= 1.0, f"Precision should be in [0, 1], got {precision}"
                assert 0.0 <= recall <= 1.0, f"Recall should be in [0, 1], got {recall}"
                assert 0.0 <= accuracy <= 1.0, f"Accuracy should be in [0, 1], got {accuracy}"
                assert 0.0 <= dice <= 1.0, f"Dice should be in [0, 1], got {dice}"
                assert 0.0 <= jaccard <= 1.0, f"Jaccard should be in [0, 1], got {jaccard}"

                # IoU and Jaccard should be the same
                assert abs(iou - jaccard) < 1e-6, f"IoU and Jaccard should be equal, got {iou} vs {jaccard}"

                # Dice and F1 should be the same
                assert abs(dice - f1score) < 1e-6, f"Dice and F1 should be equal, got {dice} vs {f1score}"

                # Confusion matrix values should be non-negative integers
                tp = metrics["true_positives"][category_metrics]
                tn = metrics["true_negatives"][category_metrics]
                fp = metrics["false_positives"][category_metrics]
                fn = metrics["false_negatives"][category_metrics]

                assert tp >= 0, f"True positives should be non-negative, got {tp}"
                assert tn >= 0, f"True negatives should be non-negative, got {tn}"
                assert fp >= 0, f"False positives should be non-negative, got {fp}"
                assert fn >= 0, f"False negatives should be non-negative, got {fn}"

                # At least one of TP, TN, FP, FN should be positive (some prediction/ground truth exists)
                assert (tp + tn + fp + fn) > 0, "At least one confusion matrix value should be positive"
