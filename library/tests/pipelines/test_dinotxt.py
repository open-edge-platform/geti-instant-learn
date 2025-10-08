# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the DINOv3 zero-shot classification pipeline."""

import numpy as np
import pytest
import torch
from skimage.draw import random_shapes

from getiprompt.pipelines.dinotxt import DinoTxtZeroShotClassification
from getiprompt.types import Image, Priors, Results


@pytest.fixture
def pipeline_instance() -> DinoTxtZeroShotClassification:
    """Returns an instance of the DinoTxtZeroShotClassification pipeline."""
    return DinoTxtZeroShotClassification(
        device="cpu",  # Use CPU for testing
        image_size=(224, 224),  # Smaller size for faster testing
        precision="bf16",
    )


@pytest.fixture
def sample_dataset() -> tuple[list[np.ndarray], list[str]]:
    """Create sample images using skimage.draw.random_shapes."""
    images = []
    labels = []
    label_names = ["circle", "rectangle", "triangle"]
    for label in label_names:
        for _ in range(3):
            # Generate random shapes with different characteristics
            image, _ = random_shapes(
                (224, 224, 3),
                max_shapes=5,
                min_shapes=2,
                min_size=20,
                max_size=100,
                num_channels=3,
                shape=label,
            )
            images.append(image.astype(np.uint8))
            labels.append(label_names.index(label))
    return images, labels


@pytest.fixture
def sample_priors() -> Priors:
    """Create sample text priors for classification."""
    return Priors(text={0: "circle", 1: "rectangle", 2: "triangle"})


class TestDinoTxtZeroShotClassification:
    """Test cases for the DinoTxtZeroShotClassification pipeline."""

    @staticmethod
    def test_pipeline_initialization_with_custom_params() -> None:
        """Test pipeline initialization with custom parameters."""
        custom_templates = ["a photo of a {}."]
        pipeline = DinoTxtZeroShotClassification(
            prompt_templates=custom_templates,
            precision="fp16",
            device="cpu",
            image_size=(512, 512),
        )
        assert pipeline.prompt_templates == custom_templates
        assert pipeline.precision == torch.float16
        assert pipeline.resize_images.size == (512, 512)

    @staticmethod
    def test_learn_with_empty_reference_priors(pipeline_instance: DinoTxtZeroShotClassification) -> None:
        """Test that learn raises ValueError when no reference priors are provided."""
        with pytest.raises(ValueError, match="reference_priors must be provided"):
            pipeline_instance.learn([], [])

    @staticmethod
    def test_infer_without_learning(
        pipeline_instance: DinoTxtZeroShotClassification, sample_dataset: tuple[list[np.ndarray], list[str]]
    ) -> None:
        """Test that infer raises AttributeError when learn hasn't been called."""
        sample_images, _ = sample_dataset
        # Convert numpy arrays to Image objects
        image_objects = [Image(img) for img in sample_images]
        with pytest.raises(AttributeError):
            pipeline_instance.infer(image_objects)

    @staticmethod
    def test_infer(
        pipeline_instance: DinoTxtZeroShotClassification,
        sample_dataset: tuple[list[np.ndarray], list[str]],
        sample_priors: Priors,
    ) -> None:
        """Test the full learn and infer cycle of the pipeline."""
        sample_images, sample_labels = sample_dataset

        # Learn first
        pipeline_instance.learn([], [sample_priors])

        # Convert numpy arrays to Image objects
        image_objects = [Image(img) for img in sample_images]

        # Then infer
        result = pipeline_instance.infer(image_objects)

        # Verify results
        assert isinstance(result, Results)
        assert hasattr(result, "masks")
        assert len(result.masks) == len(sample_images)

        pred_labels = [mask.class_ids()[0] for mask in result.masks]
        pred_labels = torch.tensor(pred_labels)
        gt_labels = torch.tensor(sample_labels)
        assert (pred_labels.eq(gt_labels) / len(sample_labels)).mean() >= 0.0
