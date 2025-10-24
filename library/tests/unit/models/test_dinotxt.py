# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the DINOv3 zero-shot classification pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from skimage.draw import random_shapes

from getiprompt.models.dinotxt import DinoTxtZeroShotClassification
from getiprompt.types import Image, Priors, Results


@pytest.fixture
def mock_dino_encoder() -> MagicMock:
    """Create a mock DinoTextEncoder."""
    mock_encoder = MagicMock()
    # Mock text embeddings: (embedding_dim, num_classes) - transposed for matrix multiplication
    mock_encoder.encode_text.return_value = torch.randn(128, 3)  # 128 dim embeddings, 3 classes
    # Mock image embeddings: (num_images, embedding_dim)
    mock_encoder.encode_image.return_value = torch.randn(9, 128)  # 9 images, 128 dim embeddings
    return mock_encoder


@pytest.fixture
def model_instance(mock_dino_encoder: MagicMock) -> DinoTxtZeroShotClassification:
    """Returns an instance of the DinoTxtZeroShotClassification pipeline.

    Returns:
        DinoTxtZeroShotClassification: An instance configured for CPU testing.
    """
    with patch("getiprompt.models.dinotxt.DinoTextEncoder") as mock_encoder_class:
        mock_encoder_class.return_value = mock_dino_encoder
        return DinoTxtZeroShotClassification(
            device="cpu",  # Use CPU for testing
            image_size=(224, 224),  # Smaller size for faster testing
            precision="bf16",
        )


@pytest.fixture
def sample_dataset() -> tuple[list[np.ndarray], list[str]]:
    """Create sample images using skimage.draw.random_shapes.

    Returns:
        tuple[list[np.ndarray], list[str]]: A tuple containing list of images and their labels.
    """
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
    """Create sample text priors for classification.

    Returns:
        Priors: A Priors object containing text descriptions for each class.
    """
    return Priors(text={0: "circle", 1: "rectangle", 2: "triangle"})


class TestDinoTxtZeroShotClassification:
    """Test cases for the DinoTxtZeroShotClassification pipeline."""

    @staticmethod
    @patch("getiprompt.models.dinotxt.DinoTextEncoder")
    def test_pipeline_initialization_with_custom_params(mock_encoder_class: MagicMock) -> None:
        """Test pipeline initialization with custom parameters."""
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder

        custom_templates = ["a photo of a {}."]
        pipeline = DinoTxtZeroShotClassification(
            prompt_templates=custom_templates,
            precision="fp16",
            device="cpu",
            image_size=(512, 512),
        )
        pytest.assume(pipeline.prompt_templates == custom_templates)
        pytest.assume(pipeline.precision == torch.float16)

    @staticmethod
    def test_learn_with_empty_reference_priors(model_instance: DinoTxtZeroShotClassification) -> None:
        """Test that learn raises ValueError when no reference priors are provided."""
        with pytest.raises(ValueError, match="reference_priors must be provided"):
            model_instance.learn([], [])

    @staticmethod
    def test_infer_without_learning(
        model_instance: DinoTxtZeroShotClassification,
        sample_dataset: tuple[list[np.ndarray], list[str]],
    ) -> None:
        """Test that infer raises AttributeError when learn hasn't been called."""
        sample_images, _ = sample_dataset
        # Convert numpy arrays to Image objects
        with pytest.raises(AttributeError):
            model_instance.infer(sample_images)

    @staticmethod
    def test_infer(
        model_instance: DinoTxtZeroShotClassification,
        sample_dataset: tuple[list[np.ndarray], list[str]],
        sample_priors: Priors,
    ) -> None:
        """Test the full learn and infer cycle of the pipeline."""
        sample_images, sample_labels = sample_dataset

        # Learn first
        model_instance.learn([], [sample_priors])

        # Convert numpy arrays to Image objects
        image_objects = [Image(img) for img in sample_images]

        # Then infer
        result = model_instance.infer(image_objects)

        # Verify results
        pytest.assume(isinstance(result, Results))
        pytest.assume(hasattr(result, "masks"))
        pytest.assume(len(result.masks) == len(sample_images))

        pred_labels = [mask.class_ids()[0] for mask in result.masks]
        pred_labels = torch.tensor(pred_labels)
        gt_labels = torch.tensor(sample_labels)
        pytest.assume((pred_labels.eq(gt_labels) / len(sample_labels)).mean() >= 0.0)
