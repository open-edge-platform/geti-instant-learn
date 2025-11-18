# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base SAM Predictor interface for multi-backend support."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseSAMPredictor(ABC):
    """Common interface for SAM predictors across different backends.

    This abstract base class defines the interface that all SAM predictor
    implementations must follow, regardless of backend (PyTorch, OpenVINO, etc.).

    The interface provides backend-agnostic method names that clearly describe
    what operations they perform, without leaking implementation details.
    """

    def __init__(
        self,
        sam_model_name: "SAMModelName",  # type: ignore # noqa: F821
        device: str,
        model_path: Path | None = None,
        target_length: int = 1024,
    ) -> None:
        """Initialize SAM predictor.

        Args:
            sam_model_name: The SAM model architecture (e.g., SAM_HQ_TINY, SAM2_BASE)
            device: Device to run inference on:
                - PyTorch backend: "cuda", "cpu"
                - OpenVINO backend: "CPU", "GPU", "AUTO"
            model_path: Path to model weights/IR (interpretation varies by backend):
                - PyTorch: Path to .pth checkpoint file (optional, auto-downloads if None)
                - OpenVINO: Path to .xml IR file (optional, uses default path if None)
            target_length: Target length for the longest side of the image
        """
        self.sam_model_name = sam_model_name
        self._device = device
        self.model_path = model_path
        self.target_length = target_length
        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self) -> None:
        """Backend-specific initialization logic.

        This method should load the model weights, compile the model,
        and perform any other backend-specific setup.
        """

    @abstractmethod
    def set_image(
        self,
        image: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        """Set the image for prediction and compute embeddings if needed.

        This method prepares the predictor for inference on a specific image.
        For models with separate image encoding, this method may compute and
        cache image embeddings. For end-to-end models, this may just store
        the image for later use.

        Args:
            image: Preprocessed image tensor of shape (C, H, W)
            original_size: Original image size (H, W) before preprocessing
        """

    @abstractmethod
    def predict(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict segmentation masks given prompts.

        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) pixel format
            point_labels: Point labels [B, N] where:
                - 1 = foreground point
                - 0 = background point
                - -1 = padding (ignored)
            boxes: Box prompts [B, 4] in (x1, y1, x2, y2) pixel format
            mask_input: Low-resolution mask input [B, 1, 256, 256] from previous iteration
            multimask_output: If True, return 3 masks with different quality/coverage tradeoffs
            return_logits: If True, return logits instead of binary masks

        Returns:
            Tuple containing:
                - masks: Segmentation masks [B, C, H, W] where C is number of masks
                - iou_predictions: Model's predicted IoU scores [B, C] for each mask
                - low_res_logits: Low-resolution mask logits [B, C, 256, 256]
                  (can be used as mask_input for refinement)
        """

    @property
    def device(self) -> str:
        """Return the device this predictor runs on.

        Returns:
            Device string (e.g., "cuda", "cpu", "CPU", "GPU")
        """
        return self._device
