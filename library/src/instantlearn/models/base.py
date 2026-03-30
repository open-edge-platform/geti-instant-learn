# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from instantlearn.components.postprocessing.base import PostProcessor, apply_postprocessing
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.data.base.batch import Batch, Collatable
    from instantlearn.data.base.sample import Sample


class Model(nn.Module):
    """This class is the base class for all models.

    Args:
        postprocessor: Optional post-processor (single or pipeline) applied
            after ``predict()`` to clean masks, resolve overlaps, etc.
            Use :class:`~instantlearn.components.postprocessing.PostProcessorPipeline`
            to chain multiple processors.
    """

    def __init__(self, postprocessor: PostProcessor | None = None) -> None:
        """Initialize the model with an optional post-processor."""
        super().__init__()
        self.postprocessor = postprocessor

    def apply_postprocessing(
        self,
        predictions: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Apply the configured post-processor to prediction dicts.

        If no post-processor is set, returns predictions unchanged.

        Args:
            predictions: List of prediction dicts from ``predict()``.

        Returns:
            Post-processed prediction dicts.
        """
        return apply_postprocessing(predictions, self.postprocessor)

    @abstractmethod
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn the context from reference samples.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """

    @abstractmethod
    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Use the learned context to infer object locations.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            A list of predictions, one per sample. Each prediction contains:
                - "pred_masks": torch.Tensor of shape [num_masks, H, W]
                - "pred_points": torch.Tensor of shape [num_points, 4] with [x, y, score, fg_label]
                - "pred_boxes": torch.Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score]
                - "pred_labels": torch.Tensor of shape [num_masks]
        """

    @abstractmethod
    def export(
        self,
        export_dir: str | Path,
        backend: str | Backend = Backend.ONNX,
        **kwargs: Any,  # noqa: ANN401
    ) -> Path:
        """This method exports the model to a given path.

        Args:
            export_dir: The directory to export the model to.
            backend: The backend to export the model to.
            **kwargs: Additional arguments to pass to the export method.

        Returns:
            The path to the exported model.
        """
