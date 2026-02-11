# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.utils.constants import Backend


class Model(nn.Module):
    """This class is the base class for all models."""

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
        **kwargs,
    ) -> Path:
        """This method exports the model to a given path.

        Args:
            export_dir: The directory to export the model to.
            backend: The backend to export the model to.
            **kwargs: Additional arguments to pass to the export method.

        Returns:
            The path to the exported model.
        """
