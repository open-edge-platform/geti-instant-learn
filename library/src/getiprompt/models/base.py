# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from abc import abstractmethod

import torch
from torch import nn

from getiprompt.data.base.batch import Batch


class Model(nn.Module):
    """This class is the base class for all models."""

    @abstractmethod
    def learn(self, reference_batch: Batch) -> None:
        """This method learns the context.

        Args:
            reference_batch(Batch): A batch of reference samples to learn from.
        """

    @abstractmethod
    def infer(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """This method uses the learned context to infer object locations.

        Args:
            target_batch(Batch): A batch of target samples to infer.

        Returns:
            predictions(list[dict[str, torch.Tensor]]): A list of predictions.
            Each prediction contains:
                "pred_masks": torch.Tensor of shape [num_masks, H, W]
                "pred_points": torch.Tensor of shape [num_points, 4]
                "pred_boxes": torch.Tensor of shape [num_boxes, 6]
                "pred_labels": torch.Tensor of shape [num_masks]
        """
