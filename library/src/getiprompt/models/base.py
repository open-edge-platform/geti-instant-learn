# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from abc import abstractmethod
from logging import getLogger

from torch import nn
from torchvision import tv_tensors

from getiprompt.types import Priors, Results

logger = getLogger("Geti Prompt")


class Model(nn.Module):
    """This class is the base class for all models."""

    @abstractmethod
    def learn(self, reference_images: list[tv_tensors.Image], reference_priors: list[Priors]) -> None:
        """This method learns the context.

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.
        """

    @abstractmethod
    def infer(self, target_images: list[tv_tensors.Image]) -> Results:
        """This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None
        """
