# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from abc import abstractmethod
from logging import getLogger

from torch import nn

from getiprompt.data.base.batch import Batch
from getiprompt.types import Results

logger = getLogger("Geti Prompt")


class Model(nn.Module):
    """This class is the base class for all models."""

    @abstractmethod
    def learn(self, reference_batch: Batch) -> None:
        """This method learns the context.

        Args:
            reference_batch(Batch): A batch of reference samples to learn from.
        """

    @abstractmethod
    def infer(self, target_batch: Batch) -> Results:
        """This method uses the learned context to infer object locations.

        Args:
            target_batch(Batch): A batch of target samples to infer.

        Returns:
            Results: The results of the inference.
        """
