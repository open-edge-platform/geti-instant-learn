# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prompt type for Geti Prompt."""

import numpy as np
import torch

from getiprompt.types.data import Data


class Prompt(Data):
    """This class represents a prompt for a single image.

    This can be points, boxes or polygons. It serves as a base class for masks.
    """

    def __init__(self, data: dict[int, torch.Tensor | np.ndarray] | None = None) -> None:
        """Initializes the Prompt."""
        self._data: dict[int, torch.Tensor | np.ndarray] = data if data is not None else {}

    def add(self, data: torch.Tensor | np.ndarray, class_id: int = 0) -> None:
        """Add data for a given class."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        # add data to existing data
        if class_id in self._data:
            self._data[class_id] = torch.cat([self._data[class_id], data], dim=0)
        else:
            self._data[class_id] = data

    def get(self, class_id: int = 0) -> torch.Tensor | np.ndarray:
        """Get the data for a given class."""
        if class_id in self._data:
            return self._data[class_id]
        # Return an empty tensor with appropriate dimensions
        # For Masks, this would be an empty tensor with shape (0, H, W)
        if len(self._data) > 0:
            # Get shape from first available class
            first_key = next(iter(self._data))
            shape = self._data[first_key].shape
            # Return empty tensor with same dimensions except first one is 0
            return torch.empty((0, *shape[1:]), dtype=self._data[first_key].dtype)
        # No data at all, return completely empty tensor
        return torch.empty((0, 0, 0))

    @property
    def data(self) -> dict[int, torch.Tensor | np.ndarray]:
        """Get the data."""
        return self._data

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get the shape of the data."""
        return self.get().shape

    @property
    def is_empty(self) -> bool:
        """Check if the prompts are empty."""
        return not self._data

    def class_ids(self) -> list[int]:
        """Get the number of classes."""
        return list(self._data.keys())
