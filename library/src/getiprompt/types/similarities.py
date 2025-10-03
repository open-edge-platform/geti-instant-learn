# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from getiprompt.types.data import Data


class Similarities(Data):
    """This class represents similarities of a single image.

    Each image has a similarity representation and can have multiple similarity representations
    based on the number of masks per class. Similarities are stored as stacked tensors per class.
    """

    def __init__(self) -> None:
        self._data: dict[int, torch.Tensor] = {}

    def add(
        self,
        similarities: torch.Tensor,
        class_id: int,
    ) -> None:
        """Add similarities to the class.

        Args:
            similarities: The similarities to add.
            class_id: The class ID to add similarities for.
        """
        if class_id not in self._data:
            self._data[class_id] = similarities
        else:
            # Concatenate along first dimension since similarities are (1, encoder_dim, encoder_dim)
            self._data[class_id] = torch.cat(
                [self._data[class_id], similarities],
                dim=0,
            )

    def get(self, class_id: int) -> torch.Tensor | None:
        """Get similarities for a class.

        Args:
            class_id: The class ID to get similarities for.

        Returns:
            The similarities for the class.
        """
        return self._data.get(class_id)
