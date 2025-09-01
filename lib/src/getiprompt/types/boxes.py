# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from getiprompt.types.prompts import Prompt


class Boxes(Prompt):
    """This class represent box prompts for a single image.

    Since there can be varying amount of boxes, we use a list of tensors to store the boxes.
    The points have a shape of (nboxes, 6) where each row is [x1, y1, x2, y2, score, label].
    """

    def __init__(
        self,
        data: dict[int, list[torch.Tensor | np.ndarray]] | None = None,
    ) -> None:
        self._data: dict[int, list[torch.Tensor]] = data if data is not None else {}

    def add(self, data: torch.Tensor | np.ndarray, class_id: int = 0) -> None:
        """Adds data for a given class by extending the list."""
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if class_id in self._data:
            self._data[class_id].append(data)
        else:
            self._data[class_id] = [data]

    def get(self, class_id: int = 0) -> list[torch.Tensor]:
        """Get the data for a given class."""
        if class_id in self._data:
            return self._data[class_id]
        return []

    def only_foreground(self) -> dict[int, list[torch.Tensor]]:
        """Get a copy of data containing only the foreground points."""
        # filter out all points with label p[3] == 0
        return {k: [p[p[:, 5] > 0] for p in v] for k, v in self._data.items()}

    def only_background(self) -> dict[int, list[torch.Tensor]]:
        """Get a copy of data containing only the background points."""
        return {k: [p[p[:, 5] == 0] for p in v] for k, v in self._data.items()}
