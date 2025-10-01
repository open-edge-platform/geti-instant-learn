# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from getiprompt.types.prompts import Prompt


class Masks(Prompt):
    """This class represents all class masks for a single image.

    Masks are stored as a dictionary of torch tensors, where the key is the class id.
    Masks per class are stored as a 3D tensor with shape n_masks x H x W and boolean values.
    """

    def add(self, mask: torch.Tensor | np.ndarray, class_id: int = 0) -> None:
        """Add a mask for a given class."""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.dtype != torch.bool:
            mask = (mask > 0).bool()

        if mask.ndim == 3 and mask.shape[0] != 1:  # HWC format
            max_channel = 0 if mask.shape[-1] == 1 else torch.argmax(mask.sum(dim=(0, 1)))
            mask = mask[:, :, max_channel].unsqueeze(0)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if class_id not in self._data:
            self._data[class_id] = mask
        else:
            # Concatenate along existing batch dimension (dim=0) without adding new dimension
            self._data[class_id] = torch.cat([self._data[class_id], mask], dim=0)

    def resize(self, size: tuple[int, int] | int | None = None) -> "Masks":
        """Return a resized copy of the masks.

        Args:
            size: The size to resize the masks to.
                  If a tuple (width, height) is provided, the masks will be resized to the given width and height.
                  If an integer is provided, the largest side of the masks will be resized to this value,
                  maintaining aspect ratio.
                  If None is provided, the original masks are returned.

        Returns:
            A resized copy of the masks.
        """
        if size is None:
            return self

        if not self._data:
            return Masks()

        original_h, original_w = self.shape[1], self.shape[2]

        if isinstance(size, tuple):
            user_w, user_h = size
            target_h = user_h
            target_w = user_w
        else:
            target_largest_dim = size
            if original_h == 0 or original_w == 0:
                target_h = target_largest_dim
                target_w = target_largest_dim
            elif original_h > original_w:
                target_h = target_largest_dim
                target_w = round(original_w * (target_largest_dim / original_h)) if original_h > 0 else 0
            else:
                target_w = target_largest_dim
                target_h = round(original_h * (target_largest_dim / original_w)) if original_w > 0 else 0

        if (target_h, target_w) == (original_h, original_w):
            return self

        resized_masks = Masks()
        for class_id, mask_tensor_data in self._data.items():
            resized_masks.add(
                torch.nn.functional.interpolate(
                    mask_tensor_data.unsqueeze(1).float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )
                .squeeze(1)
                .bool(),
                class_id,
            )
        return resized_masks

    def resize_inplace(self, size: tuple[int, int] | int | None = None) -> None:
        """Resize the masks in place.

        Args:
            size: The size to resize the masks to. If a tuple is provided, the masks will be resized to the given width
              and height. If an integer is provided, the masks will be resized to the given size, maintaining aspect
                ratio. If None is provided, the masks will not be resized.

        Returns:
            The resized masks.
        """
        self._data = self.resize(size).data

    def to_numpy(self, class_id: int = 0) -> np.ndarray:
        """Convert the masks to a numpy array with shape HxWxC in uint8 format."""
        return self._data[class_id].numpy().astype(np.uint8)

    def class_ids(self) -> list[int]:
        """Return the class ids."""
        return list(self._data.keys())

    @property
    def mask_shape(self) -> tuple[int, int]:
        """Get the shape of a mask."""
        return self._data[0].shape[1:]
