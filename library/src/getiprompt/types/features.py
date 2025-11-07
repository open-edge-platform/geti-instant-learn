# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from getiprompt.types.data import Data


class Features(Data):
    """This class represents features of a single image.

    Each image has a global feature representation and
    can have multiple local feature representations based on the number of masks per class.
    """

    def __init__(self, global_features: torch.Tensor = None) -> None:
        """Initialize the features from a torch tensor.

        Args:
            global_features: The global features of the image.
        """
        self._global_features: torch.Tensor = global_features
        self._local_features: dict[int, list[torch.Tensor]] = {}

    def add_local_features(
        self,
        local_features: torch.Tensor,
        class_id: int,
    ) -> None:
        """Add features to the features object."""
        if class_id not in self._local_features:
            self._local_features[class_id] = []
        self._local_features[class_id].append(local_features)

    @property
    def global_features(self) -> torch.Tensor:
        """Get the global features."""
        return self._global_features

    @global_features.setter
    def global_features(self, global_features: torch.Tensor) -> None:
        self._global_features = global_features

    @property
    def local_features(self) -> dict[int, list[torch.Tensor]]:
        """Get the local features."""
        return self._local_features

    @local_features.setter
    def local_features(self, local_features: dict[int, list[torch.Tensor]]) -> None:
        self._local_features = local_features

    def get_local_features(self, class_idx: int) -> list[torch.Tensor]:
        """Get the local features for a specific class.

        Args:
            class_idx: The class index of the features to get.

        Returns:
            The local features for the specific class.
        """
        return self._local_features[class_idx]

    @property
    def global_embedding_dim(self) -> int:
        """Get the embedding dimension of the global features."""
        return self._global_features.shape[-1]

    @property
    def global_features_shape(self) -> torch.Size:
        """Get the shape of the global features."""
        return self._global_features.shape

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the features."""
        return self._global_features.shape[-1]
