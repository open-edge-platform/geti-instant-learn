from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms

from getiprompt.types import Features, Masks
from getiprompt.utils import MaybeToTensor


class LocalFeatureExtractor(nn.Module):
    """Extracts local features inside pooled masks and records Masks.

    Given batched feature tensors and samples with masks, pools pixel-space masks
    to the patch grid and extracts local features inside each mask.
    """

    def __init__(self, input_size: int, patch_size: int, device: str) -> None:
        """Initialize the local feature extractor.

        Args:
            input_size: The input image size.
            patch_size: The patch size of the encoder.
            device: The device to use.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.device = device
        self.transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([input_size, input_size]),
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(kernel_size=(patch_size, patch_size)),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    def forward(
        self,
        batched_features: torch.Tensor,
        batched_masks: torch.Tensor,
        batched_category_ids: torch.Tensor,
        batched_is_reference: torch.Tensor,
    ) -> tuple[list[Features], list[Masks]]:
        """Extract local features from batched features and samples.

        Args:
            batched_features: Batched feature tensor of shape
                (batch_size, num_patches, embedding_dim).
            samples: List of Sample objects, each containing masks and category_ids.

        Returns:
            tuple[list[Features], list[Masks]]: Tuple of Features objects (with local features added)
                and Masks objects, one per sample.
        """
        # Split batched tensor into individual tensors and wrap in Features objects
        features_list = [Features(global_features=emb) for emb in batched_features.unbind(0)]

        resized_masks_per_image = []
        for embedding, masks_tensor, category_ids, is_reference_list in zip(
            features_list,
            batched_masks,
            batched_category_ids,
            batched_is_reference,
            strict=False,
        ):
            resized_masks = Masks()
            for category_id, mask, is_reference in zip(category_ids, masks_tensor, is_reference_list, strict=True):
                category_id = category_id.item()
                if is_reference:
                    pooled_mask = self.transform(mask)
                    resized_masks.add(mask=pooled_mask, class_id=category_id)
                    keep = pooled_mask.flatten().bool()
                    local_features = embedding.global_features[keep]
                    embedding.add_local_features(local_features=local_features, class_id=category_id)
            resized_masks_per_image.append(resized_masks)

        return features_list, resized_masks_per_image
