# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model."""

import torch
import torchvision
from torch import nn
from torchvision.transforms.v2.functional import to_dtype, to_image

from getiprompt.types import Priors
from getiprompt.utils.constants import IMAGENET_TEMPLATES


class DinoTextEncoder(nn.Module):
    """DINOv3 text encoder for zero-shot classification.

    Args:
        pretrained: Whether to use a pretrained model.
        image_size: The size of the input image.
        repo_id: The repo id of the model.
        model_id: The model id of the model.
        precision: The precision to use for the model.
        device: The device to use for the model.
        mean: The mean to use for image normalization.
        std: The standard deviation to use for image normalization.
    """

    def __init__(
        self,
        pretrained: bool = True,
        image_size: int = 512,
        repo_id: str = "facebookresearch/dinov3",
        model_id: str = "dinov3_vitl16_dinotxt_tet1280d20h24l",
        precision: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mean: tuple[float] = (123.675, 116.28, 103.53),
        std: tuple[float] = (58.395, 57.12, 57.375),
    ) -> None:
        super().__init__()
        model, tokenizer = torch.hub.load(
            repo_id,
            model_id,
            pretrained=pretrained,
        )
        self.tokenizer = tokenizer
        self.device = device
        self.precision = precision
        self.model = model.to(dtype=self.precision)
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.v2.Resize(image_size),
            torchvision.transforms.v2.Normalize(mean=mean, std=std),
            torchvision.transforms.v2.ToDtype(dtype=self.precision),
        ])

    @torch.no_grad()
    def encode_text(
        self,
        reference_prior: Priors,
        prompt_template: list[str] = IMAGENET_TEMPLATES,
    ) -> torch.Tensor:
        """Encode the class text prompt to text embedding.

        Args:
            reference_prior: The prior to encode.
            prompt_template: The prompt template to use for the model.

        Returns:
            The text embedding.

        Examples:
            >>> from getiprompt.models.dinotxt import DinoTextEncoder
            >>> from getiprompt.types import Priors
            >>> encoder = DinoTextEncoder()
            >>> prior = Priors(text={0: "cat", 1: "dog"})
            >>> text_embedding = encoder.encode_text(prior)
            >>> text_embedding.shape
            torch.Size([2, 4])
        """
        zero_shot_weights = []
        for label_name in reference_prior.text.values():
            texts = [template.format(label_name) for template in prompt_template]
            texts = self.tokenizer.tokenize(texts)
            if self.device == "cuda":
                texts = texts.cuda()
            class_embeddings = self.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zero_shot_weights.append(class_embedding)
        return torch.stack(zero_shot_weights, dim=1)

    @torch.no_grad()
    def encode_image(
        self,
        target_images: list,
    ) -> torch.Tensor:
        """Encode the reference images to image embedding."""
        images = [self.transforms(to_dtype(to_image(image), dtype=self.precision)) for image in target_images]
        images = torch.stack(images, dim=0)
        if self.device == "cuda":
            images = images.cuda()
        with torch.autocast(device_type=self.device, dtype=self.precision):
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.to(self.precision)
