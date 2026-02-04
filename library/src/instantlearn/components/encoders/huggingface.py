# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace backend implementation for ImageEncoder."""

from logging import getLogger

import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from transformers import AutoImageProcessor, AutoModel

from instantlearn.utils import HUGGINGFACE_AVAILABLE_IMAGE_ENCODERS as AVAILABLE_IMAGE_ENCODERS
from instantlearn.utils import precision_to_torch_dtype

logger = getLogger("Geti Instant Learn")


class HuggingFaceImageEncoder(nn.Module):
    """HuggingFace backend for DINO image encoder.

    This encoder uses a model from HuggingFace to encode images into
    normalized patch embeddings.

    Examples:
        >>> from instantlearn.components.encoders import HuggingFaceImageEncoder
        >>> from torchvision import tv_tensors
        >>> import torch
        >>>
        >>> # Create a sample image
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> encoder = HuggingFaceImageEncoder(model_id="dinov2_large")
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1369, 1024])
    """

    def __init__(
        self,
        model_id: str = "dinov3_large",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        input_size: int = 518,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_id: The model id to use.
            device: The device to use.
            precision: The precision to use.
            compile_models: Whether to compile the models.
            input_size: The input size to use.

        Raises:
            ValueError: If the model ID is invalid.
        """
        from instantlearn.utils.optimization import optimize_model

        super().__init__()

        if model_id not in AVAILABLE_IMAGE_ENCODERS:
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {list(AVAILABLE_IMAGE_ENCODERS.keys())}"
            raise ValueError(msg)

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        hf_model_id, revision = AVAILABLE_IMAGE_ENCODERS[model_id]

        msg = f"Loading DINO model {hf_model_id} with revision {revision}"
        logger.info(msg)
        self.model, self.processor = self._load_hf_model(hf_model_id, revision, input_size)
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.feature_size = self.input_size // self.patch_size
        # Ignore CLS token and register tokens
        self.ignore_token_length = 1 + self.model.config.num_register_tokens

        self.precision = precision_to_torch_dtype(precision)

        self.model = optimize_model(
            model=self.model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
        ).eval()

    @staticmethod
    def _load_hf_model(model_id: str, revision: str, input_size: int) -> tuple[nn.Module, AutoImageProcessor]:
        """Load DINO model from HuggingFace with error handling.

        Meta requires huggingface users to access weights by first requesting access on the HuggingFace website.
        This function will raise an error if the user does not have access to the weights.

        Args:
            model_id: The model id of the model.
            revision: Specific revision (commit SHA, tag, or branch) to pin
            input_size: The size of the input image.

        Returns:
            The model and processor.

        Raises:
            ValueError: If the user does not have access to the weights of the model.
            OSError: If the model is not found.
        """
        err_msg = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            f"1. Request access on the HuggingFace website: https://huggingface.co/{model_id}\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token\n"
        )
        try:
            # B615 - revision is pinned in AVAILABLE_IMAGE_ENCODERS
            model = AutoModel.from_pretrained(model_id, revision=revision)  # nosec: B615
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                size={"height": input_size, "width": input_size},
                do_center_crop=False,
                use_fast=True,  # uses Rust based image processor
            )  # nosec: B615
        except OSError as e:
            # Check if this is specifically a HuggingFace gated repo access error
            if "gated repo" in str(e).lower():
                raise ValueError(err_msg) from None
            raise
        return model, processor

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into normalized patch embeddings.

        Args:
            images(list[tv_tensors.Image]): A list of images.

        Returns:
            torch.Tensor: Normalized patch-grid feature tensor of shape
                (batch_size, num_patches, embedding_dim).
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        last_hidden_state = self.model(**inputs).last_hidden_state
        features = last_hidden_state[:, self.ignore_token_length :, :]  # Remove CLS token (and register tokens if used)
        return functional.normalize(features, p=2, dim=-1)
