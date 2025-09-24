from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from getiprompt.models.model_optimizer import optimize_model
from getiprompt.utils.utils import MaybeToTensor, precision_to_torch_dtype


class DinoV2(nn.Module):
    """DINOv2 model for extracting features from images.

    Examples:
        >>> from getiprompt.models.dinov2 import DinoV2
        >>> model = DinoV2()
        >>> model.forward(torch.randn(1, 3, 224, 224))
        torch.Size([1, 1369, 1024])
    """

    class Size(Enum):
        """The size of the DINOv2 model."""

        SMALL = "small"
        BASE = "base"
        LARGE = "large"
        GIANT = "giant"

        @classmethod
        def from_str(cls, size: str) -> "DinoV2.Size":
            """Convert a string to a DINOv2 size."""
            return cls(size.lower())

    def __init__(
        self,
        size: "DinoV2.Size | str" = Size.LARGE,
        use_registers: bool = True,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
    ) -> None:
        """Initialize the DINOv2 model.

        Args:
            size: The size of the DINOv2 model.
            use_registers: Whether to use registers. This reduces noise and improves performance. See https://arxiv.org/abs/2309.16588
            device: The device to use for the model.
            precision: The precision to use for the model.
            compile_models: Whether to compile the model.
            benchmark_inference_speed: Whether to benchmark the inference speed.
        """
        super().__init__()
        if isinstance(size, str):
            size = DinoV2.Size.from_str(size)
        model_id = f"facebook/dinov2{'-with-registers' if use_registers else ''}-{size.value}"
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.input_size = model.config.image_size
        self.patch_size = model.config.patch_size
        self.feature_size = self.input_size // self.patch_size
        self.ignore_token_length = 5 if use_registers else 1

        self.precision = precision_to_torch_dtype(precision)
        # TODO move optimization code outside of this constructor.
        self.model = optimize_model(
            model=model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        ).eval()

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            size={"height": self.input_size, "width": self.input_size},
            do_center_crop=False,
            use_fast=True,  # uses Rust based image processor
        )
        self.mask_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([self.input_size, self.input_size]),
            # MinPool to make sure we do not use background features
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(
                kernel_size=(self.patch_size, self.patch_size),
            ),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images using DINOv2.

        Args:
            x: The input images.

        Returns:
            The normalized features.
        """
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        last_hidden_state = self.model(**inputs).last_hidden_state
        features = last_hidden_state[:, self.ignore_token_length :, :]  # Remove CLS token (and register tokens if used)
        return F.normalize(features, p=2, dim=-1)
