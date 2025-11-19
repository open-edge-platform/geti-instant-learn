# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend implementation for ImageEncoder."""

from logging import getLogger
from pathlib import Path

import openvino as ov
import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from transformers import AutoImageProcessor
from transformers import AutoConfig
logger = getLogger("Geti Prompt")


class OpenVINOImageEncoder(nn.Module):
    """OpenVINO backend for DINO image encoder.

    This implementation uses OpenVINO Runtime for efficient inference on
    Intel hardware (CPU, GPU, VPU). It expects an exported OpenVINO IR model
    created by PyTorchImageEncoder.export().

    Examples:
        >>> from getiprompt.components.encoders import OpenVINOImageEncoder
        >>> from pathlib import Path
        >>> import torch
        >>>
        >>> # First export from PyTorch (see PyTorchImageEncoder.export)
        >>> encoder = OpenVINOImageEncoder(
        ...     model_path=Path("./exported/dinov2_large"),
        ...     input_size=518
        ... )
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1369, 1024])
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "CPU",
        input_size: int = 518,
    ) -> None:
        """Initialize the OpenVINO encoder.

        Args:
            model_path: Path to the exported OpenVINO IR directory
                (should contain .xml and .bin files).
            device: OpenVINO device to use (CPU, GPU, AUTO).
            input_size: The input size to use for preprocessing.

        Raises:
            FileNotFoundError: If the model path doesn't exist.
            ValueError: If model_id is not provided and can't be loaded from config.
        """
        super().__init__()

        model_path = Path(model_path)
        if not model_path.exists():
            msg = (
                f"OpenVINO model not found at {model_path}. "
                f"Please export the model first using PyTorchImageEncoder.export(). "
                f"Example:\n"
                f"  from getiprompt.components.encoders import PyTorchImageEncoder\n"
                f"  encoder = PyTorchImageEncoder(model_id='dinov2_large')\n"
                f"  encoder.export(Path('{model_path}'))"
            )
            raise FileNotFoundError(msg)

        self.model_path = model_path
        self.input_size = input_size
        self.device = device

        # Load config to get model metadata
        try:
            config = AutoConfig.from_pretrained(model_path.parent)
            self.patch_size = config.patch_size
            self.feature_size = input_size // self.patch_size
            self.ignore_token_length = 1 + getattr(config, "num_register_tokens", 0)            
        except Exception as e:
            msg = (
                f"Could not load config from {model_path} and model_id not provided. "
                f"Please provide model_id explicitly."
            )
            raise ValueError(msg) from e

        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(
            model_path.parent,
            size={"height": input_size, "width": input_size},
            do_center_crop=False,
            use_fast=True,
        )

        # Load and compile OpenVINO model
        logger.info(f"Loading OpenVINO DINO encoder from {model_path}")
        self.core = ov.Core()

        # Look for model.xml (created by export)
        ov_model = self.core.read_model(model_path)
        ov_device = self._map_device_name(device)
        self.compiled_model = self.core.compile_model(ov_model, ov_device)

        # Store input/output names
        self.input_name = self.compiled_model.input(0).any_name
        self.output_name = self.compiled_model.output(0).any_name

    def _map_device_name(self, device: str) -> str:
        """Map PyTorch device names to OpenVINO device names.

        Args:
            device: Device name in PyTorch style

        Returns:
            Device name in OpenVINO style
        """
        device_map = {
            "cuda": "GPU",
            "cpu": "CPU",
            "GPU": "GPU",
            "CPU": "CPU",
            "AUTO": "AUTO",
        }
        return device_map.get(device.upper() if device else "CPU", "CPU")

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into normalized patch embeddings using OpenVINO.

        Args:
            images(list[tv_tensors.Image]): A list of images.

        Returns:
            torch.Tensor: Normalized patch-grid feature tensor of shape
                (batch_size, num_patches, embedding_dim).
        """
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].cpu().numpy()

        # Run OpenVINO inference
        outputs = self.compiled_model({self.input_name: pixel_values})
        last_hidden_state = outputs[self.output_name]

        # Convert to torch and post-process
        last_hidden_state = torch.from_numpy(last_hidden_state)
        # Remove CLS token and register tokens
        features = last_hidden_state[:, self.ignore_token_length :, :]

        return functional.normalize(features, p=2, dim=-1)



