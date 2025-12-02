# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend implementation for ImageEncoder."""

from logging import getLogger
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype

logger = getLogger("Geti Prompt")

# Default normalization values for DINO models
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class OpenVINOImageEncoder(nn.Module):
    """OpenVINO backend for DINO image encoder.

    This implementation uses OpenVINO Runtime for efficient inference on
    Intel hardware (CPU, GPU, VPU). It expects an exported OpenVINO IR model
    created by TimmImageEncoder.export() or HuggingFaceImageEncoder.export().

    Examples:
        >>> from getiprompt.components.encoders import OpenVINOImageEncoder
        >>> from pathlib import Path
        >>> import torch
        >>>
        >>> # First export from Timm (see TimmImageEncoder.export)
        >>> encoder = OpenVINOImageEncoder(
        ...     model_path=Path("./exported/image_encoder.xml"),
        ... )
        >>> sample_image = torch.zeros((3, 512, 512))
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1, 1024, 1024])
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "CPU",
    ) -> None:
        """Initialize the OpenVINO encoder.

        Args:
            model_path: Path to the exported OpenVINO IR model (.xml file).
            device: OpenVINO device to use (CPU, GPU, AUTO).

        Raises:
            FileNotFoundError: If the model path doesn't exist.
            ValueError: If required model metadata is not found in the IR.
        """
        super().__init__()

        model_path = Path(model_path)
        if not model_path.exists():
            msg = (
                f"OpenVINO model not found at {model_path}. "
                f"Please export the model first using TimmImageEncoder.export(). "
                f"Example:\n"
                f"  from getiprompt.components.encoders import TimmImageEncoder\n"
                f"  encoder = TimmImageEncoder(model_id='dinov3_large')\n"
                f"  encoder.export(Path('./exported'), backend='openvino')"
            )
            raise FileNotFoundError(msg)

        self.model_path = model_path
        self.device = device

        # Load OpenVINO model
        logger.info(f"Loading OpenVINO DINO encoder from {model_path}")
        self.core = ov.Core()
        ov_model = self.core.read_model(model_path)

        # Load model configuration from runtime info
        try:
            self.patch_size = ov_model.get_rt_info(["model_info", "patch_size"]).astype(int)
            self.feature_size = ov_model.get_rt_info(["model_info", "feature_size"]).astype(int)
            self.ignore_token_length = ov_model.get_rt_info(["model_info", "ignore_token_length"]).astype(int)
            self.input_size = ov_model.get_rt_info(["model_info", "input_size"]).astype(int)
        except RuntimeError as e:
            msg = (
                f"Could not load model configuration from {model_path}. "
                f"The model may have been exported with an older version. "
                f"Please re-export the model using TimmImageEncoder.export()."
            )
            raise ValueError(msg) from e

        # Create processor with default ImageNet normalization
        self.processor = Compose([
            ToDtype(dtype=torch.float32, scale=True),
            Resize(size=(self.input_size, self.input_size)),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

        # Compile model
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
        images_tensor = torch.stack([self.processor(image) for image in images])
        pixel_values = images_tensor.numpy()

        # Run OpenVINO inference
        outputs = self.compiled_model({self.input_name: pixel_values})
        features = outputs[self.output_name]

        return torch.from_numpy(np.ascontiguousarray(features))
