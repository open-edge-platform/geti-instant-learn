# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend implementation for ImageEncoder."""

from logging import getLogger
from pathlib import Path

import openvino
import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from transformers import AutoImageProcessor, AutoModel

from getiprompt.utils import precision_to_torch_dtype

logger = getLogger("Geti Prompt")

AVAILABLE_IMAGE_ENCODERS = {
    "dinov2_small": "facebook/dinov2-with-registers-small",
    "dinov2_base": "facebook/dinov2-with-registers-base",
    "dinov2_large": "facebook/dinov2-with-registers-large",
    "dinov2_giant": "facebook/dinov2-with-registers-giant",
    "dinov3_small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


class PyTorchImageEncoder(nn.Module):
    """PyTorch backend for DINO image encoder.

    This encoder uses a model from HuggingFace to encode images into
    normalized patch embeddings.

    Examples:
        >>> from getiprompt.components.encoders import PyTorchImageEncoder
        >>> from torchvision import tv_tensors
        >>> import torch
        >>>
        >>> # Create a sample image
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> encoder = PyTorchImageEncoder(model_id="dinov2_large")
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
        from getiprompt.utils.optimization import optimize_model

        super().__init__()

        if model_id not in AVAILABLE_IMAGE_ENCODERS:
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {list(AVAILABLE_IMAGE_ENCODERS.keys())}"
            raise ValueError(msg)

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        msg = f"Loading DINO model {model_id}"
        logger.info(msg)
        self.model, self.processor = self._load_hf_model(AVAILABLE_IMAGE_ENCODERS[model_id], input_size)
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
    def _load_hf_model(model_id: str, input_size: int) -> tuple[nn.Module, AutoImageProcessor]:
        """Load DINO model from HuggingFace with error handling.

        Meta requires huggingface users to access weights by first requesting access on the HuggingFace website.
        This function will raise an error if the user does not have access to the weights.

        Args:
            model_id: The model id of the model.
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
            model = AutoModel.from_pretrained(model_id)
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                size={"height": input_size, "width": input_size},
                do_center_crop=False,
                use_fast=True,  # uses Rust based image processor
            )
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

    def export(self, output_path: Path) -> Path:
        """Export this PyTorch encoder to OpenVINO IR format.

        This uses direct ONNX + OpenVINO conversion to export the DINO model.
        The exported model can then be loaded using OpenVINOImageEncoder.

        Args:
            output_path: Directory to save exported model.
                Creates the directory if it doesn't exist.

        Returns:
            Path to the exported OpenVINO IR directory.

        Example:
            >>> encoder = PyTorchImageEncoder(
            ...     model_id="dinov2_large",
            ...     device="cuda"
            ... )
            >>> ov_path = encoder.export(Path("./exported/dinov2_large"))
            >>>
            >>> # Now load with OpenVINO backend
            >>> ov_encoder = OpenVINOImageEncoder(
            ...     model_path=ov_path,
            ...     input_size=518
            ... )
        """

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {self.model_id} to OpenVINO IR format at {output_path}")

        # Create dummy input that matches the processor output
        dummy_image = [torch.zeros((3, self.input_size, self.input_size))]
        inputs = self.processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        onnx_path = output_path / "image_encoder.onnx"

        # Export to ONNX
        logger.info("Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (inputs,),
                onnx_path,
                input_names=["pixel_values"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "last_hidden_state": {0: "batch_size"},
                },
                opset_version=20,
            )

        # Convert ONNX to OpenVINO IR
        logger.info("Converting to OpenVINO IR...")
        ov_model = openvino.convert_model(onnx_path)

        # Save OpenVINO model
        xml_path = output_path / "image_encoder.xml"
        openvino.save_model(ov_model, xml_path)

        # Save config and processor for later loading
        self.model.config.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        return output_path
