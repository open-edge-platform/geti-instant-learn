# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace backend implementation for ImageEncoder."""

from logging import getLogger
from pathlib import Path

import openvino
import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from transformers import AutoImageProcessor, AutoModel

from getiprompt.utils import precision_to_torch_dtype
from getiprompt.utils.constants import Backend

logger = getLogger("Geti Prompt")

AVAILABLE_IMAGE_ENCODERS = {
    "dinov2_small": ("facebook/dinov2-with-registers-small", "0d9846e56b43a21fa46d7f3f5070f0506a5795a9"),
    "dinov2_base": ("facebook/dinov2-with-registers-base", "a1d738ccfa7ae170945f210395d99dde8adb1805"),
    "dinov2_large": ("facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"),
    "dinov2_giant": ("facebook/dinov2-with-registers-giant", "8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb"),
    "dinov3_small": ("facebook/dinov3-vits16-pretrain-lvd1689m", "114c1379950215c8b35dfcd4e90a5c251dde0d32"),
    "dinov3_small_plus": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", "c93d816fc9e567563bc068f01475bec89cc634a6"),
    "dinov3_base": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "5931719e67bbdb9737e363e781fb0c67687896bc"),
    "dinov3_large": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51"),
    "dinov3_huge": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", "c807c9eeea853df70aec4069e6f56b28ddc82acc"),
}


class HuggingFaceImageEncoder(nn.Module):
    """HuggingFace backend for DINO image encoder.

    This encoder uses a model from HuggingFace to encode images into
    normalized patch embeddings.

    Examples:
        >>> from getiprompt.components.encoders import HuggingFaceImageEncoder
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
        from getiprompt.utils.optimization import optimize_model

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
            model = AutoModel.from_pretrained(model_id, revision=revision) # nosec: B615
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                revision=revision,
                size={"height": input_size, "width": input_size},
                do_center_crop=False,
                use_fast=True,  # uses Rust based image processor
            ) # nosec: B615
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

    def export(self, output_path: Path, backend: str | Backend = Backend.OPENVINO) -> Path:
        """Export this PyTorch encoder to ONNX or OpenVINO IR format.

        This uses direct ONNX export or OpenVINO conversion to export the DINO model.
        The exported model can then be loaded using OpenVINOImageEncoder.

        Args:
            output_path: Directory to save exported model.
                Creates the directory if it doesn't exist.
            backend: The backend to export to. Can be a Backend enum
                (e.g., Backend.ONNX, Backend.OPENVINO) or a string
                (e.g., "onnx", "openvino").

        Returns:
            Path to the exported model file.

        Example:
            >>> encoder = HuggingFaceImageEncoder(
            ...     model_id="dinov2_large",
            ...     device="cuda"
            ... )
            >>> ov_path = encoder.export(Path("./exported"), backend="openvino")
            >>>
            >>> # Now load with OpenVINO backend
            >>> ov_encoder = OpenVINOImageEncoder(
            ...     model_path=ov_path,
            ... )
        """
        # Convert string to Backend enum if needed
        if isinstance(backend, str):
            backend = Backend(backend.lower())

        # Wrapper to export only the feature extraction (without CLS/register tokens)
        class ForwardFeaturesWrapper(nn.Module):
            def __init__(self, model: nn.Module, ignore_token_length: int):
                super().__init__()
                self.model = model
                self.ignore_token_length = ignore_token_length

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                last_hidden_state = self.model(pixel_values).last_hidden_state
                features = last_hidden_state[:, self.ignore_token_length :, :]
                return functional.normalize(features, p=2, dim=-1)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        msg = f"Exporting {self.model_id} to {backend.name} format at {output_path}"
        logger.info(msg)

        # Create dummy input that matches the processor output
        dummy_image = [torch.zeros((3, self.input_size, self.input_size))]
        inputs = self.processor(images=dummy_image, return_tensors="pt")
        dummy_pixel_values = inputs["pixel_values"].to(self.device)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Wrap model to export with token removal and normalization
        export_model = ForwardFeaturesWrapper(self.model, self.ignore_token_length)
        export_model.eval()

        # Define input and output names
        input_names = ["x"]
        output_names = ["features"]

        if backend == Backend.ONNX:
            onnx_path = output_path / "image_encoder.onnx"
            dynamic_axes = {
                "x": {0: "batch_size"},
                "features": {0: "batch_size"},
            }
            with torch.no_grad():
                try:
                    msg = f"Exporting {self.model_id} to ONNX format at {onnx_path}"
                    logger.info(msg)
                    torch.onnx.export(
                        export_model,
                        (dummy_pixel_values,),
                        onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=20,
                    )
                    msg = f"Export complete. Model saved to {onnx_path}"
                    logger.info(msg)
                    return onnx_path
                except Exception as e:
                    msg = f"Error exporting to ONNX: {e}"
                    logger.exception(msg)
                    raise

        if backend == Backend.OPENVINO:
            xml_path = output_path / "image_encoder.xml"
            dynamic_shapes = {
                "x": openvino.PartialShape([-1, 3, self.input_size, self.input_size]),
            }
            try:
                msg = f"Converting {self.model_id} to OpenVINO IR format at {xml_path}"
                logger.info(msg)
                ov_model = openvino.convert_model(
                    input_model=export_model,
                    example_input=(dummy_pixel_values,),
                    input=dynamic_shapes,
                )
                for i, ov_output in enumerate(ov_model.outputs):
                    ov_output.get_tensor().set_names({output_names[i]})

                # Store model configuration as runtime info in the IR
                ov_model.set_rt_info(self.patch_size, ["model_info", "patch_size"])
                ov_model.set_rt_info(self.feature_size, ["model_info", "feature_size"])
                ov_model.set_rt_info(self.ignore_token_length, ["model_info", "ignore_token_length"])
                ov_model.set_rt_info(self.input_size, ["model_info", "input_size"])

                openvino.save_model(ov_model, xml_path)
                msg = f"Export complete. Model saved to {xml_path}"
                logger.info(msg)
                return xml_path
            except Exception as e:
                msg = f"Error exporting to OpenVINO IR: {e}"
                logger.exception(msg)
                raise

        msg = f"Invalid backend: {backend}. Valid backends: ['onnx', 'openvino']"
        raise ValueError(msg)
