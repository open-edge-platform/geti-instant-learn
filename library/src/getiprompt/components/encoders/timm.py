# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image encoder using TIMM models."""

from logging import getLogger
from pathlib import Path

import openvino
import timm
import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype

from getiprompt.utils import precision_to_torch_dtype
from getiprompt.utils.constants import Backend
from getiprompt.models.registry import ModelType, get_model, get_models_by_type

logger = getLogger("Geti Prompt")


class TimmImageEncoder(nn.Module):
    """This encoder uses a model from timm to encode the images.

    Examples:
        >>> from getiprompt.components.encoders.timm import TimmImageEncoder
        >>> from torchvision import tv_tensors
        >>> import torch

        >>> # Create a sample image
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> encoder = TimmImageEncoder(model_id="dinov3-large")
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1, 1369, 1024])
    """

    def __init__(
        self,
        model_id: str = "dinov3-large",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        input_size: int = 512,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_id: The model ID (e.g., "dinov3-large").
            device: The device to use.
            precision: The precision to use.
            compile_models: Whether to compile the models.
            input_size: The input size to use.

        Raises:
            ValueError: If the model ID is invalid or not an encoder type.
        """
        from getiprompt.utils.optimization import optimize_model

        super().__init__()

        # Validate model exists in registry
        model_meta = get_model(model_id)
        if model_meta is None:
            valid = [m.id for m in get_models_by_type(ModelType.ENCODER)]
            msg = f"Invalid model ID: '{model_id}'. Valid encoders: {valid}"
            raise ValueError(msg)
        if model_meta.type != ModelType.ENCODER:
            msg = f"Model '{model_id}' is not an encoder (type: {model_meta.type})"
            raise ValueError(msg)
        if model_meta.hf_model_id is None:
            msg = f"Model '{model_id}' has no hf_model_id configured"
            raise ValueError(msg)

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        msg = f"Loading DINO model {model_id}"
        logger.info(msg)
        self.precision = precision_to_torch_dtype(precision)
        self.model, self.processor = self._load_timm_model(
            model_meta.hf_model_id,
            input_size,
            self.precision,
        )
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.feature_size = self.input_size // self.patch_size
        # Ignore CLS token and register tokens
        self.ignore_token_length = self.model.num_prefix_tokens
        self.model = optimize_model(
            model=self.model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
        ).eval()

    @staticmethod
    def _load_timm_model(model_id: str, input_size: int, precision: torch.dtype) -> tuple[nn.Module, Compose]:
        """Load DINO model from timm with error handling.

        Args:
            model_id: The model id of the model.
            input_size: The size of the input image.

        Returns:
            The model and processor.
        """
        model = timm.create_model(model_id, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        data_config["input_size"] = (3, input_size, input_size)
        processor = Compose([
            ToDtype(dtype=precision, scale=True),
            Resize(size=(input_size, input_size)),
            Normalize(mean=data_config["mean"], std=data_config["std"]),
        ])
        return model, processor

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into patch embeddings.

        Args:
            images(list[tv_tensors.Image]): A list of images.

        Returns:
            torch.Tensor: patch-grid feature tensor of shape (batch_size, num_patches, embedding_dim).
        """
        images = torch.stack([self.processor(image.to(self.device)) for image in images])
        features = self.model.forward_features(images)  # (B, N, D)
        features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
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

        Raises:
            ValueError: If the backend is invalid.

        Example:
            >>> encoder = TimmImageEncoder(
            ...     model_id="dinov3_large",
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

        # Wrapper to export only forward_features (without the classification head)
        class ForwardFeaturesWrapper(nn.Module):
            def __init__(self, model: nn.Module, ignore_token_length: int):
                super().__init__()
                self.model = model
                self.ignore_token_length = ignore_token_length

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.model.forward_features(x)
                features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
                return functional.normalize(features, p=2, dim=-1)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        msg = f"Exporting {self.model_id} to {backend.name} format at {output_path}"
        logger.info(msg)

        # Create dummy input that matches the processor output
        dummy_image = torch.rand((1, 3, self.input_size, self.input_size), device=self.device, dtype=self.precision)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Wrap model to only export forward_features
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
                        (dummy_image,),
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
                    example_input=(dummy_image,),
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
