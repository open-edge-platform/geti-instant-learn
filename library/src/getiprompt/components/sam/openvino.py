# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend implementation for SAM predictor."""

from logging import getLogger
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from openvino.properties import hint
from torch import nn

from getiprompt.utils.constants import Backend, SAMModelName

logger = getLogger("Geti Prompt")


class OpenVINOSAMPredictor(nn.Module):
    """OpenVINO implementation of SAM predictor.

    This implementation uses OpenVINO Runtime for efficient inference on
    Intel hardware (CPU, GPU, VPU). It expects an exported OpenVINO IR model
    (.xml/.bin files) created by ExportableSAMPredictor.
    """

    def __init__(
        self,
        sam_model_name: SAMModelName,
        device: str,
        model_path: Path,
        precision: str = "fp32",
    ) -> None:
        """Initialize SAM predictor.

        Args:
            sam_model_name: The SAM model architecture (e.g., SAM_HQ_TINY, SAM2_BASE)
            device: Device to run inference on ("CPU", "GPU", "AUTO")
            model_path: Path to .xml IR file (required)
            precision: Precision to use for the model

        Raises:
            FileNotFoundError: If the model path doesn't exist.
        """
        super().__init__()
        self.device = device

        # Validate model path
        if not model_path.exists():
            msg = (
                f"OpenVINO model not found at {model_path}. "
                f"Please export the model first using ExportableSAMPredictor.export(). "
                f"Example:\n"
                f"  from getiprompt.components.sam import load_sam_model\n"
                f"  predictor = load_sam_model(SAMModelName.{sam_model_name.name}, backend=Backend.PYTORCH)\n"
                f"  predictor.export(output_path)"
            )
            raise FileNotFoundError(msg)

        msg = f"Loading OpenVINO SAM: {sam_model_name} from {model_path}"
        logger.info(msg)

        ov_device = self._map_device_name(device)
        # Load and compile model
        core = ov.Core()
        core.set_property(ov_device, {hint.inference_precision: self._map_precision_name(precision)})
        ov_model = core.read_model(model_path)

        # Map device names (PyTorch style -> OpenVINO style)

        self.compiled_model = core.compile_model(ov_model, ov_device)

        # Store state (OpenVINO model does full inference, not separate encoding)
        self._current_image = None
        self._original_size = None

    def _map_precision_name(self, precision: str) -> str:
        """Map precision names to OpenVINO precision names.

        Args:
            precision: Precision name in PyTorch style

        Returns:
            Precision name in OpenVINO style
        """
        precision_map = {
            "fp32": ov.Type.f32,
            "fp16": ov.Type.f16,
            "bf16": ov.Type.f16,
        }
        return precision_map.get(precision.lower(), ov.Type.f32)

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

    def set_image(
        self,
        image: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        """Set image for OpenVINO inference.

        Unlike PyTorch backend which computes embeddings here, OpenVINO
        backend performs full end-to-end inference in predict(). This method
        just stores the image and metadata for later use.

        Args:
            image: Preprocessed image tensor of shape (C, H, W)
            original_size: Original image size (H, W) before preprocessing
        """
        self._current_image = image
        self._original_size = original_size

    def export(self, export_path: Path, backend: Backend = Backend.ONNX) -> None:
        """Dummy export method.

        This is OV SAM predictor implementation for running inference with
        OpenVINO models. To export OpenVINO models, please use the
        ExportableSAMPredictor class from PyTorchSAMPredictor backend.

        Args:
            export_path: Path to save the exported model
            backend: Backend format to export to (e.g., "openvino", "onnx")
        """
        msg = "Exporting OpenVINO models is not supported. Please export from PyTorchSAMPredictor."
        raise NotImplementedError(msg)

    def predict(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference using OpenVINO compiled model.

        Performs end-to-end inference from image to masks using the compiled
        OpenVINO model. Converts inputs to numpy, runs inference, and converts
        outputs back to PyTorch tensors.

        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) format
            point_labels: Point labels [B, N] (1=foreground, 0=background, -1=padding)
            boxes: Box prompts [B, 4] in (x1, y1, x2, y2) format
            mask_input: Low-res mask input [B, 1, 256, 256]
            multimask_output: Whether to return multiple masks (currently always True)
            return_logits: Whether to return logits instead of binary masks

        Returns:
            Tuple of (masks, iou_predictions, low_res_logits)

        Raises:
            RuntimeError: If set_image() was not called before predict()
        """
        if self._current_image is None:
            msg = "Must call set_image() before predict()"
            raise RuntimeError(msg)

        # Prepare inputs - convert torch tensors to numpy
        num_masks = len(point_coords)

        # Pass dummy all-zero to satisfy ONNX model input requirements
        # The ExportableSAMPredictor prompt encoder is designed to detect
        # all-zero values and skip embedding, treating them as "no values"
        boxes = np.zeros((num_masks, 1, 4), dtype=np.float32) if boxes is None else boxes.cpu().numpy()
        mask_input = (
            np.zeros((num_masks, 1, 256, 256), dtype=np.float32) if mask_input is None else mask_input.cpu().numpy()
        )

        inputs = {
            "transformed_image": self._current_image.cpu().numpy(),
            "point_coords": point_coords.cpu().numpy(),
            "point_labels": point_labels.cpu().numpy(),
            "boxes": boxes,
            "mask_input": mask_input,
            "original_size": np.array(self._original_size),
        }

        # Run OpenVINO inference
        outputs = self.compiled_model(inputs)
        masks = torch.from_numpy(outputs["masks"])
        iou_predictions = torch.from_numpy(outputs["iou_predictions"])
        low_res_logits = torch.from_numpy(outputs["low_res_logits"])
        return masks, iou_predictions, low_res_logits
