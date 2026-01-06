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

from getiprompt.data import ResizeLongestSide
from getiprompt.utils.constants import Backend, SAMModelName
from getiprompt.utils.utils import device_to_openvino_device, precision_to_openvino_type

logger = getLogger("Geti Prompt")


class OpenVINOSAMPredictor(nn.Module):
    """OpenVINO implementation of SAM predictor.

    This implementation uses OpenVINO Runtime for efficient inference on
    Intel hardware (CPU, GPU, VPU). It expects an exported OpenVINO IR model
    (.xml/.bin files) created by PyTorchSAMPredictor.export().
    """

    def __init__(
        self,
        sam_model_name: SAMModelName,
        device: str,
        model_path: Path,
        precision: str = "fp32",
        target_length: int = 1024,
    ) -> None:
        """Initialize SAM predictor.

        Args:
            sam_model_name: The SAM model architecture (e.g., SAM_HQ_TINY, SAM2_BASE)
            device: Device to run inference on ("CPU", "GPU", "AUTO")
            model_path: Path to .xml IR file (required)
            precision: Precision to use for the model
            target_length: Target length for the longest side of the image (for image transform).

        Raises:
            FileNotFoundError: If the model path doesn't exist.
        """
        super().__init__()
        self.device = device
        self.transform = ResizeLongestSide(target_length)

        # Validate model path
        if not model_path.exists():
            msg = (
                f"OpenVINO model not found at {model_path}. "
                f"Please export the model first using PyTorchSAMPredictor.export(). "
                f"Example:\n"
                f"  from getiprompt.components.sam import load_sam_model\n"
                f"  predictor = load_sam_model(SAMModelName.{sam_model_name.name}, backend=Backend.PYTORCH)\n"
                f"  predictor.export(output_path)"
            )
            raise FileNotFoundError(msg)

        msg = f"Loading OpenVINO SAM: {sam_model_name} from {model_path}"
        logger.info(msg)

        ov_device = device_to_openvino_device(device)
        # Load and compile model
        core = ov.Core()
        core.set_property(ov_device, {hint.inference_precision: precision_to_openvino_type(precision)})
        ov_model = core.read_model(model_path)

        # Map device names (PyTorch style -> OpenVINO style)

        self.compiled_model = core.compile_model(ov_model, ov_device)

        # Store state (OpenVINO model does full inference, not separate encoding)
        self._current_image = None
        self._original_size = None

    def set_image(self, image: torch.Tensor) -> None:
        """Set image for OpenVINO inference.

        Transforms the image to the target size. Unlike PyTorch backend which
        computes embeddings here, OpenVINO backend performs full end-to-end
        inference in predict(). This method transforms and stores the image
        and metadata for later use.

        Args:
            image: Raw image tensor of shape (C, H, W)
        """
        self._original_size = image.shape[-2:]
        self._current_image = self.transform.apply_image_torch(image)

    def export(self, export_path: Path, backend: str | Backend = Backend.ONNX) -> None:
        """Dummy export method.

        This is OV SAM predictor implementation for running inference with
        OpenVINO models. To export models, please use the PyTorchSAMPredictor
        backend and call its export() method.

        Args:
            export_path: Path to save the exported model
            backend: Backend format to export to. Can be a Backend enum
                (e.g., Backend.ONNX, Backend.OPENVINO) or a string
                (e.g., "onnx", "openvino").
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

        Transforms point coordinates and boxes to the target image size,
        then performs end-to-end inference from image to masks using the
        compiled OpenVINO model. Converts inputs to numpy, runs inference,
        and converts outputs back to PyTorch tensors.

        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) format (original image coordinates)
            point_labels: Point labels [B, N] (1=foreground, 0=background, -1=padding)
            boxes: Box prompts [B, 4] or [B, 1, 4] in (x1, y1, x2, y2) format (original image coordinates)
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

        # Transform point coordinates to target image size
        original_shape = point_coords.shape
        coords_flat = point_coords.reshape(-1, 2)
        transformed_coords = self.transform.apply_coords_torch(coords_flat, self._original_size)
        transformed_point_coords = transformed_coords.reshape(original_shape)

        # Transform boxes to target image size if provided
        if boxes is not None:
            boxes_original_shape = boxes.shape
            boxes_flat = boxes.reshape(-1, 4)
            transformed_boxes_flat = self.transform.apply_boxes_torch(boxes_flat, self._original_size)
            transformed_boxes = transformed_boxes_flat.reshape(boxes_original_shape)
        else:
            transformed_boxes = None

        # Prepare inputs - convert torch tensors to numpy
        num_masks = len(transformed_point_coords)

        boxes_np = (
            np.zeros((num_masks, 1, 4), dtype=np.float32)
            if transformed_boxes is None
            else transformed_boxes.cpu().numpy()
        )
        mask_input_np = (
            np.zeros((num_masks, 1, 256, 256), dtype=np.float32) if mask_input is None else mask_input.cpu().numpy()
        )

        inputs = {
            "transformed_image": self._current_image.cpu().numpy(),
            "point_coords": transformed_point_coords.cpu().numpy(),
            "point_labels": point_labels.cpu().numpy(),
            "boxes": boxes_np,
            "mask_input": mask_input_np,
            "original_size": np.array(self._original_size),
        }

        # Run OpenVINO inference
        outputs = self.compiled_model(inputs)
        masks = torch.from_numpy(outputs["masks"])
        iou_predictions = torch.from_numpy(outputs["iou_predictions"])
        low_res_logits = torch.from_numpy(outputs["low_res_logits"])
        return masks, iou_predictions, low_res_logits
