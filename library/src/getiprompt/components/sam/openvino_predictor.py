# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend implementation for SAM predictor."""

from logging import getLogger
from pathlib import Path

import numpy as np
import torch

from getiprompt.utils.constants import DATA_PATH, SAMModelName

from .base_predictor import BaseSAMPredictor

logger = getLogger("Geti Prompt")


class OpenVINOSAMPredictor(BaseSAMPredictor):
    """OpenVINO implementation of SAM predictor.
    
    This implementation uses OpenVINO Runtime for efficient inference on
    Intel hardware (CPU, GPU, VPU). It expects an exported OpenVINO IR model
    (.xml/.bin files) created by ExportableSAMPredictor.
    """
    
    def _initialize_backend(self) -> None:
        """Load OpenVINO IR model.
        
        Loads the compiled OpenVINO model and prepares it for inference.
        If no model path is provided, uses a default path convention based
        on the model name.
        """
        import openvino as ov
        
        # Determine model path
        if self.model_path is None:
            # Use default path convention: DATA_PATH/openvino/{model_name}.xml
            self.model_path = DATA_PATH / "openvino" / f"{self.sam_model_name.value}.xml"
        
        if not self.model_path.exists():
            msg = (
                f"OpenVINO model not found at {self.model_path}. "
                f"Please export the model first using ExportableSAMPredictor.export(). "
                f"Example:\n"
                f"  from getiprompt.components.inference import ExportableSAMPredictor\n"
                f"  from getiprompt.models.foundation import load_sam_model\n"
                f"  predictor = load_sam_model(SAMModelName.{self.sam_model_name.name}, backend='pytorch')\n"
                f"  exportable = ExportableSAMPredictor(predictor)\n"
                f"  exportable.export(output_path)"
            )
            raise FileNotFoundError(msg)
        
        logger.info(f"Loading OpenVINO SAM: {self.sam_model_name} from {self.model_path}")
        
        # Load and compile model
        self.core = ov.Core()
        ov_model = self.core.read_model(self.model_path)
        
        # Map device names (PyTorch style -> OpenVINO style)
        ov_device = self._map_device_name(self._device)
        self.compiled_model = self.core.compile_model(ov_model, ov_device)
        
        # Store state (OpenVINO model does full inference, not separate encoding)
        self._current_image = None
        self._original_size = None
        self._input_size = None
        
        # Create mock model object for backward compatibility with SamDecoder
        # SamDecoder accesses predictor.model.image_encoder.img_size and predictor.model.image_size
        self.model = self._create_mock_model()
        
        logger.info(f"OpenVINO model compiled for device: {ov_device}")
    
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
    
    def _create_mock_model(self):
        """Create a mock model object for backward compatibility.
        
        SamDecoder accesses predictor.model.image_encoder.img_size to determine
        the image size for preprocessing. This mock object provides that interface.
        
        Returns:
            Mock model object with necessary attributes
        """
        class MockImageEncoder:
            """Mock image encoder with img_size attribute."""
            img_size = 1024  # Default SAM image size
        
        class MockModel:
            """Mock model object for OpenVINO backend."""
            def __init__(self, sam_name: SAMModelName):
                self.image_encoder = MockImageEncoder()
                self.image_size = 1024
                # Could be customized based on sam_name if needed
        
        return MockModel(self.sam_model_name)
    
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
        self._input_size = tuple(image.shape[-2:])
    
    def predict(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
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
        # Handle None values by providing default empty arrays
        inputs = {
            "transformed_image": self._current_image.cpu().numpy(),
            "point_coords": (
                point_coords.cpu().numpy() 
                if point_coords is not None 
                else np.zeros((1, 1, 2), dtype=np.float32)
            ),
            "point_labels": (
                point_labels.cpu().numpy() 
                if point_labels is not None 
                else np.zeros((1, 1), dtype=np.int32)
            ),
            "boxes": (
                boxes.cpu().numpy() 
                if boxes is not None 
                else np.zeros((1, 4), dtype=np.float32)
            ),
            "mask_input": (
                mask_input.cpu().numpy() 
                if mask_input is not None 
                else np.zeros((1, 1, 256, 256), dtype=np.float32)
            ),
        }
        
        # Add original_size if the model expects it (some exported models include it)
        try:
            # Check if model has original_size input
            input_names = [inp.get_any_name() for inp in self.compiled_model.inputs]
            if "original_size" in input_names:
                inputs["original_size"] = np.array(self._original_size, dtype=np.int32)
        except Exception:
            pass  # If we can't check inputs, proceed without original_size
        
        # Run OpenVINO inference
        outputs = self.compiled_model(inputs)
        
        # Convert outputs back to torch tensors
        # Output names should match ExportableSAMPredictor: masks, iou_predictions, low_res_logits
        masks = torch.from_numpy(outputs["masks"]).to(self._device)
        iou_predictions = torch.from_numpy(outputs["iou_predictions"]).to(self._device)
        low_res_logits = torch.from_numpy(outputs["low_res_logits"]).to(self._device)
        
        # Apply thresholding if binary masks are requested
        if not return_logits:
            # Match PyTorch behavior: masks > threshold
            masks = masks > 0.0
        
        return masks, iou_predictions, low_res_logits

