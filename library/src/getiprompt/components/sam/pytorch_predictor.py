# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend implementation for SAM predictor."""

from logging import getLogger
from pathlib import Path

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_hq import sam_model_registry
from segment_anything_hq.predictor import SamPredictor

from getiprompt.utils.constants import DATA_PATH, MODEL_MAP, SAMModelName
from getiprompt.utils.utils import download_file

from .base_predictor import BaseSAMPredictor

logger = getLogger("Geti Prompt")


def check_model_weights(model_name: SAMModelName) -> None:
    """Check if model weights exist locally, download if necessary.

    Args:
        model_name: The name of the model.

    Raises:
        ValueError: If the model is not found in MODEL_MAP.
        ValueError: If the model weights are missing.
    """
    if model_name not in MODEL_MAP:
        msg = f"Model '{model_name.value}' not found in MODEL_MAP for weight checking."
        raise ValueError(msg)

    model_info = MODEL_MAP[model_name]
    local_filename = model_info["local_filename"]
    download_url = model_info["download_url"]
    sha_sum = model_info["sha_sum"]

    if not local_filename or not download_url:
        msg = f"Missing 'local_filename' or 'download_url' for {model_name.value} in MODEL_MAP."
        raise ValueError(msg)

    target_path = DATA_PATH.joinpath(local_filename)

    if not target_path.exists():
        msg = f"Model weights for {model_name.value} not found at {target_path}, downloading..."
        logger.info(msg)
        download_file(download_url, target_path, sha_sum)


class PyTorchSAMPredictor(BaseSAMPredictor):
    """PyTorch implementation of SAM predictor.
    
    This implementation wraps the original SAM predictor from segment_anything_hq
    and SAM2 predictors, providing a unified interface while delegating to the
    appropriate backend predictor.
    """
    
    def _initialize_backend(self) -> None:
        """Load PyTorch model from checkpoint.
        
        Loads the appropriate SAM model based on the model name and creates
        the corresponding predictor. Supports both SAM-HQ and SAM2 variants.
        """
        # Determine checkpoint path
        if self.model_path is None:
            # Auto-download if needed
            check_model_weights(self.sam_model_name)
            
            model_info = MODEL_MAP[self.sam_model_name]
            checkpoint_path = DATA_PATH.joinpath(model_info["local_filename"])
        else:
            checkpoint_path = self.model_path
        
        msg = f"Loading PyTorch SAM: {self.sam_model_name} from {checkpoint_path}"
        logger.info(msg)
        
        # Load model based on type
        if self.sam_model_name in {
            SAMModelName.SAM2_TINY,
            SAMModelName.SAM2_SMALL,
            SAMModelName.SAM2_BASE,
            SAMModelName.SAM2_LARGE,
        }:
            model_info = MODEL_MAP[self.sam_model_name]
            config_path = "configs/sam2.1/" + model_info["config_filename"]
            sam_model = build_sam2(config_path, str(checkpoint_path))
            self._predictor = SAM2ImagePredictor(sam_model)
        elif self.sam_model_name in {SAMModelName.SAM_HQ, SAMModelName.SAM_HQ_TINY}:
            registry_name = MODEL_MAP[self.sam_model_name]["registry_name"]
            sam_model = sam_model_registry[registry_name](
                checkpoint=str(checkpoint_path)
            ).to(self._device).eval()
            self._predictor = SamPredictor(sam_model)
            self.target_length = self._predictor.model.image_encoder.img_size
        else:
            msg = f"Model {self.sam_model_name} not implemented"
            raise NotImplementedError(msg)
    
    def set_image(
        self,
        image: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        """Set image using PyTorch backend.
        
        Delegates to the underlying predictor's set_torch_image method,
        which computes and caches image embeddings for efficient inference.
        
        Args:
            image: Preprocessed image tensor of shape (C, H, W)
            original_size: Original image size (H, W) before preprocessing
        """
        return self._predictor.set_torch_image(image, original_size)
    
    def predict(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks using PyTorch backend.
        
        Delegates to the underlying predictor's predict_torch method.
        
        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) format
            point_labels: Point labels [B, N] (1=foreground, 0=background, -1=padding)
            boxes: Box prompts [B, 4] in (x1, y1, x2, y2) format
            mask_input: Low-res mask input [B, 1, 256, 256]
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of binary masks
            
        Returns:
            Tuple of (masks, iou_predictions, low_res_logits)
        """
        return self._predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
    
    def export(self, output_path: Path) -> Path:
        """Export this PyTorch predictor to ONNX and OpenVINO IR format.
        
        This is a convenience method that wraps the predictor in an
        ExportableSAMPredictor and performs the export. The exported
        model can then be loaded using load_sam_model() with backend="openvino".
        
        Args:
            output_path: Directory to save exported models.
                Creates the directory if it doesn't exist.
        
        Returns:
            Path to the exported OpenVINO IR file (.xml)
        
        Example:
            >>> predictor = load_sam_model(
            ...     SAMModelName.SAM_HQ_TINY,
            ...     backend="pytorch"
            ... )
            >>> ov_path = predictor.export(Path("./exported"))
            >>> 
            >>> # Now load with OpenVINO backend
            >>> ov_predictor = load_sam_model(
            ...     SAMModelName.SAM_HQ_TINY,
            ...     backend="openvino",
            ...     model_path=ov_path
            ... )
        """
        from .exportable import ExportableSAMPredictor
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Wrap and export
        exportable = ExportableSAMPredictor(self._predictor)
        exportable.export(output_path)
        
        exported_xml = output_path / "exported_sam.xml"
        logger.info(f"Successfully exported to {exported_xml}")
        
        return exported_xml

