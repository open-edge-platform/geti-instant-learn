# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend implementation for SAM predictor."""

from logging import getLogger
from pathlib import Path

import numpy as np
import openvino
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_hq import sam_model_registry
from segment_anything_hq.modeling.prompt_encoder import PositionEmbeddingRandom as _PositionEmbeddingRandom
from segment_anything_hq.modeling.prompt_encoder import PromptEncoder as _PromptEncoder
from segment_anything_hq.predictor import SamPredictor as _SamPredictor
from torch import nn

from getiprompt.data import ResizeLongestSide
from getiprompt.utils.constants import DATA_PATH, MODEL_MAP, Backend, SAMModelName
from getiprompt.utils.utils import download_file

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


class PositionEmbeddingRandom(_PositionEmbeddingRandom):
    """Dtype-aware positional encoding using random spatial frequencies.

    This is a drop-in replacement for segment_anything_hq's PositionEmbeddingRandom
    that preserves the model's dtype (e.g., bfloat16) instead of forcing float32.

    The original implementation hardcodes `coords.to(torch.float)` which causes
    dtype mismatch when the model runs in bfloat16 or float16 precision.

    See Also:
        segment_anything_hq.modeling.prompt_encoder.PositionEmbeddingRandom
    """

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points normalized to [0,1], preserving dtype."""
        # Convert coords to match the gaussian matrix dtype
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)


class PromptEncoder(_PromptEncoder):
    """ONNX-compatible prompt encoder for SAM model export.

    This is a drop-in replacement for segment_anything_hq's PromptEncoder
    that uses ONNX-traceable operations. Key differences:

    - Replaces boolean indexing with element-wise multiplication
    - Uses sentinel values (all-zero tensors) for optional inputs
    - All operations are pure tensor ops (no .item(), no Python conditionals)
    - Uses dtype-aware PositionEmbeddingRandom for bfloat16/float16 support

    This encoder works for both PyTorch inference and ONNX/OpenVINO export.

    See Also:
        segment_anything_hq.modeling.prompt_encoder.PromptEncoder
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        """Initialize with dtype-aware PositionEmbeddingRandom."""
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        # Replace pe_layer with dtype-aware version
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points += 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device, dtype=points.dtype)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device, dtype=labels.dtype)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # Use ONNX-compatible operations instead of boolean indexing
        # Create masks for each label type
        mask_neg1 = (labels == -1).float().unsqueeze(-1)  # [B, N, 1]
        mask_0 = (labels == 0).float().unsqueeze(-1)  # [B, N, 1]
        mask_1 = (labels == 1).float().unsqueeze(-1)  # [B, N, 1]

        # Apply embeddings using element-wise multiplication
        point_embedding *= 1 - mask_neg1  # Zero out -1 labels
        point_embedding += mask_neg1 * self.not_a_point_embed.weight
        point_embedding += mask_0 * self.point_embeddings[0].weight
        point_embedding += mask_1 * self.point_embeddings[1].weight
        return point_embedding

    def _get_dtype(self) -> torch.dtype:
        return self.point_embeddings[0].weight.dtype

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ONNX-traceable forward pass with optional box and mask prompts.

        Uses sentinel values (all-zero tensors) to handle optional inputs:
        - All-zero boxes → skip box embedding (zero out embeddings)
        - All-zero masks → use default no_mask_embed (blend to default)

        All operations are pure tensor operations (no .item(), no Python
        conditionals on tensor values) to ensure ONNX traceability.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device(), dtype=self._get_dtype())

        if points is not None:
            coords, labels = points
            # Always pad points when boxes input exists (even if dummy)
            # The box embeddings will be masked out later if they're dummy
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            # Embed boxes first
            box_embeddings = self._embed_boxes(boxes)

            # Detect dummy boxes: check if all coordinates are zero
            # box shape is typically [B, 4] or [B, 1, 4]
            boxes_flat = boxes.reshape(boxes.shape[0], -1)  # [B, 4] or [B, num_boxes*4]
            boxes_sum = boxes_flat.abs().sum(dim=1, keepdim=True)  # [B, 1]

            # Create mask: 1.0 if boxes are valid (non-zero), 0.0 if dummy (all zeros)
            has_valid_boxes = (boxes_sum > 0).float()  # [B, 1]
            has_valid_boxes = has_valid_boxes.unsqueeze(-1)  # [B, 1, 1]
            has_valid_boxes = has_valid_boxes.expand(-1, box_embeddings.shape[1], -1)  # [B, num_boxes, 1]
            has_valid_boxes = has_valid_boxes.expand_as(box_embeddings)  # [B, num_boxes, embed_dim]

            # Zero out box embeddings for dummy boxes (element-wise multiplication)
            box_embeddings *= has_valid_boxes

            # Concatenate (zeros will be concatenated if boxes were dummy)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            # Compute mask embeddings
            mask_embeddings = self._embed_masks(masks)

            # Detect dummy masks: check if all values are zero
            # mask shape is [B, 1, H, W]
            masks_flat = masks.reshape(masks.shape[0], -1)  # [B, H*W]
            masks_sum = masks_flat.abs().sum(dim=1, keepdim=True)  # [B, 1]

            # Create mask: 1.0 if masks are valid (non-zero), 0.0 if dummy (all zeros)
            has_valid_masks = (masks_sum > 0).to(self._get_dtype())  # [B, 1]

            # Get default "no mask" embedding
            no_mask_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

            # Blend between mask embeddings and no_mask embeddings
            # has_valid_masks: [B, 1] -> [B, 1, 1, 1] -> [B, embed_dim, H, W]
            has_valid_masks = has_valid_masks.view(bs, 1, 1, 1)  # [B, 1, 1, 1]
            has_valid_masks = has_valid_masks.expand_as(mask_embeddings)  # [B, embed_dim, H, W]

            # If masks are valid, use mask_embeddings; otherwise use no_mask_embed
            dense_embeddings = has_valid_masks * mask_embeddings + (1 - has_valid_masks) * no_mask_embed
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings


class PyTorchSAMPredictor(nn.Module):
    """PyTorch implementation of SAM predictor.

    This implementation wraps the original SAM predictor from segment_anything_hq
    and SAM2 predictors, providing a unified interface while delegating to the
    appropriate backend predictor.

    The prompt encoder is patched with an ONNX-compatible version that uses
    element-wise operations instead of boolean indexing, enabling both efficient
    PyTorch inference and seamless export to ONNX/OpenVINO formats.

    Note:
        **Optional Prompts**: When using exported models, boxes and mask_input
        support "not provided" scenarios using sentinel values:

        - **Boxes**: Pass all-zero boxes (e.g., [[0, 0, 0, 0]]) to indicate "no boxes".
          The prompt encoder detects these and zeros out box embeddings.

        - **Mask Input**: Pass all-zero masks (e.g., zeros((B, 1, 256, 256))) to indicate
          "no mask input". The prompt encoder detects these and uses the default no_mask_embed.
    """

    def __init__(
        self,
        sam_model_name: SAMModelName,
        device: str,
        model_path: Path | None = None,
        target_length: int = 1024,
    ) -> None:
        """Initialize SAM predictor.

        Args:
            sam_model_name: The SAM model architecture (e.g., SAM_HQ_TINY, SAM2_BASE)
            device: Device to run inference on ("cuda", "cpu")
            model_path: Path to .pth checkpoint file (optional, auto-downloads if None)
            target_length: Target length for the longest side of the image during transformation. Defaults to 1024.

        Raises:
            NotImplementedError: If the model type is not supported.
        """
        super().__init__()
        self.device = device
        self._sam_model_name = sam_model_name
        self.transform = ResizeLongestSide(target_length)
        self._original_size: tuple[int, int] | None = None

        # Determine checkpoint path
        if model_path is None:
            check_model_weights(sam_model_name)
            model_info = MODEL_MAP[sam_model_name]
            checkpoint_path = DATA_PATH.joinpath(model_info["local_filename"])
        else:
            checkpoint_path = model_path

        msg = f"Loading PyTorch SAM: {sam_model_name} from {checkpoint_path}"
        logger.info(msg)

        # Load model based on type
        if sam_model_name in {
            SAMModelName.SAM2_TINY,
            SAMModelName.SAM2_SMALL,
            SAMModelName.SAM2_BASE,
            SAMModelName.SAM2_LARGE,
        }:
            model_info = MODEL_MAP[sam_model_name]
            config_path = "configs/sam2.1/" + model_info["config_filename"]
            sam_model = build_sam2(config_path, str(checkpoint_path))
            self._predictor = SAM2ImagePredictor(sam_model)
        elif sam_model_name in {SAMModelName.SAM_HQ, SAMModelName.SAM_HQ_TINY}:
            registry_name = MODEL_MAP[sam_model_name]["registry_name"]
            sam_model = sam_model_registry[registry_name]().to(device)
            # suppress - loading the snapshot from the local path
            # nosemgrep trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            state_dict = torch.load(checkpoint_path, map_location=device) # nosec: B614
            info = sam_model.load_state_dict(state_dict, strict=False)
            if info.missing_keys:
                msg = f"Missing keys when loading SAM-HQ model: {info.missing_keys}"
                logger.warning(msg)
            if info.unexpected_keys:
                msg = f"Unexpected keys when loading SAM-HQ model: {info.unexpected_keys}"
                logger.warning(msg)
            sam_model.eval()
            self._predictor = _SamPredictor(sam_model)
            # Patch with ONNX-compatible prompt encoder for SAM-HQ models
            self._patch_prompt_encoder(device)
        else:
            msg = f"Model {sam_model_name} not implemented"
            raise NotImplementedError(msg)

    def _patch_prompt_encoder(self, device: str) -> None:
        """Replace prompt encoder with ONNX-compatible version.

        This patches the SAM-HQ model's prompt encoder to use element-wise
        operations instead of boolean indexing, enabling ONNX/OpenVINO export.
        The patched encoder also supports bfloat16/float16 precision.
        """
        original_encoder = self._predictor.model.prompt_encoder
        patched_encoder = PromptEncoder(
            embed_dim=original_encoder.embed_dim,
            image_embedding_size=original_encoder.image_embedding_size,
            input_image_size=original_encoder.input_image_size,
            mask_in_chans=16,  # It's always 16
        )
        # Load weights from original encoder (preserves original dtype)
        patched_encoder.load_state_dict(original_encoder.state_dict(), strict=True)
        patched_encoder.to(device)
        self._predictor.model.prompt_encoder = patched_encoder

    def set_image(self, image: torch.Tensor) -> None:
        """Set image using PyTorch backend.

        Transforms the image to the target size and delegates to the underlying
        predictor's set_torch_image method, which computes and caches image
        embeddings for efficient inference.

        Args:
            image: Raw image tensor of shape (C, H, W)
        """
        self._original_size = image.shape[-2:]
        transformed_image = self.transform.apply_image_torch(image).to(self.device)
        return self._predictor.set_torch_image(transformed_image, self._original_size)

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

        Transforms point coordinates and boxes to the target image size,
        then delegates to the underlying predictor's predict_torch method.

        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) format (original image coordinates)
            point_labels: Point labels [B, N] (1=foreground, 0=background, -1=padding)
            boxes: Box prompts [B, 4] or [B, 1, 4] in (x1, y1, x2, y2) format (original image coordinates)
            mask_input: Low-res mask input [B, 1, 256, 256]
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of binary masks

        Returns:
            Tuple of (masks, iou_predictions, low_res_logits)

        Raises:
            RuntimeError: If set_image() was not called before predict()
        """
        if self._original_size is None:
            msg = "Must call set_image() before predict()"
            raise RuntimeError(msg)

        # Transform point coordinates to target image size
        transformed_point_coords = None
        if point_coords is not None:
            # point_coords shape: [B, N, 2]
            original_shape = point_coords.shape
            coords_flat = point_coords.reshape(-1, 2)
            transformed_coords = self.transform.apply_coords_torch(coords_flat, self._original_size)
            transformed_point_coords = transformed_coords.reshape(original_shape)

        # Transform boxes to target image size
        transformed_boxes = None
        if boxes is not None:
            # boxes shape: [B, 4] or [B, 1, 4]
            original_shape = boxes.shape
            boxes_flat = boxes.reshape(-1, 4)
            transformed_boxes_flat = self.transform.apply_boxes_torch(boxes_flat, self._original_size)
            transformed_boxes = transformed_boxes_flat.reshape(original_shape)

        return self._predictor.predict_torch(
            point_coords=transformed_point_coords,
            point_labels=point_labels,
            boxes=transformed_boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    @staticmethod
    def _freeze_modules(modules: list[nn.Module]) -> None:
        """Freeze the modules."""
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(requires_grad=False)

    @torch.inference_mode()
    def forward(
        self,
        transformed_image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        boxes: torch.Tensor,
        mask_input: torch.Tensor,
        original_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX/OpenVINO export."""
        h, w = original_size
        original_size = (h, w)
        self._predictor.set_torch_image(transformed_image, original_size)
        return self._predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=False,
        )

    def export(self, output_path: Path, backend: str | Backend = Backend.ONNX) -> Path:
        """Export this PyTorch predictor to the specified format.

        The exported model performs end-to-end inference from preprocessed image
        to segmentation masks, including image encoding, prompt encoding, and mask decoding.

        Args:
            output_path: Directory to save exported models.
                Creates the directory if it doesn't exist.
            backend: The backend format to export to. Can be a Backend enum
                (e.g., Backend.ONNX, Backend.OPENVINO) or a string
                (e.g., "onnx", "openvino").

        Returns:
            Path to the exported model file (.onnx or .xml)

        Example:
            >>> predictor = load_sam_model(
            ...     SAMModelName.SAM_HQ_TINY,
            ...     backend=Backend.PYTORCH
            ... )
            >>> ov_path = predictor.export(Path("./exported"), backend="openvino")
            >>> ov_predictor = load_sam_model(
            ...     SAMModelName.SAM_HQ_TINY,
            ...     backend=Backend.OPENVINO,
            ...     model_path=ov_path
            ... )
        """
        # Convert string to Backend enum if needed
        if isinstance(backend, str):
            backend = Backend(backend.lower())

        # Only SAM-HQ models support export currently
        if self._sam_model_name not in {SAMModelName.SAM_HQ, SAMModelName.SAM_HQ_TINY}:
            msg = f"Export not supported for {self._sam_model_name}. Only SAM-HQ models are supported."
            raise NotImplementedError(msg)

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Move all model components to CPU for export compatibility
        logger.info("Moving SAM model to CPU for export...")
        self._predictor.model.to("cpu")

        self._freeze_modules([
            self._predictor.model.mask_decoder,
            self._predictor.model.prompt_encoder,
            self._predictor.model.image_encoder,
        ])

        input_size = self._predictor.model.image_encoder.img_size

        # Create dummy inputs on CPU to match model device
        dummy_image = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32, device="cpu")
        dummy_coords = torch.rand((10, 1, 2), dtype=torch.float32, device="cpu") * input_size
        dummy_labels = torch.ones((10, 1), dtype=torch.float32, device="cpu")
        dummy_boxes = torch.rand((10, 1, 4), dtype=torch.float32, device="cpu") * input_size
        dummy_mask_input = torch.rand((10, 1, 256, 256), dtype=torch.float32, device="cpu")
        original_size = torch.tensor((input_size, input_size), dtype=torch.int32, device="cpu")

        model_inputs = (
            dummy_image,
            dummy_coords,
            dummy_labels,
            dummy_boxes,
            dummy_mask_input,
            original_size,
        )

        # Define input and output names
        input_names = [
            "transformed_image",
            "point_coords",
            "point_labels",
            "boxes",
            "mask_input",
            "original_size",
        ]
        output_names = ["masks", "iou_predictions", "low_res_logits"]

        if backend == Backend.ONNX:
            # Define dynamic axes
            dynamic_axes = {
                "transformed_image": {0: "batch_size", 2: "height", 3: "width"},
                "point_coords": {0: "num_masks", 1: "num_points"},
                "point_labels": {0: "num_masks", 1: "num_points"},
                "boxes": {0: "num_masks", 1: "num_boxes"},
                "mask_input": {0: "num_masks"},
            }
            with torch.no_grad():
                try:
                    # Export to ONNX
                    torch.onnx.export(
                        self,
                        model_inputs,
                        output_path / "exported_sam.onnx",
                        opset_version=20,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        verbose=True,
                    )
                    return output_path / "exported_sam.onnx"
                except Exception as e:
                    msg = f"Error exporting to ONNX: {e}"
                    logger.exception(msg)
                    raise

        if backend == Backend.OPENVINO:
            dynamic_shapes = {
                "transformed_image": openvino.PartialShape([-1, 3, -1, -1]),
                "point_coords": openvino.PartialShape([-1, -1, 2]),
                "point_labels": openvino.PartialShape([-1, -1]),
                "boxes": openvino.PartialShape([-1, 1, 4]),
                "mask_input": openvino.PartialShape([-1, 1, 256, 256]),
                "original_size": openvino.PartialShape([2]),
            }
            try:
                ov_model = openvino.convert_model(
                    input_model=self,
                    example_input=model_inputs,
                    input=dynamic_shapes,
                )
                for i, ov_output in enumerate(ov_model.outputs):
                    ov_output.get_tensor().set_names({output_names[i]})
                openvino.save_model(ov_model, output_path / "exported_sam.xml")
                return output_path / "exported_sam.xml"
            except Exception as e:
                msg = f"Error exporting to OpenVINO IR: {e}"
                logger.exception(msg)
                raise

        msg = f"Invalid backend: {backend}. Valid backends: ['onnx', 'openvino']"
        raise ValueError(msg)
